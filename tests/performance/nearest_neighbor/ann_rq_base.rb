# Copyright Vespa.ai. All rights reserved.

require 'app_generator/search_app'
require 'performance/fbench'
require 'performance/nearest_neighbor/common_ann_base'

class AnnRQBase < CommonAnnBaseTest

  FBENCH_TIME = 10
  RQ_ALGORITHM = "rq_algorithm"
  DISTANCE_METRIC = "distance_metric"

  def create_rq_app(test_folder, concurrency = nil, threads_per_search = 1)
    add_bundle(selfdir + "NearestNeighborRecallSearcher.java")
    searching = Searching.new
    searching.chain(Chain.new("default", "vespa").add(Searcher.new("ai.vespa.test.NearestNeighborRecallSearcher")))
    app = SearchApp.new.sd(selfdir + test_folder + "/test.sd").
      search_dir(selfdir + test_folder + "/search").
      threads_per_search(threads_per_search).
      container(Container.new("combinedcontainer").
                jvmoptions('-Xms8g -Xmx8g').
                search(searching).
                docproc(DocumentProcessing.new).
                documentapi(ContainerDocumentApi.new)).
      indexing("combinedcontainer")
    if (concurrency != nil)
      app.tune_searchnode({:feeding => {:concurrency => concurrency}})
    end
    return app
  end

  def get_rq_type_string(filter_percent, threads_per_search)
    if threads_per_search > 0
      return "rq_query_threads"
    elsif filter_percent == 0
      return "rq_query"
    else
      return "rq_query_filter"
    end
  end

  def compile_rq_generators
    @container = vespa.container.values.first
    @container_tmp_bin_dir = @container.create_tmp_bin_dir
    @adminserver_tmp_bin_dir = vespa.adminserver.create_tmp_bin_dir

    # Compile the RQ document and query generators
    vespa.adminserver.execute("g++ -g -O3 -std=c++20 -o #{@adminserver_tmp_bin_dir}/make_rq_docs #{selfdir}make_rq_docs.cpp")
    @container.execute("g++ -g -O3 -std=c++20 -o #{@container_tmp_bin_dir}/make_rq_queries #{selfdir}make_rq_queries.cpp")
  end

  def generate_rq_vectors_for_recall(num_queries_for_recall)
    @query_fvecs_container = nn_download_file(@query_fvecs, @container)

    # Generate query vectors for RQ
    query_vectors_container = dirs.tmpdir + "rq_query_vectors_container.txt"
    @container.execute("#{@container_tmp_bin_dir}/make_rq_queries #{@query_fvecs_container} " +
                       "#{@dimensions} #{num_queries_for_recall} #{@seed} --only-vectors q_rq > #{query_vectors_container}")

    # Copy query vectors to localhost for recall computation
    @local_rq_query_vectors = dirs.tmpdir + "rq_query_vectors.txt"
    find_and_copy_to_localhost(query_vectors_container, @local_rq_query_vectors)

    # Also generate float query vectors for ground truth comparison
    float_query_vectors_container = dirs.tmpdir + "float_query_vectors_container.txt"
    @container.execute("#{@container_tmp_bin_dir}/make_rq_queries #{@query_fvecs_container} " +
                    "#{@dimensions} #{num_queries_for_recall} #{@seed} --only-float-vectors dummy > #{float_query_vectors_container}")


    @local_float_query_vectors = dirs.tmpdir + "float_query_vectors.txt"
    find_and_copy_to_localhost(float_query_vectors_container, @local_float_query_vectors)
  end

  def find_and_copy_to_localhost(remote_file, local_file)
    copied = false
    vespa.nodeproxies.each_value do |node|
      if node.file?(remote_file)
        node.copy_remote_file_to_local_file(remote_file, local_file)
        copied = true
        break
      end
    end
    assert(copied, "Failed to copy #{remote_file} to localhost")
  end

  def feed_rq_documents(num_documents, label, params = {})
    # Support comma-separated field names for multi-field feeding
    rq_tensors = params[:rq_tensors] || params[:rq_tensor] || "vec_rq_euclidean"
    float_tensors = params[:float_tensors] || params[:float_tensor] || "vec_float"
    operation = params[:operation] || "put"
    start_with_docid = params[:start_with_docid] || 0
    start_with_vector = params[:start_with_vector] || 0
    filter_values = params[:filter_values] || nil

    profiler_start
    base_fvecs_local = nn_download_file(@base_fvecs, vespa.adminserver)

    # Generate RQ-encoded documents with both RQ and float tensors
    # Field names can be comma-separated for multi-field output
    command = "#{@adminserver_tmp_bin_dir}/make_rq_docs #{base_fvecs_local} " +
              "#{@dimensions} #{operation} #{start_with_docid} " +
              "#{start_with_vector} #{start_with_vector + num_documents} " +
              "#{@seed} #{rq_tensors} #{float_tensors}"

    if filter_values != nil && !filter_values.empty?
      command += " [#{filter_values.join(',')}]"
    end

    run_stream_feeder(command, [parameter_filler(TYPE, "feed"), parameter_filler(LABEL, label)])
    profiler_report("feed")

    # Print stats for first RQ tensor
    first_rq_tensor = rq_tensors.split(',').first
    print_nni_stats("test", first_rq_tensor)
  end

  def get_rq_filename(doc_tensor, approximate, target_hits, explore_hits, filter_percent)
    filter_str = (filter_percent == 0) ? "" : ".f-#{filter_percent}"
    "rq_queries.#{doc_tensor}.ap-#{approximate}.th-#{target_hits}.eh-#{explore_hits}#{filter_str}.txt"
  end

  def query_and_benchmark_rq(algorithm, target_hits, explore_hits, params = {})
    filter_percent = params[:filter_percent] || 0
    approximate_threshold = params[:approximate_threshold] || 0.05
    clients = params[:clients] || 1
    threads_per_search = params[:threads_per_search] || 0
    doc_tensor = params[:rq_tensor] || "vec_rq_euclidean"
    query_tensor = params[:query_tensor] || "q_rq"
    distance_metric = params[:distance_metric] || "rq_euclidean"

    approximate = algorithm == HNSW ? "true" : "false"
    query_file = dirs.tmpdir + get_rq_filename(doc_tensor, approximate, target_hits, explore_hits, filter_percent)

    @container.execute("#{@container_tmp_bin_dir}/make_rq_queries #{@query_fvecs_container} " +
                       "#{@dimensions} #{@num_queries_for_benchmark} #{@seed} " +
                       "#{doc_tensor} #{query_tensor} #{approximate} #{target_hits} #{explore_hits} " +
                       "#{filter_percent} > #{query_file}")

    puts "Generated on container: #{query_file}"

    label = params[:label] || "#{distance_metric}-#{algorithm}-th#{target_hits}-eh#{explore_hits}-f#{filter_percent}-n#{clients}-t#{threads_per_search}"
    result_file = dirs.tmpdir + "fbench_result.#{label}.txt"

    fillers = [parameter_filler(TYPE, get_rq_type_string(filter_percent, threads_per_search)),
               parameter_filler(LABEL, label),
               parameter_filler(ALGORITHM, algorithm),
               parameter_filler(DISTANCE_METRIC, distance_metric),
               parameter_filler(TARGET_HITS, target_hits),
               parameter_filler(EXPLORE_HITS, explore_hits),
               parameter_filler(FILTER_PERCENT, filter_percent),
               parameter_filler(APPROXIMATE_THRESHOLD, approximate_threshold),
               parameter_filler(CLIENTS, clients),
               parameter_filler(THREADS_PER_SEARCH, threads_per_search)]

    profiler_start
    run_fbench2(@container,
                query_file,
                {:runtime => FBENCH_TIME,
                 :clients => clients,
                 :append_str => "&summary=minimal&hits=#{target_hits}&ranking.matching.approximateThreshold=#{approximate_threshold}",
                 :result_file => result_file},
                fillers)
    profiler_report(label)
    @container.execute("head -10 #{result_file}")
  end

  def query_and_benchmark_float(algorithm, target_hits, explore_hits, params = {})
    filter_percent = params[:filter_percent] || 0
    approximate_threshold = params[:approximate_threshold] || 0.05
    clients = params[:clients] || 1
    doc_tensor = params[:float_tensor] || "vec_float"
    query_tensor = params[:query_tensor] || "q_float"

    approximate = algorithm == HNSW ? "true" : "false"
    query_file = dirs.tmpdir + "float_queries.#{doc_tensor}.ap-#{approximate}.th-#{target_hits}.txt"

    # Use standard make_queries for float vectors (same format as existing ANN tests)
    @container.execute("#{@container_tmp_bin_dir}/make_rq_queries #{@query_fvecs_container} " +
                       "#{@dimensions} #{@num_queries_for_benchmark} #{@seed} " +
                       "#{doc_tensor} #{query_tensor} #{approximate} #{target_hits} #{explore_hits} " +
                       "#{filter_percent} > #{query_file}")

    puts "Generated on container: #{query_file}"

    label = params[:label] || "float-#{algorithm}-th#{target_hits}-eh#{explore_hits}-f#{filter_percent}-n#{clients}"
    result_file = dirs.tmpdir + "fbench_result.#{label}.txt"

    fillers = [parameter_filler(TYPE, "float_query"),
               parameter_filler(LABEL, label),
               parameter_filler(ALGORITHM, algorithm),
               parameter_filler(TARGET_HITS, target_hits),
               parameter_filler(EXPLORE_HITS, explore_hits),
               parameter_filler(FILTER_PERCENT, filter_percent),
               parameter_filler(APPROXIMATE_THRESHOLD, approximate_threshold),
               parameter_filler(CLIENTS, clients)]

    profiler_start
    run_fbench2(@container,
                query_file,
                {:runtime => FBENCH_TIME,
                 :clients => clients,
                 :append_str => "&summary=minimal&hits=#{target_hits}&ranking=float-exact&ranking.matching.approximateThreshold=#{approximate_threshold}",
                 :result_file => result_file},
                fillers)
    profiler_report(label)
    @container.execute("head -10 #{result_file}")
  end

  def calc_rq_recall_for_queries(target_hits, explore_hits, params = {})
    filter_percent = params[:filter_percent] || 0
    approximate_threshold = params[:approximate_threshold] || 0.05
    doc_tensor = params[:rq_tensor] || "vec_rq_euclidean"
    query_tensor = params[:query_tensor] || "q_rq"
    distance_metric = params[:distance_metric] || "rq_euclidean"

    puts "calc_rq_recall_for_queries: target_hits=#{target_hits}, explore_hits=#{explore_hits}, " +
         "filter_percent=#{filter_percent}, doc_tensor=#{doc_tensor}, distance_metric=#{distance_metric}"

    result = RecallResult.new(target_hits)
    query_data = []
    num_threads = 5

    File.open(@local_rq_query_vectors, "r").each do |vector|
      query_data.push(vector.strip)
    end

    batch_size = (query_data.size.to_f / num_threads.to_f).ceil
    batches = query_data.each_slice(batch_size).to_a

    threads = []
    for i in 0...num_threads
      threads << Thread.new(batches[i]) do |batch|
        calc_rq_recall_for_query_batch(target_hits, explore_hits, filter_percent,
                                        approximate_threshold, batch, result,
                                        doc_tensor, query_tensor)
      end
    end
    threads.each(&:join)

    puts "RQ recall: avg=#{result.avg}, median=#{result.median}, min=#{result.min}, max=#{result.max}"

    label = params[:label] || "#{distance_metric}-th#{target_hits}-eh#{explore_hits}-f#{filter_percent}"
    write_report([parameter_filler(TYPE, "rq_recall"),
                  parameter_filler(LABEL, label),
                  parameter_filler(DISTANCE_METRIC, distance_metric),
                  parameter_filler(TARGET_HITS, target_hits),
                  parameter_filler(EXPLORE_HITS, explore_hits),
                  parameter_filler(FILTER_PERCENT, filter_percent),
                  parameter_filler(APPROXIMATE_THRESHOLD, approximate_threshold),
                  metric_filler(RECALL_AVG, result.avg),
                  metric_filler(RECALL_MEDIAN, result.median)])
  end

  def calc_rq_recall_for_query_batch(target_hits, explore_hits, filter_percent,
                                      approximate_threshold, batch, result,
                                      doc_tensor, query_tensor)
    batch.each do |vector|
      raw_recall = calc_rq_recall_in_searcher(target_hits, explore_hits, filter_percent,
                                               approximate_threshold, vector,
                                               doc_tensor, query_tensor)
      result.add(raw_recall)
    end
  end

  def calc_rq_recall_in_searcher(target_hits, explore_hits, filter_percent,
                                  approximate_threshold, vector,
                                  doc_tensor, query_tensor)
    query = "query=sddocname:test&summary=minimal&ranking.features.query(#{query_tensor})=#{vector}" +
            "&nnr.enable=true&nnr.docTensor=#{doc_tensor}&nnr.targetHits=#{target_hits}" +
            "&nnr.exploreHits=#{explore_hits}&nnr.filterPercent=#{filter_percent}" +
            "&nnr.approximateThreshold=#{approximate_threshold}&nnr.queryTensor=#{query_tensor}"

    result = search_with_timeout(20, query)
    assert_hitcount(result, 1)
    hit = result.hit[0]
    recall = hit.field["recall"]
    if recall == nil
      error = hit.field["error"]
      assert(false, "Error while calculating recall for query: #{error}")
    end
    recall.to_i
  end

  # Compare RQ results with exact float results
  def calc_rq_vs_float_recall(target_hits, explore_hits, params = {})
    rq_tensor = params[:rq_tensor] || "vec_rq_euclidean"
    float_tensor = params[:float_tensor] || "vec_float"
    rq_query_tensor = params[:rq_query_tensor] || "q_rq"
    float_query_tensor = params[:float_query_tensor] || "q_float"
    distance_metric = params[:distance_metric] || "rq_euclidean"

    puts "calc_rq_vs_float_recall: comparing #{rq_tensor} with #{float_tensor}"

    rq_vectors = []
    float_vectors = []

    File.open(@local_rq_query_vectors, "r").each { |v| rq_vectors.push(v.strip) }
    File.open(@local_float_query_vectors, "r").each { |v| float_vectors.push(v.strip) }

    assert_equal(rq_vectors.length, float_vectors.length, "RQ and float query vector counts must match")

    total_overlap = 0
    num_queries = [rq_vectors.length, 100].min  # Limit for performance

    num_queries.times do |i|
      # Get RQ HNSW results
      rq_query = "yql=select id from test where {targetHits:#{target_hits}}nearestNeighbor(#{rq_tensor},#{rq_query_tensor})" +
                 "&hits=#{target_hits}&ranking.features.query(#{rq_query_tensor})=#{rq_vectors[i]}"
      rq_result = search(rq_query)

      # Get exact float results (brute force)
      float_query = "yql=select id from test where {targetHits:#{target_hits},approximate:false}nearestNeighbor(#{float_tensor},#{float_query_tensor})" +
                    "&hits=#{target_hits}&ranking=float-exact&ranking.features.query(#{float_query_tensor})=#{float_vectors[i]}"
      float_result = search(float_query)

      # Calculate overlap
      rq_ids = rq_result.hit.map { |h| h.field["id"] }.to_set
      float_ids = float_result.hit.map { |h| h.field["id"] }.to_set
      overlap = (rq_ids & float_ids).size
      total_overlap += overlap
    end

    recall_vs_float = (total_overlap.to_f / (num_queries * target_hits)) * 100
    puts "RQ vs Float recall@#{target_hits}: #{recall_vs_float}%"

    label = "#{distance_metric}-vs-float-th#{target_hits}"
    write_report([parameter_filler(TYPE, "rq_vs_float_recall"),
                  parameter_filler(LABEL, label),
                  parameter_filler(DISTANCE_METRIC, distance_metric),
                  parameter_filler(TARGET_HITS, target_hits),
                  metric_filler("recall_vs_float", recall_vs_float)])

    recall_vs_float
  end

end
