# Copyright Vespa.ai. All rights reserved.

require 'performance/nearest_neighbor/common_sift_gist_base'

class CommonRQBase < CommonSiftGistBase

  DISTANCE_METRIC = "distance_metric"

  # Override to compile RQ-specific generators
  def compile_generators
    @container = vespa.container.values.first
    @container_tmp_bin_dir = @container.create_tmp_bin_dir
    @adminserver_tmp_bin_dir = vespa.adminserver.create_tmp_bin_dir

    # Compile the RQ document and query generators
    vespa.adminserver.execute("g++ -g -O3 -std=c++20 -o #{@adminserver_tmp_bin_dir}/make_docs #{selfdir}make_docs.cpp")
    vespa.adminserver.execute("g++ -g -O3 -std=c++20 -o #{@adminserver_tmp_bin_dir}/make_rq_docs #{selfdir}make_rq_docs.cpp")
    @container.execute("g++ -g -O3 -std=c++20 -o #{@container_tmp_bin_dir}/make_queries #{selfdir}make_queries.cpp")
    @container.execute("g++ -g -O3 -std=c++20 -o #{@container_tmp_bin_dir}/make_rq_queries #{selfdir}make_rq_queries.cpp")
  end

  # Override to generate RQ-encoded query vectors (needs @seed)
  def generate_vectors_for_recall(num_queries_for_recall)
    @query_fvecs_container = @dataset ? @dataset.prepare_query_fvecs(self, @container) : nn_download_file(@query_fvecs, @container)

    # Generate query vectors for RQ
    query_vectors_container = dirs.tmpdir + "rq_query_vectors_container.txt"
    no_rotation_flag = @skip_rotation ? " --no-rotation" : ""
    @container.execute("#{@container_tmp_bin_dir}/make_rq_queries #{@query_fvecs_container} " +
                       "#{@dimensions} #{num_queries_for_recall} #{@seed} --only-vectors q_rq#{no_rotation_flag} > #{query_vectors_container}")

    # Copy query vectors to localhost for recall computation
    @local_query_vectors = dirs.tmpdir + "rq_query_vectors.txt"
    find_and_copy_to_localhost(query_vectors_container, @local_query_vectors)
  end

  # Override to feed RQ-encoded documents
  def feed_and_benchmark(num_documents, label, params = {})
    doc_tensor = params[:doc_tensor] || @doc_tensor || "vec_rq_euclidean"
    operation = params[:operation] || "put"
    start_with_docid = params[:start_with_docid] || 0
    start_with_vector = params[:start_with_vector] || 0
    filter_values = params[:filter_values] || nil

    profiler_start
    base_fvecs_local = @dataset ? @dataset.prepare_base_fvecs(self, vespa.adminserver) : nn_download_file(@base_fvecs, vespa.adminserver)

    # Generate RQ-encoded documents
    no_rotation_flag = @skip_rotation ? " --no-rotation" : ""
    command = "#{@adminserver_tmp_bin_dir}/make_rq_docs #{base_fvecs_local} " +
              "#{@dimensions} #{operation} #{start_with_docid} " +
              "#{start_with_vector} #{start_with_vector + num_documents} " +
              "#{@seed} #{doc_tensor}#{no_rotation_flag}"

    if filter_values != nil && !filter_values.empty?
      command += " [#{filter_values.join(',')}]"
    end

    # Add rotation indicator to label for tracking in results
    rotation_suffix = @skip_rotation ? "-norot" : ""
    label_with_rotation = "#{label}#{rotation_suffix}"

    run_stream_feeder(command, [parameter_filler(TYPE, "feed"), parameter_filler(LABEL, label_with_rotation)], 
                      {:client => :vespa_feed_perf})
    profiler_report("feed")

    print_nni_stats("test", doc_tensor)
  end

  # Override to generate RQ queries and benchmark
  def query_and_benchmark(algorithm, target_hits, explore_hits, params = {})
    filter_percent = params[:filter_percent] || 0
    approximate_threshold = params[:approximate_threshold] || 0.05
    slack = params[:slack] || 0.0
    clients = params[:clients] || 1
    threads_per_search = params[:threads_per_search] || 0
    doc_tensor = params[:doc_tensor] || @doc_tensor || "vec_rq_euclidean"
    query_tensor = params[:query_tensor] || "q_rq"
    distance_metric = params[:distance_metric] || @distance_metric || "rq_euclidean"

    approximate = algorithm == HNSW ? "true" : "false"
    query_file = dirs.tmpdir + get_filename(doc_tensor, approximate, target_hits, explore_hits, filter_percent, nil)

    no_rotation_flag = @skip_rotation ? " --no-rotation" : ""
    @container.execute("#{@container_tmp_bin_dir}/make_rq_queries #{@query_fvecs_container} " +
                       "#{@dimensions} #{@num_queries_for_benchmark} #{@seed} " +
                       "#{doc_tensor} #{query_tensor} #{approximate} #{target_hits} #{explore_hits} " +
                       "#{filter_percent}#{no_rotation_flag} > #{query_file}")

    puts "Generated on container: #{query_file}"

    slack_str = (slack == 0.0) ? "" : "-s#{slack}"
    rotation_str = @skip_rotation ? "-norot" : ""
    label = params[:label] || "#{distance_metric}-#{algorithm}-th#{target_hits}-eh#{explore_hits}-f#{filter_percent}#{slack_str}-n#{clients}-t#{threads_per_search}#{rotation_str}"
    result_file = dirs.tmpdir + "fbench_result.#{label}.txt"

    fillers = [parameter_filler(TYPE, get_type_string(filter_percent, threads_per_search)),
               parameter_filler(LABEL, label),
               parameter_filler(ALGORITHM, algorithm),
               parameter_filler(DISTANCE_METRIC, distance_metric),
               parameter_filler(TARGET_HITS, target_hits),
               parameter_filler(EXPLORE_HITS, explore_hits),
               parameter_filler(FILTER_PERCENT, filter_percent),
               parameter_filler(APPROXIMATE_THRESHOLD, approximate_threshold),
               parameter_filler(SLACK, slack),
               parameter_filler(CLIENTS, clients),
               parameter_filler(THREADS_PER_SEARCH, threads_per_search)]

    profiler_start
    run_fbench2(@container,
                query_file,
                {:runtime => FBENCH_TIME,
                 :clients => clients,
                 :append_str => "&summary=minimal&hits=#{target_hits}&ranking.matching.approximateThreshold=#{approximate_threshold}&ranking.matching.explorationSlack=#{slack}",
                 :result_file => result_file},
                fillers)
    profiler_report(label)
    @container.execute("head -10 #{result_file}")
  end

  def get_rq_sd(dims, metric)
    rq_dims = dims + 16
    metric_short = metric.sub('rq_', '')
    tensor_name = "vec_rq_#{metric_short}"

    <<~SD
      # Copyright Vespa.ai. All rights reserved.
      schema test {
        document test {
          field id type int {
            indexing: attribute | summary
          }
          field filter type array<int> {
            indexing: attribute | summary
            attribute: fast-search
          }
          field #{tensor_name} type tensor<int8>(x[#{rq_dims}]) {
            indexing: attribute | index | summary
            index {
              hnsw {
                max-links-per-node: 16
                neighbors-to-explore-at-insert: 500
              }
            }
            attribute {
              distance-metric: #{metric}
            }
          }
        }
        rank-profile default {
          inputs {
            query(q_rq) tensor<int8>(x[#{rq_dims}])
          }
          first-phase {
            expression: closeness(label,nns)
          }
          approximate-threshold: 0.05
          num-threads-per-search: 1
        }
        #{[1, 2, 4, 8, 16].map { |t| "rank-profile threads-#{t} inherits default { num-threads-per-search: #{t} }" }.join("\n  ")}
        document-summary minimal {
          summary id {}
        }
      }
SD
  end

  def create_app_from_sd_file(sd_file, concurrency = 1.0)
    add_bundle(selfdir + "NearestNeighborRecallSearcher.java")
    searching = Searching.new
    searching.chain(Chain.new("default", "vespa").add(Searcher.new("ai.vespa.test.NearestNeighborRecallSearcher")))
    app = SearchApp.new.sd(sd_file).
      threads_per_search(1).
      container(Container.new("combinedcontainer").
                jvmoptions('-Xms8g -Xmx8g').
                search(searching).
                docproc(DocumentProcessing.new).
                documentapi(ContainerDocumentApi.new)).
      indexing("combinedcontainer")
    app.tune_searchnode({:feeding => {:concurrency => concurrency}})
    app
  end

  # Main test orchestration method for RQ tests
  def run_rq_test(params = {})
    @distance_metric = params[:distance_metric] || (@dataset ? @dataset.metric : "rq_euclidean")
    # Map distance metric to field name (distance-metric is schema-level, not runtime)
    @doc_tensor = "vec_rq_#{@distance_metric.sub('rq_', '')}"
    @query_tensor = "q_rq"
    num_documents = params[:num_documents] || 1_000_000
    num_queries_for_recall = params[:num_queries_for_recall] || 100
    @skip_rotation = params[:skip_rotation] || false

    # Write dynamic schema to temp file
    sd_content = get_rq_sd(@dimensions, @distance_metric)
    sd_file = dirs.tmpdir + "test.sd"
    File.write(sd_file, sd_content)

    app = create_app_from_sd_file(sd_file)
    deploy_app(app)
    start

    compile_generators
    generate_vectors_for_recall(num_queries_for_recall)

    feed_and_benchmark(num_documents, "#{num_documents/1000}K-rq-docs")

    # Brute force baseline for target_hits=10
    query_and_benchmark(BRUTE_FORCE, 10, 0)


    # Full test suite
    run_target_hits_10_tests

    # Brute force baseline for target_hits=100
    query_and_benchmark(BRUTE_FORCE, 100, 0)

    run_target_hits_100_tests

  end

  # Override to ensure correct tensor names are used in recall calculation
  def run_target_hits_10_tests
    [0, 10, 30, 70, 110, 190, 390, 590, 790].each do |explore_hits|
      query_and_benchmark(HNSW, 10, explore_hits)
      calc_recall_for_queries(10, explore_hits, :doc_tensor => @doc_tensor, :query_tensor => @query_tensor)
    end
  end

  # Override to ensure correct tensor names are used in recall calculation
  def run_target_hits_100_tests
    [0, 20, 100, 300, 500, 700].each do |explore_hits|
      query_and_benchmark(HNSW, 100, explore_hits)
      calc_recall_for_queries(100, explore_hits, :doc_tensor => @doc_tensor, :query_tensor => @query_tensor)
    end
  end

end
