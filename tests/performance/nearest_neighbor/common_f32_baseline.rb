# Copyright Vespa.ai. All rights reserved.

require 'performance/nearest_neighbor/common_sift_gist_base'

# Module for running float32 baseline comparisons alongside RQ tests
module CommonF32Baseline
  # Reference constants from CommonAnnBaseTest
  TYPE = CommonAnnBaseTest::TYPE
  LABEL = CommonAnnBaseTest::LABEL
  ALGORITHM = CommonAnnBaseTest::ALGORITHM
  TARGET_HITS = CommonAnnBaseTest::TARGET_HITS
  EXPLORE_HITS = CommonAnnBaseTest::EXPLORE_HITS
  FILTER_PERCENT = CommonAnnBaseTest::FILTER_PERCENT
  APPROXIMATE_THRESHOLD = CommonAnnBaseTest::APPROXIMATE_THRESHOLD
  SLACK = CommonAnnBaseTest::SLACK
  HNSW = CommonAnnBaseTest::HNSW
  BRUTE_FORCE = CommonAnnBaseTest::BRUTE_FORCE
  CLIENTS = CommonAnnBaseTest::CLIENTS
  THREADS_PER_SEARCH = CommonAnnBaseTest::THREADS_PER_SEARCH
  DISTANCE_METRIC = "distance_metric"
  FBENCH_TIME = 10

  def run_f32_baseline_test(params = {})
    # For float32, we use clean metric names (euclidean, angular, dotproduct)
    metric = params[:distance_metric] || (@dataset ? @dataset.metric.sub('rq_', '') : "euclidean")
    @doc_tensor = "vec_f32"
    @query_tensor = "q_vec"  # must match make_queries.cpp which hardcodes q_vec
    num_documents = params[:num_documents] || 1_000_000
    num_queries_for_recall = params[:num_queries_for_recall] || 100

    # Write dynamic schema to temp file
    sd_content = get_f32_sd(@dimensions, metric)
    sd_file = dirs.tmpdir + "test.sd"
    File.write(sd_file, sd_content)

    app = create_app_from_sd_file(sd_file)
    
    add_bundle(selfdir + "NearestNeighborRecallSearcher.java")
    deploy_app(app)
    start

    compile_generators
    
    profiler_start
    base_fvecs_local = @dataset ? @dataset.prepare_base_fvecs(self, vespa.adminserver) : nn_download_file(@base_fvecs, vespa.adminserver)
    command = "#{@adminserver_tmp_bin_dir}/make_docs #{base_fvecs_local} #{@dimensions} put 0 0 #{num_documents} [] [0,-1] [0,-1] false #{@doc_tensor}"
    run_stream_feeder(command, [parameter_filler(TYPE, "feed"), parameter_filler(LABEL, "f32-baseline")],
                      {:client => :vespa_feed_perf})
    profiler_report("feed_f32")
    print_nni_stats("test", @doc_tensor)

    # Generate query vectors for recall calculation
    generate_f32_vectors_for_recall(num_queries_for_recall)

    # Run full test suite to match RQ tests
    # Brute force baseline for target_hits=10
    query_and_benchmark_f32(BRUTE_FORCE, 10, 0, metric)

    # Full HNSW test suite for target_hits=10
    run_f32_target_hits_10_tests(metric)

    # Brute force baseline for target_hits=100
    query_and_benchmark_f32(BRUTE_FORCE, 100, 0, metric)

    # Full HNSW test suite for target_hits=100
    run_f32_target_hits_100_tests(metric)
  end

  def query_and_benchmark_f32(algorithm, target_hits, explore_hits, metric)
    @query_fvecs_container ||= @dataset ? @dataset.prepare_query_fvecs(self, @container) : nn_download_file(@query_fvecs, @container)
    
    approximate = algorithm == HNSW ? "true" : "false"
    query_file = dirs.tmpdir + "queries.f32-th#{target_hits}-eh#{explore_hits}.txt"
    # make_queries args: file dims num_queries doc_tensor approximate target_hits explore_hits filter_percent
    @container.execute("#{@container_tmp_bin_dir}/make_queries #{@query_fvecs_container} #{@dimensions} #{@num_queries_for_benchmark} #{@doc_tensor} #{approximate} #{target_hits} #{explore_hits} 0 > #{query_file}")
    
    label = "f32-#{metric}-#{algorithm}-th#{target_hits}-eh#{explore_hits}"
    
    fillers = [parameter_filler(TYPE, "query"),
               parameter_filler(LABEL, label),
               parameter_filler(ALGORITHM, algorithm),
               parameter_filler(DISTANCE_METRIC, metric),
               parameter_filler(TARGET_HITS, target_hits),
               parameter_filler(EXPLORE_HITS, explore_hits),
               parameter_filler(FILTER_PERCENT, 0),
               parameter_filler(APPROXIMATE_THRESHOLD, 0.05),
               parameter_filler(SLACK, 0.0),
               parameter_filler(CLIENTS, 1),
               parameter_filler(THREADS_PER_SEARCH, 0)]
    
    profiler_start
    run_fbench2(@container, query_file, 
                {:runtime => FBENCH_TIME, :clients => 1, :append_str => "&summary=minimal&hits=#{target_hits}&ranking.matching.approximateThreshold=0.05"}, 
                fillers)
    profiler_report(label)
  end

  def generate_f32_vectors_for_recall(num_queries)
    @query_fvecs_container = @dataset ? @dataset.prepare_query_fvecs(self, @container) : nn_download_file(@query_fvecs, @container)
    
    query_vectors_container = dirs.tmpdir + "f32_query_vectors_container.txt"
    # make_queries outputs only vectors when given just 3 args: file, dims, num_queries
    @container.execute("#{@container_tmp_bin_dir}/make_queries #{@query_fvecs_container} " +
                       "#{@dimensions} #{num_queries} > #{query_vectors_container}")
    
    @local_query_vectors = dirs.tmpdir + "f32_query_vectors.txt"
    find_and_copy_to_localhost(query_vectors_container, @local_query_vectors)
  end

  def get_f32_sd(dims, metric)
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
          field vec_f32 type tensor<float>(x[#{dims}]) {
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
            query(q_vec) tensor<float>(x[#{dims}])
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

  def run_f32_target_hits_10_tests(metric)
    [0, 10, 30, 70, 110, 190, 390, 590, 790].each do |explore_hits|
      query_and_benchmark_f32(HNSW, 10, explore_hits, metric)
      calc_recall_for_queries(10, explore_hits, :doc_type => "test", :doc_tensor => @doc_tensor, :query_tensor => @query_tensor)
    end
  end

  def run_f32_target_hits_100_tests(metric)
    [0, 20, 100, 300, 500, 700].each do |explore_hits|
      query_and_benchmark_f32(HNSW, 100, explore_hits, metric)
      calc_recall_for_queries(100, explore_hits, :doc_type => "test", :doc_tensor => @doc_tensor, :query_tensor => @query_tensor)
    end
  end

end
