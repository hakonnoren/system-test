# Copyright Vespa.ai. All rights reserved.

require 'performance/nearest_neighbor/ann_rq_base'

class AnnSiftRQPerfTest < AnnRQBase

  def initialize(*args)
    super(*args)
    @data_path = "sift-data/"
    @base_fvecs = @data_path + "sift_base.fvecs"
    @query_fvecs = @data_path + "sift_query.fvecs"
    @dimensions = 128
    @packed_dimensions = 144  # 128 codes + 16 metadata bytes
    @seed = 42  # Match Java implementation seed
    @num_queries_for_benchmark = 10000
  end

  def setup
    super
    set_owner("haakno")
  end

  # Main test: Compare RQ Euclidean HNSW with Float HNSW
  def test_sift_rq_euclidean
    set_description("Test RQ Euclidean distance performance and recall on SIFT 1M dataset")

    deploy_app(create_rq_app("rq_test", 0.3, 1))
    start

    num_queries_for_recall = 100
    num_documents = 1_000_000

    # For quick testing, use smaller values:
    # num_queries_for_recall = 10
    # num_documents = 50_000

    compile_rq_generators
    generate_rq_vectors_for_recall(num_queries_for_recall)

    # Feed documents with all fields (RQ euclidean, RQ angular, float, float HNSW)
    feed_rq_documents(num_documents, "1M-rq-docs", {
      :rq_tensors => "vec_rq_euclidean,vec_rq_angular",
      :float_tensors => "vec_float,vec_float_hnsw"
    })

    # Benchmark: Exact float search (brute force - ground truth)
    puts "=== Ground Truth: Float Brute Force ==="
    query_and_benchmark_rq(BRUTE_FORCE, 100, 0, {
      :rq_tensor => "vec_float",
      :query_tensor => "q_float",
      :distance_metric => "float-exact"
    })

    # Benchmark: RQ Euclidean HNSW with various explore_hits values
    puts "=== RQ Euclidean HNSW ==="
    [0, 100, 300].each do |explore_hits|
      query_and_benchmark_rq(HNSW, 100, explore_hits, {
        :rq_tensor => "vec_rq_euclidean",
        :distance_metric => "rq_euclidean"
      })
    end

    # Calculate recall: RQ HNSW vs brute force
    puts "=== RQ Recall (HNSW vs Brute Force) ==="
    [0, 100, 300].each do |explore_hits|
      calc_rq_recall_for_queries(100, explore_hits, {
        :rq_tensor => "vec_rq_euclidean",
        :distance_metric => "rq_euclidean"
      })
    end

    # Calculate recall: RQ vs Float (how well does RQ approximate exact float distances)
    puts "=== RQ vs Float Recall ==="
    calc_rq_vs_float_recall(100, 0, {
      :rq_tensor => "vec_rq_euclidean",
      :float_tensor => "vec_float",
      :distance_metric => "rq_euclidean"
    })
  end

  # Test RQ Angular distance
  def test_sift_rq_angular
    set_description("Test RQ Angular distance performance and recall on SIFT 1M dataset")

    deploy_app(create_rq_app("rq_test", 0.3, 1))
    start

    num_queries_for_recall = 100
    num_documents = 1_000_000

    compile_rq_generators
    generate_rq_vectors_for_recall(num_queries_for_recall)

    feed_rq_documents(num_documents, "1M-rq-angular", {
      :rq_tensors => "vec_rq_angular",
      :float_tensors => "vec_float"
    })

    # Benchmark: RQ Angular HNSW
    puts "=== RQ Angular HNSW ==="
    [0, 100, 300].each do |explore_hits|
      query_and_benchmark_rq(HNSW, 100, explore_hits, {
        :rq_tensor => "vec_rq_angular",
        :distance_metric => "rq_angular"
      })
    end

    # Calculate recall
    puts "=== RQ Angular Recall ==="
    [0, 100].each do |explore_hits|
      calc_rq_recall_for_queries(100, explore_hits, {
        :rq_tensor => "vec_rq_angular",
        :distance_metric => "rq_angular"
      })
    end
  end

end
