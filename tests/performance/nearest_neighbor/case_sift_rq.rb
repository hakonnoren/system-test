# Copyright Vespa.ai. All rights reserved.

require 'performance/nearest_neighbor/common_rq_base'

class AnnSiftRQPerfTest < CommonRQBase

  def initialize(*args)
    super(*args)
    @data_path = "sift-data/"
    @base_fvecs = @data_path + "sift_base.fvecs"
    @query_fvecs = @data_path + "sift_query.fvecs"
    @dimensions = 128
    @seed = 42
    @num_queries_for_benchmark = 10000
  end

  def setup
    super
    set_owner("haakno")
  end

  if true
    def test_sift_rq_euclidean
      set_description("Test RQ Euclidean distance performance and recall on SIFT 1M dataset")
      run_rq_test(:distance_metric => "rq_euclidean")
    end
  end
  if false
    def test_sift_rq_angular
      set_description("Test RQ Angular distance performance and recall on SIFT 1M dataset")
      run_rq_test(:distance_metric => "rq_angular")
    end
  end

  if false
    # Quick test with reduced dataset for development
    def test_sift_rq_euclidean_quick
      set_description("Quick test of RQ Euclidean on reduced SIFT dataset")
      @num_queries_for_benchmark = 100
      run_rq_test(
        :distance_metric => "rq_euclidean",
        :num_documents => 20_000,
        :num_queries_for_recall => 10,
        :quick => true  # Skip some expensive tests
      )
    end
  end
end
