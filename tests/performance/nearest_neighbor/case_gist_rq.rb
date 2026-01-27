# Copyright Vespa.ai. All rights reserved.

require 'performance/nearest_neighbor/common_rq_base'
require 'performance/nearest_neighbor/common_f32_baseline'
require 'performance/nearest_neighbor/ann_dataset_manager'

class AnnGistRQPerfTest < CommonRQBase
  include CommonF32Baseline

  def initialize(*args)
    super(*args)
    @dataset = AnnDatasetManager.new("gist")
    @base_fvecs = @dataset.base_file
    @query_fvecs = @dataset.query_file
    @dimensions = @dataset.dims
    @seed = 42
    @num_queries_for_benchmark = 1000  # GIST uses fewer queries due to higher dimensionality
  end

  def setup
    super
    set_owner("haakno")
    @perf_recording = "all"
  end

  def test_gist_rq_no_rotation
    set_description("Test RQ Euclidean without rotation step (baseline for rotation impact)")
    run_rq_test(:distance_metric => "rq_euclidean", :num_documents => @dataset.num_docs, :skip_rotation => true)
  end

  def test_gist_f32_baseline
    set_description("Test F32 Euclidean distance performance on GIST dataset")
    run_f32_baseline_test(:distance_metric => "euclidean", :num_documents => @dataset.num_docs)
  end

  def test_gist_rq_euclidean
    set_description("Test RQ Euclidean distance performance and recall on GIST dataset")
    run_rq_test(:distance_metric => "rq_euclidean", :num_documents => @dataset.num_docs)
  end

end
