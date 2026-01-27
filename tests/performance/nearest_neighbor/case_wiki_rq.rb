# Copyright Vespa.ai. All rights reserved.

require 'performance/nearest_neighbor/common_rq_base'
require 'performance/nearest_neighbor/common_f32_baseline'
require 'performance/nearest_neighbor/ann_dataset_manager'

class AnnWikiRQPerfTest < CommonRQBase
  include CommonF32Baseline

  def initialize(*args)
    super(*args)
    @dataset = AnnDatasetManager.new("wiki")
    @base_fvecs = @dataset.base_file
    @query_fvecs = @dataset.query_file
    @dimensions = @dataset.dims
    @seed = 42
    @num_queries_for_benchmark = 3452 # natural questions size
  end

  def setup
    super
    set_owner("haakno")
    @perf_recording = "all"
  end

  def test_wiki_rq_no_rotation
    set_description("Test RQ Angular without rotation step (baseline for rotation impact)")
    run_rq_test(:distance_metric => "rq_angular", :num_documents => @dataset.num_docs, :skip_rotation => true)
  end

  def test_wiki_rq_angular
    set_description("Test RQ Angular distance performance and recall on Wikipedia dataset")
    run_rq_test(:distance_metric => "rq_angular", :num_documents => @dataset.num_docs)
  end

  def test_wiki_f32_baseline
    set_description("Test F32 Angular distance performance on Wikipedia dataset")
    run_f32_baseline_test(:distance_metric => "angular", :num_documents => @dataset.num_docs)
  end

end
