# Copyright Vespa.ai. All rights reserved.

class AnnDatasetManager
  attr_reader :name, :dims, :metric, :base_file, :query_file, :num_docs

  DATASETS = {
    "sift" => {
      dims: 128,
      metric: "rq_euclidean",
      base_file: "sift-data/sift_base.fvecs",
      query_file: "sift-data/sift_query.fvecs",
      num_docs: 1_000_000
    },
    "gist" => {
      dims: 960,
      metric: "rq_euclidean",
      base_file: "gist-data/gist_base_300k.fvecs",
      query_file: "gist-data/gist_query.fvecs",
      num_docs: 300_000
    },
    "wiki" => {
      dims: 384,
      metric: "rq_angular",
      base_file: "wiki-data/paragraph_docs.all.json",
      query_file: "wiki-data/queries.txt",
      num_docs: 485_851  # Wikipedia paragraph dataset size
    }
  }

  def initialize(name)
    @name = name
    config = DATASETS[name]
    raise "Unknown dataset: #{name}" unless config
    @dims = config[:dims]
    @metric = config[:metric]
    @base_file = config[:base_file]
    @query_file = config[:query_file]
    @num_docs = config[:num_docs]
  end

  def prepare_base_fvecs(test_case, node)
    downloaded_file = download_base(test_case, node)
    if downloaded_file.end_with?(".json")
      fvecs_file = downloaded_file.sub(".json", ".fvecs")
      
      # Check if fvecs file already exists (cached conversion)
      (exitcode, _) = node.execute("test -f #{fvecs_file}", :exceptiononfailure => false, :exitcode => true)
      if exitcode == "0"
        puts "Using already converted file #{fvecs_file}"
        return fvecs_file
      end
      
      puts "Converting #{downloaded_file} to #{fvecs_file}"
      # Use the json_to_fvecs.rb script from the wiki folder
      script_path = "#{test_case.selfdir}wiki/json_to_fvecs.rb"
      node.execute("ruby #{script_path} #{downloaded_file} #{fvecs_file} #{@dims}")
      return fvecs_file
    end
    downloaded_file
  end

  def prepare_query_fvecs(test_case, node)
    downloaded_file = download_queries(test_case, node)
    if downloaded_file.end_with?(".txt") && @name == "wiki"
      fvecs_file = downloaded_file.sub(".txt", ".fvecs")
      
      # Check if fvecs file already exists (cached conversion)
      (exitcode, _) = node.execute("test -f #{fvecs_file}", :exceptiononfailure => false, :exitcode => true)
      if exitcode == "0"
        puts "Using already converted file #{fvecs_file}"
        return fvecs_file
      end
      
      puts "Converting #{downloaded_file} to #{fvecs_file}"
      # Use the queries_to_fvecs.rb script from the wiki folder
      script_path = "#{test_case.selfdir}wiki/queries_to_fvecs.rb"
      node.execute("ruby #{script_path} #{downloaded_file} #{fvecs_file}")
      return fvecs_file
    end
    downloaded_file
  end

  private

  def download_base(test_case, node)
    test_case.nn_download_file(@base_file, node)
  end

  def download_queries(test_case, node)
    test_case.nn_download_file(@query_file, node)
  end
end
