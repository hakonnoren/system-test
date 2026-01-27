#!/usr/bin/env ruby
# Copyright Vespa.ai. All rights reserved.
# Convert JSON documents with embeddings to fvecs format for RQ encoding
# Supports both JSON array format and JSONL (one JSON object per line) format

require 'json'

if ARGV.length < 2
  STDERR.puts "Usage: #{$0} <input.json> <output.fvecs> [expected_dims]"
  exit 1
end

input_file = ARGV[0]
output_file = ARGV[1]
expected_dims = ARGV[2]&.to_i

count = 0
skipped = 0

def extract_embedding(doc, expected_dims)
  fields = doc['fields'] || doc['put'] || doc
  return nil unless fields
  
  tensor = fields['embedding']
  return nil unless tensor
  
  values = if tensor.is_a?(Array)
    tensor
  elsif tensor.is_a?(Hash)
    tensor['values'] || tensor['cells']
  end
  
  return nil unless values.is_a?(Array)
  
  # Handle cells format: [{"address": {"x": "0"}, "value": 0.5}, ...]
  if values[0].is_a?(Hash)
    dims = expected_dims || values.map { |c| c['address'].values.first.to_i }.max + 1
    dense = Array.new(dims, 0.0)
    values.each { |c| dense[c['address'].values.first.to_i] = c['value'].to_f }
    values = dense
  end
  
  values
end

def write_fvecs(out, values)
  out.write([values.size].pack('V'))
  out.write(values.pack('f*'))
end

# Detect file format: JSON array or JSONL
first_char = File.read(input_file, 1)
is_json_array = (first_char == '[')

File.open(output_file, 'wb') do |out|
  if is_json_array
    STDERR.puts "Detected JSON array format, parsing..."
    # Parse as JSON array - more memory intensive but handles the format
    docs = JSON.parse(File.read(input_file))
    docs.each do |doc|
      begin
        values = extract_embedding(doc, expected_dims)
        if values
          write_fvecs(out, values)
          count += 1
          STDERR.puts "Processed #{count} documents..." if count % 50000 == 0
        else
          skipped += 1
        end
      rescue => e
        skipped += 1
      end
    end
  else
    STDERR.puts "Detected JSONL format, streaming..."
    # JSONL format - one JSON object per line (memory efficient)
    File.foreach(input_file) do |line|
      begin
        next if line.strip.empty?
        doc = JSON.parse(line)
        values = extract_embedding(doc, expected_dims)
        if values
          write_fvecs(out, values)
          count += 1
          STDERR.puts "Processed #{count} documents..." if count % 50000 == 0
        else
          skipped += 1
        end
      rescue => e
        skipped += 1
      end
    end
  end
end

STDERR.puts "Converted #{count} documents to #{output_file}"
STDERR.puts "Skipped #{skipped} documents" if skipped > 0
