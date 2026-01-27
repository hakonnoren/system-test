#!/usr/bin/env ruby
# Copyright Vespa.ai. All rights reserved.
# Convert fbench query URLs with embeddings to fvecs format

require 'cgi'
require 'json'

if ARGV.length < 2
  STDERR.puts "Usage: #{$0} <queries.txt> <queries.fvecs>"
  exit 1
end

input_file = ARGV[0]
output_file = ARGV[1]

count = 0
skipped = 0

File.open(output_file, 'wb') do |out|
  File.foreach(input_file) do |line|
    begin
      line = line.strip
      next if line.empty?
      
      # Parse URL query parameters
      parts = line.split('?', 2)
      next if parts.length < 2
      
      params = CGI.parse(parts[1])
      
      # Extract vector from input.query(question) parameter
      vec_str = params['input.query(question)']&.first
      next unless vec_str
      
      # Parse JSON array: "[0.1, 0.2, ...]"
      values = JSON.parse(vec_str)
      
      # Write fvecs format: [dims:uint32][values:float32*dims]
      out.write([values.size].pack('V'))
      out.write(values.pack('f*'))
      count += 1
    rescue => e
      skipped += 1
    end
  end
end

STDERR.puts "Converted #{count} queries to #{output_file}"
STDERR.puts "Skipped #{skipped} queries" if skipped > 0
