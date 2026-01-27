// Copyright Vespa.ai. All rights reserved.
// Generate RQ-encoded documents for Vespa system-test
// Based on tests/performance/nearest_neighbor/make_docs.cpp

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "rq_encoder.h"

// ============== Interval (from shared.h) ==============

struct Interval {
    float lower;
    float upper;

    Interval() : lower(0.0f), upper(-1.0f) {}

    bool non_empty() const { return (upper - lower) >= 0.0f; }
    bool point() const { return lower == upper; }

    float random() const {
        if (point()) return lower;
        return lower + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX/(upper - lower)));
    }
};

Interval parse_interval(const std::string &str) {
    Interval interval;
    std::stringstream ss(str);
    if (ss.peek() == '[') ss.ignore();
    if (!(ss >> interval.lower)) return Interval();
    if (ss.peek() == ',') ss.ignore();
    if (!(ss >> interval.upper)) return Interval();
    return interval;
}

// ============== Filter parsing ==============

std::vector<int> parse_filters(const std::string &str) {
    std::vector<int> filters;
    std::stringstream ss(str);
    if (ss.peek() == '{' || ss.peek() == '[') ss.ignore();
    int i;
    while (ss >> i) {
        filters.push_back(i);
        if (ss.peek() == ',' || ss.peek() == '}' || ss.peek() == ']') ss.ignore();
    }
    return filters;
}

std::vector<int> gen_filter_values(size_t docid, const std::vector<int> &filters) {
    std::vector<int> result;
    for (auto filter_percent : filters) {
        if ((docid % 100) >= static_cast<size_t>(filter_percent)) {
            result.push_back(filter_percent);
        }
    }
    return result;
}

// ============== FVECS file reading ==============

std::vector<float> read_fvecs_vector(std::ifstream& is, size_t expected_dim) {
    int read_dim;
    is.read(reinterpret_cast<char*>(&read_dim), 4);
    if (!is.good()) return {};

    assert(static_cast<size_t>(read_dim) == expected_dim);
    std::vector<float> vec(expected_dim);
    is.read(reinterpret_cast<char*>(vec.data()), sizeof(float) * expected_dim);
    assert(is.good());
    return vec;
}

// ============== JSON output helpers ==============

template <typename T>
void print_vector(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) os << ",";
        os << static_cast<int>(vec[i]);  // Cast int8 to int for JSON
    }
    os << "]";
}

void print_int_vector(std::ostream& os, const std::vector<int>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) os << ",";
        os << vec[i];
    }
    os << "]";
}

void print_float_vector(std::ostream& os, const std::vector<float>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) os << ",";
        os << vec[i];
    }
    os << "]";
}

// ============== Field name parsing ==============

std::vector<std::string> parse_field_names(const std::string &str) {
    // Skip if this looks like a flag (starts with --)
    if (str.rfind("--", 0) == 0) {
        return {};
    }
    std::vector<std::string> fields;
    std::stringstream ss(str);
    std::string field;
    while (std::getline(ss, field, ',')) {
        if (!field.empty()) {
            fields.push_back(field);
        }
    }
    return fields;
}

// ============== Document printing ==============

void print_put(std::ostream& os, size_t docid,
               const std::vector<int>& filters,
               const Interval& latitude, const Interval& longitude,
               const std::vector<std::string>& rq_fields,
               const std::vector<int8_t>& rq_encoded,
               const std::vector<std::string>& float_fields,
               const std::vector<float>& float_vec) {
    os << "{" << std::endl;
    os << "  \"put\": \"id:test:test::" << docid << "\"," << std::endl;
    os << "  \"fields\": {" << std::endl;
    os << "    \"id\": " << docid;

    if (!filters.empty()) {
        os << "," << std::endl << "    \"filter\": ";
        print_int_vector(os, gen_filter_values(docid, filters));
    }

    if (latitude.non_empty() && longitude.non_empty()) {
        os << "," << std::endl << "    \"latlng\": "
           << "{ \"lat\": " << latitude.random() << ", \"lng\": " << longitude.random() << "}";
    }

    // RQ-encoded fields (all get the same packed codes + metadata)
    for (const auto& rq_field : rq_fields) {
        os << "," << std::endl << "    \"" << rq_field << "\": { \"values\": ";
        print_vector(os, rq_encoded);
        os << " }";
    }

    // Original float fields (for ground truth comparison)
    for (const auto& float_field : float_fields) {
        os << "," << std::endl << "    \"" << float_field << "\": { \"values\": ";
        print_float_vector(os, float_vec);
        os << " }";
    }

    os << std::endl << "  }" << std::endl;
    os << "}";
}

void print_update(std::ostream& os, size_t docid,
                  const std::vector<std::string>& rq_fields,
                  const std::vector<int8_t>& rq_encoded,
                  const std::vector<std::string>& float_fields,
                  const std::vector<float>& float_vec) {
    os << "{" << std::endl;
    os << "  \"update\": \"id:test:test::" << docid << "\"," << std::endl;
    os << "  \"fields\": {" << std::endl;

    bool first = true;
    // RQ-encoded fields
    for (const auto& rq_field : rq_fields) {
        if (!first) os << "," << std::endl;
        first = false;
        os << "    \"" << rq_field << "\": { \"assign\": { \"values\": ";
        print_vector(os, rq_encoded);
        os << " } }";
    }

    // Original float fields
    for (const auto& float_field : float_fields) {
        if (!first) os << "," << std::endl;
        first = false;
        os << "    \"" << float_field << "\": { \"assign\": { \"values\": ";
        print_float_vector(os, float_vec);
        os << " } }";
    }

    os << std::endl << "  }" << std::endl;
    os << "}";
}

// ============== Main ==============

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <vector-file> <num-dims> <put|update> <begin-doc> "
              << "<start-vec> <end-vec> <seed> <rq-fields> [float-fields] "
              << "[filter-values] [latitude-interval] [longitude-interval] [--no-rotation]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Example: " << prog << " sift_base.fvecs 128 put 0 0 1000000 42 vec_rq_euclidean,vec_rq_angular vec_float,vec_float_hnsw" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  vector-file:   Path to .fvecs file" << std::endl;
    std::cerr << "  num-dims:      Vector dimension (e.g., 128 for SIFT)" << std::endl;
    std::cerr << "  put|update:    Feed operation type" << std::endl;
    std::cerr << "  begin-doc:     Starting document ID" << std::endl;
    std::cerr << "  start-vec:     First vector index (inclusive)" << std::endl;
    std::cerr << "  end-vec:       Last vector index (exclusive)" << std::endl;
    std::cerr << "  seed:          Random seed for rotation (42 to match Java)" << std::endl;
    std::cerr << "  rq-fields:     Comma-separated RQ tensor field names (same data to all)" << std::endl;
    std::cerr << "  float-fields:  Comma-separated float tensor field names (optional)" << std::endl;
    std::cerr << "  filter-values: Filter percentages, e.g., [10,50,90] (optional)" << std::endl;
    std::cerr << "  latitude:      Latitude interval, e.g., [-90,90] (optional)" << std::endl;
    std::cerr << "  longitude:     Longitude interval, e.g., [-180,180] (optional)" << std::endl;
    std::cerr << "  --no-rotation: Skip random rotation step (for benchmarking)" << std::endl;
}

int main(int argc, char **argv) {
    if (argc < 9) {
        print_usage(argv[0]);
        return 1;
    }

    // Parse arguments
    std::string vector_file = argv[1];
    size_t dim_size = std::stoll(argv[2]);
    std::string feed_op = argv[3];
    size_t begin_doc = std::stoll(argv[4]);
    size_t start_vector = std::stoll(argv[5]);
    size_t end_vector = std::stoll(argv[6]);
    uint64_t seed = std::stoull(argv[7]);

    // Parse comma-separated field names
    std::vector<std::string> rq_fields = parse_field_names(argv[8]);

    std::vector<std::string> float_fields;
    if (argc > 9) float_fields = parse_field_names(argv[9]);

    std::vector<int> filters;
    if (argc > 10) filters = parse_filters(argv[10]);

    Interval latitude, longitude;
    if (argc > 11) latitude = parse_interval(argv[11]);
    if (argc > 12) longitude = parse_interval(argv[12]);

    // Check for --no-rotation flag anywhere in arguments
    bool skip_rotation = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-rotation") {
            skip_rotation = true;
            break;
        }
    }

    // Initialize random seed for filter/location generation
    srand(seed);

    // Create RQ encoder
    rq::RQEncoder encoder(dim_size, seed, skip_rotation);
    size_t packed_size = encoder.encoded_size();  // dim + 16 bytes metadata

    std::cerr << "RQ Encoder: dim=" << dim_size << ", packed_size=" << packed_size
              << ", seed=" << seed << ", rq_fields=" << rq_fields.size()
              << ", float_fields=" << float_fields.size()
              << ", skip_rotation=" << (skip_rotation ? "true" : "false") << std::endl;

    // Open input file
    std::ifstream is(vector_file, std::ifstream::binary);
    if (!is.good()) {
        std::cerr << "Could not open '" << vector_file << "'" << std::endl;
        return 1;
    }

    // Skip vectors before start_vector
    is.ignore(start_vector * (4 + sizeof(float) * dim_size));

    // Output JSON array
    std::cout << "[" << std::endl;
    bool first = true;
    bool make_puts = (feed_op == "put");

    for (size_t vec_num = start_vector; vec_num < end_vector; ++vec_num) {
        std::vector<float> float_vec = read_fvecs_vector(is, dim_size);
        if (float_vec.empty()) break;

        // Encode to RQ format
        std::vector<int8_t> rq_encoded = encoder.encode_as_int8(float_vec);

        if (!first) std::cout << "," << std::endl;
        first = false;

        size_t docid = begin_doc + vec_num - start_vector;
        if (make_puts) {
            print_put(std::cout, docid, filters, latitude, longitude,
                      rq_fields, rq_encoded, float_fields, float_vec);
        } else {
            print_update(std::cout, docid, rq_fields, rq_encoded, float_fields, float_vec);
        }
    }

    std::cout << std::endl << "]" << std::endl;
    is.close();

    return 0;
}
