// Copyright Vespa.ai. All rights reserved.
// Generate RQ-encoded queries for Vespa system-test
// Based on tests/performance/nearest_neighbor/make_queries.cpp

#include <cassert>
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

// ============== URL encoding helpers ==============

const std::string l_brace = "%7B";
const std::string r_brace = "%7D";
const std::string l_par = "(";
const std::string r_par = ")";
const std::string quot = "%22";
const std::string eq = "%3D";

// ============== Output helpers ==============

void print_int8_vector(std::ostream& os, const std::vector<int8_t>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) os << ",";
        os << static_cast<int>(vec[i]);
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

std::ostream& print_int_param(std::ostream& os, const std::string& key, int value) {
    os << quot << key << quot << ":" << value;
    return os;
}

std::ostream& print_bool_param(std::ostream& os, const std::string& key, bool value) {
    os << quot << key << quot << ":" << (value ? "true" : "false");
    return os;
}

std::ostream& print_str_param(std::ostream& os, const std::string& key, const std::string& value) {
    os << quot << key << quot << ":" << quot << value << quot;
    return os;
}

// ============== Query generation ==============

void print_nns(std::ostream& os, bool approximate, int target_hits, int explore_hits,
               const std::string& doc_tensor, const std::string& query_tensor) {
    os << "[" << l_brace;
    print_int_param(os, "targetNumHits", target_hits) << ",";
    print_int_param(os, "hnsw.exploreAdditionalHits", explore_hits) << ",";
    print_bool_param(os, "approximate", approximate) << ",";
    print_str_param(os, "label", "nns");
    os << r_brace << "]" << "nearestNeighbor(" << doc_tensor << "," << query_tensor << ")";
}

void print_rq_query(std::ostream& os, bool approximate, int target_hits, int explore_hits,
                    int filter_percent, float radius,
                    const Interval& latitude, const Interval& longitude,
                    const std::string& doc_tensor, const std::string& query_tensor,
                    const std::vector<int8_t>& rq_encoded) {
    os << "/search/?yql=select%20*%20from%20sources%20*%20where%20";
    print_nns(os, approximate, target_hits, explore_hits, doc_tensor, query_tensor);
    if (filter_percent > 0) {
        os << "%20and%20filter" << eq << filter_percent;
    }
    if (radius > 0.0f && latitude.non_empty() && longitude.non_empty()) {
        os << "%20and%20geoLocation" << l_par << "latlng," << latitude.random()
           << "," << longitude.random() << "," << quot << radius << "+km" << quot << r_par;
    }
    os << ";&ranking.features.query(" << query_tensor << ")=";
    print_int8_vector(os, rq_encoded);
    os << std::endl;
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

// ============== Main ==============

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <vector-file> <num-dims> <num-queries> <seed> "
              << "<doc-tensor> <query-tensor> [approximate] [target-hits] [explore-hits] "
              << "[filter-percent] [radius] [latitude] [longitude]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Example: " << prog << " sift_query.fvecs 128 10000 42 vec_rq q_rq true 100 0" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  vector-file:   Path to .fvecs query file" << std::endl;
    std::cerr << "  num-dims:      Vector dimension (e.g., 128 for SIFT)" << std::endl;
    std::cerr << "  num-queries:   Number of queries to generate" << std::endl;
    std::cerr << "  seed:          Random seed for rotation (42 to match documents)" << std::endl;
    std::cerr << "  doc-tensor:    Name of document RQ tensor field" << std::endl;
    std::cerr << "  query-tensor:  Name of query RQ tensor (e.g., q_rq)" << std::endl;
    std::cerr << "  approximate:   Use HNSW (true) or brute force (false)" << std::endl;
    std::cerr << "  target-hits:   Target number of hits" << std::endl;
    std::cerr << "  explore-hits:  Additional HNSW exploration" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Output: URL-encoded queries for fbench, one per line" << std::endl;
}

void print_only_vectors(std::ifstream& is, rq::RQEncoder& encoder, size_t dim_size, size_t num_queries) {
    for (size_t i = 0; i < num_queries; ++i) {
        std::vector<float> vec = read_fvecs_vector(is, dim_size);
        if (vec.empty()) break;

        std::vector<int8_t> rq_encoded = encoder.encode_as_int8(vec);
        print_int8_vector(std::cout, rq_encoded);
        std::cout << std::endl;
    }
}

void print_only_float_vectors(std::ifstream& is, size_t dim_size, size_t num_queries) {
    for (size_t i = 0; i < num_queries; ++i) {
        std::vector<float> vec = read_fvecs_vector(is, dim_size);
        if (vec.empty()) break;

        print_float_vector(std::cout, vec);
        std::cout << std::endl;
    }
}

int main(int argc, char **argv) {
    if (argc < 7) {
        print_usage(argv[0]);
        return 1;
    }

    // Parse required arguments
    std::string vector_file = argv[1];
    size_t dim_size = std::stoll(argv[2]);
    size_t num_queries = std::stoll(argv[3]);
    uint64_t seed = std::stoull(argv[4]);
    std::string doc_tensor = argv[5];
    std::string query_tensor = argv[6];

    // Parse optional arguments
    bool approximate = true;
    if (argc > 7) approximate = (std::string(argv[7]) == "true");

    int target_hits = 100;
    if (argc > 8) target_hits = std::stoi(argv[8]);

    int explore_hits = 0;
    if (argc > 9) explore_hits = std::stoi(argv[9]);

    int filter_percent = 0;
    if (argc > 10) filter_percent = std::stoi(argv[10]);

    float radius = 0;
    if (argc > 11) radius = std::stof(argv[11]);

    Interval latitude, longitude;
    if (argc > 12) latitude = parse_interval(argv[12]);
    if (argc > 13) longitude = parse_interval(argv[13]);

    // Initialize
    srand(seed);
    rq::RQEncoder encoder(dim_size, seed);

    std::cerr << "RQ Query Generator: dim=" << dim_size << ", packed_size=" << encoder.encoded_size()
              << ", seed=" << seed << ", queries=" << num_queries << std::endl;

    // Open input file
    std::ifstream is(vector_file, std::ifstream::binary);
    if (!is.good()) {
        std::cerr << "Could not open '" << vector_file << "'" << std::endl;
        return 1;
    }

    // If doc_tensor is "--only-vectors", just output RQ-encoded vectors
    if (doc_tensor == "--only-vectors") {
        print_only_vectors(is, encoder, dim_size, num_queries);
        is.close();
        return 0;
    }

    // If doc_tensor is "--only-float-vectors", just output raw float vectors
    if (doc_tensor == "--only-float-vectors") {
        print_only_float_vectors(is, dim_size, num_queries);
        is.close();
        return 0;
    }

    // Generate queries
    for (size_t i = 0; i < num_queries; ++i) {
        std::vector<float> vec = read_fvecs_vector(is, dim_size);
        if (vec.empty()) break;

        std::vector<int8_t> rq_encoded = encoder.encode_as_int8(vec);
        print_rq_query(std::cout, approximate, target_hits, explore_hits,
                       filter_percent, radius, latitude, longitude,
                       doc_tensor, query_tensor, rq_encoded);
    }

    is.close();
    return 0;
}
