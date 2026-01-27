// Copyright Vespa.ai. All rights reserved.
// RQ Encoder header for system-test document and query generation
// Based on vespa_rq_tests/rq_standalone_test.cpp

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <span>
#include <vector>

namespace rq {

// ============== RQMetadata ==============
// Must match the C++ distance function expectation (16 bytes, little-endian)

struct RQMetadata {
    float l_x;          // midpoint value: min + 128*delta
    float delta_x;      // quantization step size
    float norm_sq;      // squared norm of original vector
    int32_t code_sum;   // sum of quantized codes (can be negative)
};

static_assert(sizeof(RQMetadata) == 16, "RQMetadata must be 16 bytes");

// ============== FastRotation ==============
// FWHT-based random rotation with blocked transforms

class FastRotation {
public:
    static constexpr uint32_t BLOCK_SIZE = 32;
    static constexpr int NUM_ROUNDS = 3;

    FastRotation(uint32_t dimension, uint64_t seed)
        : _dimension(dimension),
          _padded_dim(round_up(dimension)),
          _seed(seed)
    {
        init_parameters();
    }

    void rotate(std::span<const float> input, std::span<float> output) const {
        // Copy input to output with zero padding
        std::copy(input.begin(), input.end(), output.begin());
        std::fill(output.begin() + input.size(), output.begin() + _padded_dim, 0.0f);

        // Apply each rotation round
        for (int round = 0; round < NUM_ROUNDS; ++round) {
            apply_round(output.data(), round);
        }
    }

    uint32_t dimension() const { return _dimension; }
    uint32_t padded_dimension() const { return _padded_dim; }
    uint64_t seed() const { return _seed; }

private:
    uint32_t _dimension;
    uint32_t _padded_dim;
    uint64_t _seed;
    std::vector<int8_t> _signs;
    std::vector<uint32_t> _permutation;

    static uint32_t round_up(uint32_t n) {
        return ((n + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    }

    void init_parameters() {
        std::mt19937_64 rng(_seed);
        _signs.resize(_padded_dim * NUM_ROUNDS);
        _permutation.resize(_padded_dim * NUM_ROUNDS);

        std::uniform_int_distribution<int> sign_dist(0, 1);

        for (int round = 0; round < NUM_ROUNDS; ++round) {
            size_t offset = round * _padded_dim;
            for (uint32_t i = 0; i < _padded_dim; ++i) {
                _signs[offset + i] = sign_dist(rng) * 2 - 1;
            }
            for (uint32_t i = 0; i < _padded_dim; ++i) {
                _permutation[offset + i] = i;
            }
            std::shuffle(_permutation.begin() + offset,
                         _permutation.begin() + offset + _padded_dim, rng);
        }
    }

    void apply_fwht_block(float* data, uint32_t size) const {
        for (uint32_t h = 1; h < size; h *= 2) {
            for (uint32_t i = 0; i < size; i += h * 2) {
                for (uint32_t j = i; j < i + h; ++j) {
                    float x = data[j];
                    float y = data[j + h];
                    data[j] = x + y;
                    data[j + h] = x - y;
                }
            }
        }
        float scale = 1.0f / std::sqrt(static_cast<float>(size));
        for (uint32_t i = 0; i < size; ++i) {
            data[i] *= scale;
        }
    }

    void apply_round(float* data, int round) const {
        size_t offset = round * _padded_dim;

        // Apply random signs
        for (uint32_t i = 0; i < _padded_dim; ++i) {
            data[i] *= _signs[offset + i];
        }

        // Apply blocked FWHT
        for (uint32_t block_start = 0; block_start < _padded_dim; block_start += BLOCK_SIZE) {
            apply_fwht_block(data + block_start, BLOCK_SIZE);
        }

        // Apply permutation
        std::vector<float> temp(data, data + _padded_dim);
        for (uint32_t i = 0; i < _padded_dim; ++i) {
            data[_permutation[offset + i]] = temp[i];
        }
    }
};

// ============== RQEncoder ==============
// Full RQ encoding pipeline: rotation + scalar quantization

class RQEncoder {
public:
    RQEncoder(uint32_t dimension, uint64_t seed, bool skip_rotation = false)
        : _dimension(dimension),
          _rotation(dimension, seed),
          _scratch(_rotation.padded_dimension()),
          _skip_rotation(skip_rotation)
    {}

    // Encode a vector and write packed output [codes][metadata]
    void encode(const std::vector<float>& input, void* output) const {
        uint8_t* codes = static_cast<uint8_t*>(output);
        RQMetadata* meta = reinterpret_cast<RQMetadata*>(codes + _dimension);
        rotate_and_quantize(input, codes, *meta);
    }

    // Encode and return as vector of int8 values (for JSON output)
    std::vector<int8_t> encode_as_int8(const std::vector<float>& input) const {
        std::vector<uint8_t> packed(encoded_size());
        encode(input, packed.data());

        // Reinterpret as int8 for JSON output
        std::vector<int8_t> result(encoded_size());
        std::memcpy(result.data(), packed.data(), encoded_size());
        return result;
    }

    uint32_t dimension() const { return _dimension; }
    uint32_t encoded_size() const { return _dimension + sizeof(RQMetadata); }
    bool skip_rotation() const { return _skip_rotation; }

private:
    uint32_t _dimension;
    FastRotation _rotation;
    mutable std::vector<float> _scratch;
    bool _skip_rotation;

    void rotate_and_quantize(const std::vector<float>& input, uint8_t* codes, RQMetadata& meta) const {
        // Compute original norm squared
        double norm_sq = 0.0;
        for (float x : input) {
            norm_sq += static_cast<double>(x) * x;
        }
        meta.norm_sq = static_cast<float>(norm_sq);

        // Rotate (or skip rotation for benchmarking comparison)
        if (_skip_rotation) {
            // Copy input directly to scratch (no rotation)
            std::copy(input.begin(), input.end(), _scratch.begin());
        } else {
            _rotation.rotate(std::span<const float>(input), std::span<float>(_scratch));
        }

        // Find min/max of rotated vector (only first _dimension elements)
        float min_val = *std::min_element(_scratch.begin(), _scratch.begin() + _dimension);
        float max_val = *std::max_element(_scratch.begin(), _scratch.begin() + _dimension);

        float range = max_val - min_val;
        constexpr float EPS = 1e-6f;
        float delta = std::max(range / 255.0f, EPS);

        // For int8: l_x is the reference point such that code=-128 maps to min_val
        // and code=+127 maps to max_val (approximately the midpoint)
        meta.l_x = min_val + 128.0f * delta;
        meta.delta_x = delta;

        // Quantize to [-128, 127]
        int32_t code_sum = 0;
        for (uint32_t i = 0; i < _dimension; ++i) {
            float normalized = (_scratch[i] - meta.l_x) / delta;
            int code = static_cast<int>(std::round(normalized));
            code = std::clamp(code, -128, 127);
            codes[i] = static_cast<uint8_t>(static_cast<int8_t>(code));  // Store as reinterpreted uint8
            code_sum += static_cast<int8_t>(codes[i]);  // Sum as signed
        }
        meta.code_sum = code_sum;
    }
};

} // namespace rq
