/*
    This file is part of kalien-beam project.
    Licensed under the MIT License.
    Author: Fred Kyung-jin Rezeau <hello@kyungj.in>
*/

// Must match original TypeScript implementation for ZK verification.
// See https://github.com/kalepail/kalien/blob/main/src/game/tape.ts

#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

static constexpr uint32_t TAPE_MAGIC = 0x5A4B5450;
static constexpr uint8_t TAPE_VERSION = 4;
static constexpr uint8_t TAPE_RULES_TAG = 4;
static constexpr int TAPE_HEADER_SIZE = 16;
static constexpr int TAPE_FOOTER_SIZE = 8;

// CRC-32 (ISO 3309 / ITU-T V.42 polynomial 0xEDB88320)
static uint32_t crc32(const uint8_t* data, size_t len) {
    static const auto table = []() {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int j = 0; j < 8; j++) {
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            t[i] = c;
        }
        return t;
    }();
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return (crc ^ 0xFFFFFFFFu);
}

class Tape {
  public:
    void add(uint8_t input) { frames.push_back(input & 0x0F); }

    bool write(const std::string& path, uint32_t seed, uint32_t score = 0) const {
        auto data = serialize(seed, score);
        std::ofstream f(path, std::ios::binary);
        if (!f) {
            return false;
        }
        f.write(reinterpret_cast<const char*>(data.data()), data.size());
        return f.good();
    }

    uint32_t len() const { return static_cast<uint32_t>(frames.size()); }

  private:
    std::vector<uint8_t> frames;

    std::vector<uint8_t> serialize(uint32_t seed, uint32_t score) const {
        const uint32_t count = len();
        const uint32_t bytes = (count + 1) >> 1;
        const size_t total = TAPE_HEADER_SIZE + bytes + TAPE_FOOTER_SIZE;
        std::vector<uint8_t> data(total, 0);
        auto writeU32 = [&](size_t off, uint32_t v) {
            data[off + 0] = v & 0xFF;
            data[off + 1] = (v >> 8) & 0xFF;
            data[off + 2] = (v >> 16) & 0xFF;
            data[off + 3] = (v >> 24) & 0xFF;
        };
        writeU32(0, TAPE_MAGIC);
        data[4] = TAPE_VERSION;
        data[5] = TAPE_RULES_TAG;
        writeU32(8, seed);
        writeU32(12, count);

        for (uint32_t i = 0; i < bytes; i++) {
            uint8_t lo = frames[2 * i] & 0x0F;
            uint8_t hi = (2 * i + 1 < count) ? (frames[2 * i + 1] & 0x0F) << 4 : 0;
            data[TAPE_HEADER_SIZE + i] = lo | hi;
        }

        const size_t offset = TAPE_HEADER_SIZE + bytes;
        writeU32(offset, score);
        writeU32(offset + 4, crc32(data.data(), offset));
        return data;
    }
};
