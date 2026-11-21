#pragma once
// On-device signal cleaning + feature extraction.
//
// Pure C++ (no Arduino dependency) so it can be compiled and checked on a PC
// (firmware/test_host/host_check.cpp). Mirrored line-for-line by
// training/pipeline.py:Preprocessor; the parameters come from model_data.h,
// which train.py generates, so device and training can't drift apart.
//
//   raw reading ─▶ validity gate ─▶ gap reset ─▶ Hampel spike filter ─▶ window features
//
// Hampel filter: a reading is replaced by the median of the last N readings only
// if it deviates from that median by more than K·MAD (and HAMPEL_MIN_CM). Isolated
// ghost echoes are removed, but a real oscillation passes through - a plain
// median filter would cancel any vibration faster than its window.
//
// Cost per sample: two 7-element insertion sorts + ~40 float ops (a few µs).
// RAM: ~70 bytes. No heap.

#include <stdint.h>
#include "model_data.h"

static_assert(MODEL_HAMPEL_SIZE % 2 == 1 && MODEL_HAMPEL_SIZE <= 15,
              "MODEL_HAMPEL_SIZE must be odd and small");
static_assert(MODEL_WINDOW_SIZE >= 2 && MODEL_WINDOW_SIZE <= 255,
              "MODEL_WINDOW_SIZE out of range");

// Model input (one row per valid reading)
struct Features {
    float filtered_cm;      // spike-filtered distance
    float jitter;           // std dev of the filtered window
    float rate_of_change;   // |delta filtered| per second
    float variance;         // variance of the filtered window
};

class SignalPreprocessor {
public:
    // Feed one raw reading (distance_cm < 0 = no echo).
    // Returns true and fills `out` for a valid reading. Returns false for an
    // invalid one (timeout, blind zone, over-range, NaN), which is discarded.
    bool push(float raw_cm, uint32_t ts_ms, Features& out);

    // Forget all history (called automatically after a long gap).
    void reset();

    // Consecutive invalid readings; >= MODEL_FAULT_INVALID_COUNT means sensor fault.
    uint16_t invalid_streak() const { return invalid_streak_; }
    bool     sensor_fault()   const { return invalid_streak_ >= MODEL_FAULT_INVALID_COUNT; }

private:
    float    buf_[MODEL_HAMPEL_SIZE] = {};      // circular: last valid raw readings
    uint8_t  buf_head_  = 0;
    uint8_t  buf_count_ = 0;

    float    win_[MODEL_WINDOW_SIZE] = {};      // circular: last filtered readings
    uint8_t  win_head_  = 0;
    uint8_t  win_count_ = 0;

    float    last_filtered_  = 0.0f;
    uint32_t last_ts_        = 0;
    bool     has_last_       = false;
    uint16_t invalid_streak_ = 0;

    float hampel(float raw_cm) const;
    float window_variance() const;
};
