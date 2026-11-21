#include "preprocess.h"
#include <math.h>

// Must stay in sync with training/pipeline.py:Preprocessor.
// After any change here: python training/verify_firmware.py

namespace {

// Sort a small array in place (insertion sort: fastest for n <= ~16) and
// return its middle element (upper median for even n, same as Python's s[n // 2]).
float sorted_middle(float* a, int n) {
    for (int i = 1; i < n; ++i) {
        const float v = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; --j; }
        a[j + 1] = v;
    }
    return a[n / 2];
}

} // namespace

void SignalPreprocessor::reset() {
    buf_head_      = 0;
    buf_count_     = 0;
    win_head_      = 0;
    win_count_     = 0;
    last_filtered_ = 0.0f;
    has_last_      = false;
    // invalid_streak_ is deliberately kept: it tracks the sensor, not the signal
}

// Hampel decision for the newest reading (already stored in buf_)
float SignalPreprocessor::hampel(float raw_cm) const {
    const int n = buf_count_;
    float tmp[MODEL_HAMPEL_SIZE];

    for (int i = 0; i < n; ++i) tmp[i] = buf_[i];
    const float med = sorted_middle(tmp, n);

    for (int i = 0; i < n; ++i) tmp[i] = fabsf(buf_[i] - med);
    const float mad = sorted_middle(tmp, n);

    float limit = MODEL_HAMPEL_K_MAD * mad;
    if (limit < MODEL_HAMPEL_MIN_CM) limit = MODEL_HAMPEL_MIN_CM;
    return (fabsf(raw_cm - med) > limit) ? med : raw_cm;
}

// Population variance of the filtered window
float SignalPreprocessor::window_variance() const {
    const int n = win_count_;
    if (n < 2) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += win_[i];
    const float mean = sum / n;
    float sq = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float d = win_[i] - mean;
        sq += d * d;
    }
    return sq / n;
}

bool SignalPreprocessor::push(float raw_cm, uint32_t ts_ms, Features& out) {
    // 1. Validity gate. Written as !(in range) so NaN is rejected too.
    if (!(raw_cm >= MODEL_VALID_MIN_CM && raw_cm <= MODEL_VALID_MAX_CM)) {
        if (invalid_streak_ < UINT16_MAX) invalid_streak_++;
        return false;
    }
    invalid_streak_ = 0;

    // 2. Long gap since the last valid reading -> history is stale, restart.
    //    Unsigned subtraction is millis()-rollover safe.
    if (has_last_ && (uint32_t)(ts_ms - last_ts_) > MODEL_MAX_GAP_MS) {
        reset();
    }

    // 3. Hampel spike filter (ghost echoes, multipath, electrical glitches)
    buf_[buf_head_] = raw_cm;
    buf_head_ = (buf_head_ + 1) % MODEL_HAMPEL_SIZE;
    if (buf_count_ < MODEL_HAMPEL_SIZE) buf_count_++;
    const float filtered = hampel(raw_cm);

    // 4. Window features on the cleaned signal
    win_[win_head_] = filtered;
    win_head_ = (win_head_ + 1) % MODEL_WINDOW_SIZE;
    if (win_count_ < MODEL_WINDOW_SIZE) win_count_++;

    out.filtered_cm = filtered;
    out.variance    = window_variance();
    out.jitter      = sqrtf(out.variance);

    if (has_last_ && ts_ms != last_ts_) {
        const float dt_s   = (uint32_t)(ts_ms - last_ts_) / 1000.0f;
        out.rate_of_change = fabsf(filtered - last_filtered_) / dt_s;
    } else {
        out.rate_of_change = 0.0f;
    }

    last_filtered_ = filtered;
    last_ts_       = ts_ms;
    has_last_      = true;
    return true;
}
