#pragma once
#include <Arduino.h>
#include "model_data.h"
#include "model_weights.h"   // ← add this line

struct InferenceResult {
    float    input[4];
    float    reconstructed[4];
    float    reconstruction_mse;
    bool     is_anomaly;
    uint32_t inference_us;
};

class InferenceEngine {
public:
    bool begin();
    InferenceResult run(float dist, float jitter, float roc, float variance);
    static void print(const InferenceResult& r);
    bool ready() const { return ready_; }

private:
    bool  ready_ = false;
    float normalize(float value, int idx) const;
    float compute_mse(const float* a, const float* b, int n) const;
};
