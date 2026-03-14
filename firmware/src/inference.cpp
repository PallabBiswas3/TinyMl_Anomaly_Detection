#include "inference.h"
#include "model_data.h"
#include <math.h>

// ── Hardcoded weights (filled by train.py — see model_weights.h) ─────────────
// For now we use the manual C++ autoencoder with weights baked in.
// After running train.py, copy the weights from output/model_weights.h
// into the arrays below.

// Encoder: 4 → 8 → 2
static const float W_enc1[4][8] = WEIGHTS_ENC1;
static const float B_enc1[8]    = BIAS_ENC1;
static const float W_enc2[8][2] = WEIGHTS_ENC2;
static const float B_enc2[2]    = BIAS_ENC2;

// Decoder: 2 → 8 → 4
static const float W_dec1[2][8] = WEIGHTS_DEC1;
static const float B_dec1[8]    = BIAS_DEC1;
static const float W_dec2[8][4] = WEIGHTS_DEC2;
static const float B_dec2[4]    = BIAS_DEC2;

// ── Activation functions ──────────────────────────────────────────────────────
static inline float relu(float x)    { return x > 0.0f ? x : 0.0f; }
static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// ── Dense layer forward pass ──────────────────────────────────────────────────
static void dense(const float* in, int in_n,
                  const float* W_flat, const float* B,
                  float* out, int out_n,
                  bool use_relu)
{
    for (int j = 0; j < out_n; ++j) {
        float sum = B[j];
        for (int i = 0; i < in_n; ++i)
            sum += in[i] * W_flat[i * out_n + j];
        out[j] = use_relu ? relu(sum) : sigmoid(sum);
    }
}

// ── InferenceEngine ───────────────────────────────────────────────────────────

bool InferenceEngine::begin() {
    Serial.println("[Inference] Init pure-C++ autoencoder...");
    Serial.printf("[Inference] Anomaly threshold: %.6f\n", ANOMALY_THRESHOLD);
    ready_ = true;
    return true;
}

float InferenceEngine::normalize(float value, int idx) const {
    return (value - NORM_MIN[idx]) / (NORM_MAX[idx] - NORM_MIN[idx] + 1e-8f);
}

float InferenceEngine::compute_mse(const float* a, const float* b, int n) const {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum / n;
}

InferenceResult InferenceEngine::run(
    float dist, float jitter, float roc, float variance)
{
    InferenceResult result = {};
    if (!ready_) return result;

    // Normalize
    result.input[FEAT_DISTANCE]       = normalize(dist,     FEAT_DISTANCE);
    result.input[FEAT_JITTER]         = normalize(jitter,   FEAT_JITTER);
    result.input[FEAT_RATE_OF_CHANGE] = normalize(roc,      FEAT_RATE_OF_CHANGE);
    result.input[FEAT_VARIANCE]       = normalize(variance, FEAT_VARIANCE);

    uint32_t t0 = micros();

    // Encoder
    float h1[8], latent[2];
    dense(result.input, 4, (const float*)W_enc1, B_enc1, h1,     8, true);
    dense(h1,           8, (const float*)W_enc2, B_enc2, latent, 2, true);

    // Decoder
    float h2[8];
    dense(latent, 2, (const float*)W_dec1, B_dec1, h2,                8, true);
    dense(h2,     8, (const float*)W_dec2, B_dec2, result.reconstructed, 4, false);

    result.inference_us       = micros() - t0;
    result.reconstruction_mse = compute_mse(result.input, result.reconstructed, 4);
    result.is_anomaly         = result.reconstruction_mse > ANOMALY_THRESHOLD;

    return result;
}

void InferenceEngine::print(const InferenceResult& r) {
    Serial.printf("[Inference] MSE=%.6f  threshold=%.6f  %s  (%uus)\n",
        r.reconstruction_mse, ANOMALY_THRESHOLD,
        r.is_anomaly ? "ANOMALY!" : "normal",
        r.inference_us);
}