"""
ESP32 Ultrasonic Anomaly Detector — Training Pipeline
======================================================
Uses HC-SR04 distance readings as input features.

Features extracted from a rolling window of distance readings:
  [0] distance_cm     — raw distance (cm)
  [1] jitter          — std deviation over last N readings (instability)
  [2] rate_of_change  — abs(current - previous) per second (sudden moves)
  [3] variance        — variance over last N readings

Anomaly scenarios this detects:
  - Object too close  (< safe zone)
  - Object too far / missing (> expected range)
  - Erratic movement  (high jitter/variance)
  - Sudden position change (high rate_of_change)
"""

import numpy as np
import os

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not installed. Run: pip install tensorflow")
    print("   Continuing in SIMULATION MODE\n")

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_DIM   = 4       # [distance, jitter, rate_of_change, variance]
LATENT_DIM  = 2       # bottleneck
EPOCHS      = 100
BATCH_SIZE  = 32
N_NORMAL    = 3000
N_ANOMALY   = 400
OUTPUT_DIR  = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Data Generation ────────────────────────────────────────────────────────

def simulate_window_features(
    mean_dist, std_dist, jitter_scale, roc_scale, n, seed
) -> np.ndarray:
    """
    Simulate extracted window features for a given operating condition.
    Each sample = [distance_cm, jitter, rate_of_change, variance]
    """
    rng = np.random.default_rng(seed)
    distance       = rng.normal(mean_dist, std_dist,     (n, 1)).clip(2, 400)
    jitter         = np.abs(rng.normal(0,  jitter_scale, (n, 1)))
    rate_of_change = np.abs(rng.normal(0,  roc_scale,    (n, 1)))
    variance       = jitter ** 2 + rng.normal(0, 0.01,   (n, 1))
    return np.hstack([distance, jitter, rate_of_change, variance]).astype(np.float32)

print("── Step 1: Generating ultrasonic data ──────────────────────────")

# Normal: object sitting ~30cm away, stable
normal_data = simulate_window_features(
    mean_dist=30.0, std_dist=1.5,
    jitter_scale=0.3, roc_scale=0.2,
    n=N_NORMAL, seed=42
)

# Anomaly 1: object too close (<10cm) — obstruction
anomaly_close = simulate_window_features(
    mean_dist=6.0, std_dist=1.0,
    jitter_scale=0.5, roc_scale=0.8,
    n=N_ANOMALY // 2, seed=10
)

# Anomaly 2: object missing / too far (>80cm) — object fell off
anomaly_far = simulate_window_features(
    mean_dist=150.0, std_dist=10.0,
    jitter_scale=5.0, roc_scale=8.0,
    n=N_ANOMALY // 2, seed=11
)

anomaly_data = np.vstack([anomaly_close, anomaly_far])

# ── Normalize ─────────────────────────────────────────────────────────────────

data_min = normal_data.min(axis=0)
data_max = normal_data.max(axis=0)

def normalize(x):
    return (x - data_min) / (data_max - data_min + 1e-8)

normal_norm  = normalize(normal_data)
anomaly_norm = normalize(anomaly_data)

split   = int(0.8 * N_NORMAL)
X_train = normal_norm[:split]
X_val   = normal_norm[split:]

print(f"  Normal samples   : {N_NORMAL}  (object at ~30cm, stable)")
print(f"  Anomaly samples  : {N_ANOMALY} (too close / too far / erratic)")
print(f"  Features         : [distance_cm, jitter, rate_of_change, variance]")
print(f"  Norm min: {data_min}")
print(f"  Norm max: {data_max}\n")

np.save(os.path.join(OUTPUT_DIR, "norm_min.npy"), data_min)
np.save(os.path.join(OUTPUT_DIR, "norm_max.npy"), data_max)

# ── 2. Build & Train ──────────────────────────────────────────────────────────

def build_autoencoder(input_dim, latent_dim):
    inp    = keras.Input(shape=(input_dim,), name="sensor_input")
    x      = keras.layers.Dense(8,          activation="relu",    name="enc1")(inp)
    latent = keras.layers.Dense(latent_dim, activation="relu",    name="latent")(x)
    x      = keras.layers.Dense(8,          activation="relu",    name="dec1")(latent)
    out    = keras.layers.Dense(input_dim,  activation="sigmoid", name="out")(x)
    return keras.Model(inp, out, name="ultrasonic_autoencoder")

if TF_AVAILABLE:
    print("── Step 2: Training autoencoder ────────────────────────────────")
    model = build_autoencoder(INPUT_DIM, LATENT_DIM)
    model.summary()
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, X_train,
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_val, X_val), verbose=1)

    # ── 3. Threshold ──────────────────────────────────────────────────────────
    print("\n── Step 3: Computing threshold ─────────────────────────────────")

    def compute_mse(model, data):
        pred = model.predict(data, verbose=0)
        return np.mean(np.square(data - pred), axis=1)

    val_err   = compute_mse(model, X_val)
    anom_err  = compute_mse(model, anomaly_norm)
    threshold = float(np.mean(val_err) + 3 * np.std(val_err))

    recall = np.sum(anom_err > threshold) / len(anom_err)
    fp     = np.sum(val_err  > threshold)

    print(f"  Val MSE    : {np.mean(val_err):.6f} ± {np.std(val_err):.6f}")
    print(f"  Anomaly MSE: {np.mean(anom_err):.6f} ± {np.std(anom_err):.6f}")
    print(f"  Threshold  : {threshold:.6f}")
    print(f"  Recall     : {recall*100:.1f}%")
    print(f"  False positives: {fp}/{len(val_err)}")

    # ── 4. INT8 Quantize ──────────────────────────────────────────────────────
    print("\n── Step 4: INT8 quantization ───────────────────────────────────")

    def rep_dataset():
        for i in range(0, len(X_train), 10):
            yield [X_train[i:i+1].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type  = tf.int8
    conv.inference_output_type = tf.int8
    tflite_model = conv.convert()

    fp32_size = sum(np.prod(w.shape) * 4 for w in model.weights)
    print(f"  FP32 size : {fp32_size} bytes")
    print(f"  INT8 size : {len(tflite_model)} bytes  ({fp32_size/len(tflite_model):.1f}x smaller)")

    with open(os.path.join(OUTPUT_DIR, "anomaly_model.tflite"), "wb") as f:
        f.write(tflite_model)

else:
    threshold    = 0.018
    tflite_model = bytes([0x18, 0x00, 0x00, 0x00] + [0x00] * 60)
    print("  SIMULATION: using dummy model + default threshold")

# ── 5. Export C header ────────────────────────────────────────────────────────

print("\n── Step 5: Exporting model_data.h ──────────────────────────────")

def export_header(model_bytes, threshold, data_min, data_max, path):
    lines = [
        "// Auto-generated by train.py — DO NOT EDIT",
        "// Ultrasonic anomaly detector — INT8 quantized autoencoder",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"// Model flatbuffer ({len(model_bytes)} bytes)",
        f"const unsigned int g_model_len = {len(model_bytes)};",
        "alignas(8) const uint8_t g_model_data[] = {",
    ]
    data = list(model_bytes)
    for i in range(0, len(data), 12):
        row = data[i:i+12]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in row) + ",")
    lines += [
        "};",
        "",
        f"const float ANOMALY_THRESHOLD = {threshold:.6f}f;",
        "",
        "// Normalization constants from training data",
        f"const float NORM_MIN[4] = {{{', '.join(f'{v:.4f}f' for v in data_min)}}};",
        f"const float NORM_MAX[4] = {{{', '.join(f'{v:.4f}f' for v in data_max)}}};",
        "",
        "// Feature indices",
        "#define FEAT_DISTANCE       0   // cm",
        "#define FEAT_JITTER         1   // std dev over window",
        "#define FEAT_RATE_OF_CHANGE 2   // |delta| per second",
        "#define FEAT_VARIANCE       3   // variance over window",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved → {path}")

export_header(tflite_model, threshold, data_min, data_max,
              os.path.join(OUTPUT_DIR, "model_data.h"))
# ── Step 6: Export C++ weights header ────────────────────────────────────────
print("\n── Step 6: Exporting model_weights.h ───────────────────────────")

if TF_AVAILABLE:
    def arr2c_define(arr, name):
        flat = arr.flatten().tolist()
        vals = ", ".join(f"{v:.6f}f" for v in flat)
        return f"#define {name} {{{vals}}}"

    w = {weight.name: weight.numpy() for weight in model.weights}

    weight_lines = [
        "// Auto-generated by train.py — DO NOT EDIT",
        "#pragma once",
        "",
        arr2c_define(w['enc1/kernel:0'],   'WEIGHTS_ENC1'),
        arr2c_define(w['enc1/bias:0'],     'BIAS_ENC1'),
        arr2c_define(w['latent/kernel:0'], 'WEIGHTS_ENC2'),
        arr2c_define(w['latent/bias:0'],   'BIAS_ENC2'),
        arr2c_define(w['dec1/kernel:0'],   'WEIGHTS_DEC1'),
        arr2c_define(w['dec1/bias:0'],     'BIAS_DEC1'),
        arr2c_define(w['out/kernel:0'],    'WEIGHTS_DEC2'),
        arr2c_define(w['out/bias:0'],      'BIAS_DEC2'),
    ]

    weights_path = os.path.join(OUTPUT_DIR, "model_weights.h")
    with open(weights_path, "w") as f:
        f.write("\n".join(weight_lines) + "\n")
    print(f"  Saved → {weights_path}")
else:
    # Simulation mode — dummy weights (all zeros)
    weight_lines = [
        "// DUMMY weights — run train.py with TensorFlow installed",
        "#pragma once",
        "#define WEIGHTS_ENC1 {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}",
        "#define BIAS_ENC1    {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}",
        "#define WEIGHTS_ENC2 {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}",
        "#define BIAS_ENC2    {0.0f,0.0f}",
        "#define WEIGHTS_DEC1 {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}",
        "#define BIAS_DEC1    {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}",
        "#define WEIGHTS_DEC2 {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}",
        "#define BIAS_DEC2    {0.0f,0.0f,0.0f,0.0f}",
    ]
    weights_path = os.path.join(OUTPUT_DIR, "model_weights.h")
    with open(weights_path, "w") as f:
        f.write("\n".join(weight_lines) + "\n")
    print(f"  Saved → {weights_path} (dummy — install TensorFlow for real weights)")
print("\n✅ Done!")
print("   Copy output/model_data.h → firmware/include/model_data.h")
print("   Then: cd firmware && pio run --target upload")
