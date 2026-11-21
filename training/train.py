"""
ESP32 Ultrasonic Anomaly Detector - Training Pipeline
======================================================
Raw recordings  ──▶  clean  ──▶  features  ──▶  autoencoder  ──▶  C headers

  1. Load raw captures (firmware LOG_RAW format) and report their problems
  2. On-device cleaning, mirrored exactly from the firmware (pipeline.py):
       validity gate → gap reset → Hampel spike filter → 8-sample window features
  3. Offline cleaning (PC only): drop contaminated stretches from the normal data
  4. Train a 4-8-2-8-4 autoencoder on normal data only (unsupervised)
  5. Pick the threshold, then evaluate the *complete* device logic
     (cleaning + model + alarm debounce + fault detection) on labelled test
     recordings, per anomaly event
  6. Export model_data.h / model_weights.h, verify them, copy into firmware/include

Usage:
  python train.py                       # uses data/normal/*.csv and data/test/*.csv
  python train.py --normal-dir my_caps  # your own real recordings
  python train.py --compare-raw         # also train a model WITHOUT cleaning, for comparison
"""

import argparse
import os
import shutil
import sys

import numpy as np

import pipeline as P

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    sys.exit("TensorFlow is required. Run: pip install -r requirements.txt")

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR       = os.path.join(SCRIPT_DIR, "output")
FIRMWARE_INCLUDE = os.path.join(SCRIPT_DIR, "..", "firmware", "include")

LATENT_DIM      = 2
EPOCHS          = 200
BATCH_SIZE      = 64
VAL_FRACTION    = 0.2       # last 20 % of every recording (chronological split)
# Threshold = this percentile of reconstruction MSE over all cleaned normal data.
# Normal MSE is heavy-tailed, so mean + 3σ sits inside the tail and causes
# frequent false alarms; a high percentile + the alarm debounce does not.
THRESHOLD_PERCENTILE = 99.9
SEED            = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(title):
    print(f"\n── {title} " + "─" * max(0, 62 - len(title)))


def build_dataset(captures, filter_size, offline_clean):
    """Features for training/validation from normal captures."""
    train, val = [], []
    print(f"  {'capture':<26}{'valid':>7}{'contam.':>9}{'kept':>7}")
    for cap in captures:
        feats, valid, _ = P.process_stream(cap.ts, cap.raw, filter_size)
        keep = valid.copy()
        dropped = 0
        if offline_clean:
            contam  = P.contamination_mask(feats, valid)
            dropped = int(contam.sum())
            keep   &= ~contam
        rows  = feats[keep].astype(np.float32)
        split = int(len(rows) * (1 - VAL_FRACTION))
        train.append(rows[:split])
        val.append(rows[split:])
        print(f"  {cap.name:<26}{int(valid.sum()):>7}{dropped:>9}{len(rows):>7}")
    return np.vstack(train), np.vstack(val)


def build_autoencoder():
    inp    = keras.Input(shape=(P.INPUT_DIM,), name="sensor_input")
    x      = keras.layers.Dense(8,          activation="relu",    name="enc1")(inp)
    latent = keras.layers.Dense(LATENT_DIM, activation="relu",    name="latent")(x)
    x      = keras.layers.Dense(8,          activation="relu",    name="dec1")(latent)
    out    = keras.layers.Dense(P.INPUT_DIM, activation="sigmoid", name="out")(x)
    return keras.Model(inp, out, name="ultrasonic_autoencoder")


LAYER_NAMES = [("enc1", "ENC1"), ("latent", "ENC2"), ("dec1", "DEC1"), ("out", "DEC2")]


def train_model(train_raw, val_raw):
    """Returns (keras model, layers, norm_min, norm_scale, threshold, val_mse)."""
    norm_min   = train_raw.min(axis=0)
    rng_       = train_raw.max(axis=0) - norm_min
    norm_scale = np.where(rng_ > 1e-12, 1.0 / np.maximum(rng_, 1e-12), 0.0).astype(np.float32)
    X_train    = ((train_raw - norm_min) * norm_scale).astype(np.float32)
    X_val      = ((val_raw   - norm_min) * norm_scale).astype(np.float32)

    tf.keras.utils.set_random_seed(SEED)
    model = build_autoencoder()
    model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")
    hist = model.fit(
        X_train, X_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_val, X_val), verbose=0,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-4),
            keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True),
        ],
    )
    print(f"  epochs: {len(hist.history['loss'])}   best val loss: {min(hist.history['val_loss']):.6f}")

    layers = [tuple(w.astype(np.float32) for w in model.get_layer(n).get_weights())
              for n, _ in LAYER_NAMES]
    val_mse   = P.reconstruction_mse(layers, norm_min, norm_scale, val_raw)
    all_mse   = P.reconstruction_mse(layers, norm_min, norm_scale, np.vstack([train_raw, val_raw]))
    threshold = float(np.float32(np.percentile(all_mse, THRESHOLD_PERCENTILE)))
    return model, layers, norm_min.astype(np.float32), norm_scale, threshold, val_mse


def evaluate(test_caps, layers, norm_min, norm_scale, threshold, filter_size):
    """Run the full device logic over labelled captures, aggregate per scenario."""
    per_scen, fa, hours, led_pct = {}, 0, 0.0, []
    for cap in test_caps:
        feats, valid, fault = P.process_stream(cap.ts, cap.raw, filter_size)
        mse        = P.reconstruction_mse(layers, norm_min, norm_scale, feats)
        _, led     = P.run_detector(mse, valid, fault, threshold)
        res        = P.evaluate_events(cap, led)
        fa        += res["false_alarms"]
        hours     += res["normal_hours"]
        led_pct.append(res["normal_led_pct"])
        for ev in res["events"]:
            per_scen.setdefault(ev["scenario"], []).append(ev)
    return per_scen, fa, hours, float(np.mean(led_pct)) if led_pct else 0.0


def print_eval(label, per_scen, fa, hours, led_pct):
    print(f"\n  [{label}]")
    print(f"  {'scenario':<18}{'detected':>10}{'avg latency':>14}")
    for sc in sorted(per_scen):
        evs  = per_scen[sc]
        hits = [e for e in evs if e["detected"]]
        lat  = f"{np.mean([e['latency_ms'] for e in hits]) / 1000:.1f} s" if hits else "-"
        print(f"  {sc:<18}{len(hits):>5}/{len(evs):<4}{lat:>14}")
    print(f"  false alarms      : {fa} in {hours * 60:.1f} min of normal operation "
          f"({fa / hours if hours else 0:.1f}/hour)")
    print(f"  LED on while normal: {led_pct:.2f}% of samples")


# ── Header export ─────────────────────────────────────────────────────────────

def fmt(v):
    """C float literal. 9 significant digits round-trips float32 exactly."""
    s = f"{float(v):.9g}"
    if not any(c in s for c in ".en"):   # "0" -> "0.0" ("0f" is not valid C)
        s += ".0"
    return s + "f"


def c_list(arr):
    return ", ".join(fmt(v) for v in np.asarray(arr).flatten())


def write_headers(layers, norm_min, norm_scale, threshold, train_raw):
    norm_max = train_raw.max(axis=0)
    data_lines = [
        "// Auto-generated by train.py - DO NOT EDIT",
        "// Ultrasonic anomaly detector - on-device pipeline settings, threshold, normalization",
        "#pragma once",
        "",
        "// ── On-device signal pipeline (the model was trained on exactly this) ──",
        f"#define MODEL_SAMPLE_INTERVAL_MS  {P.SAMPLE_INTERVAL_MS}",
        f"#define MODEL_VALID_MIN_CM        {fmt(P.VALID_MIN_CM)}",
        f"#define MODEL_VALID_MAX_CM        {fmt(P.VALID_MAX_CM)}",
        f"#define MODEL_HAMPEL_SIZE         {P.HAMPEL_SIZE}",
        f"#define MODEL_HAMPEL_K_MAD        {fmt(P.HAMPEL_K_MAD)}",
        f"#define MODEL_HAMPEL_MIN_CM       {fmt(P.HAMPEL_MIN_CM)}",
        f"#define MODEL_WINDOW_SIZE         {P.WINDOW_SIZE}",
        f"#define MODEL_MAX_GAP_MS          {P.MAX_GAP_MS}",
        f"#define MODEL_ALARM_ON_COUNT      {P.ALARM_ON_COUNT}",
        f"#define MODEL_ALARM_OFF_COUNT     {P.ALARM_OFF_COUNT}",
        f"#define MODEL_FAULT_INVALID_COUNT {P.FAULT_INVALID_COUNT}",
        "",
        "// Reconstruction-MSE above this value => anomalous sample",
        f"const float ANOMALY_THRESHOLD = {fmt(threshold)};",
        "",
        "// Normalization: x_norm = (x - NORM_MIN) * NORM_SCALE",
        f"// (training max: {', '.join(f'{v:.4f}' for v in norm_max)})",
        f"const float NORM_MIN[4]   = {{{c_list(norm_min)}}};",
        f"const float NORM_SCALE[4] = {{{c_list(norm_scale)}}};",
        "",
        "// Feature indices",
        "#define FEAT_DISTANCE       0   // filtered distance, cm",
        "#define FEAT_JITTER         1   // std dev over window",
        "#define FEAT_RATE_OF_CHANGE 2   // |delta| per second",
        "#define FEAT_VARIANCE       3   // variance over window",
    ]
    weight_lines = [
        "// Auto-generated by train.py - DO NOT EDIT",
        "// Autoencoder 4-8-2-8-4. Kernels are row-major [in][out] (Keras Dense layout).",
        "#pragma once",
        "",
    ]
    for (k, b), (_, c_name) in zip(layers, LAYER_NAMES):
        weight_lines.append(f"#define WEIGHTS_{c_name} {{{c_list(k)}}}")
        weight_lines.append(f"#define BIAS_{c_name} {{{c_list(b)}}}")

    paths = []
    for name, lines in (("model_data.h", data_lines), ("model_weights.h", weight_lines)):
        path = os.path.join(OUTPUT_DIR, name)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  Saved → {os.path.relpath(path, SCRIPT_DIR)}")
        paths.append(path)
    return paths


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--normal-dir", default=os.path.join(SCRIPT_DIR, "data", "normal"),
                    help="folder of normal-operation captures (*.csv)")
    ap.add_argument("--test-dir", default=os.path.join(SCRIPT_DIR, "data", "test"),
                    help="folder of labelled test captures (*.csv); optional")
    ap.add_argument("--compare-raw", action="store_true",
                    help="also train/evaluate a model without any cleaning")
    ap.add_argument("--no-copy", action="store_true", help="don't copy headers into firmware/include")
    args = ap.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load ------------------------------------------------------------------
    banner("Step 1: Raw data quality")
    normal_caps = P.load_dir(args.normal_dir)
    test_caps   = [c for c in P.load_dir(args.test_dir) if c.label is not None] \
        if os.path.isdir(args.test_dir) else []
    if not normal_caps:
        sys.exit(f"No *.csv captures in {args.normal_dir}. Run generate_data.py or capture_serial.py.")

    keys = ["rows", "timeouts", "blind_zone", "over_range", "gaps", "reboots", "junk_lines", "duplicates"]
    print(f"  {'capture':<26}" + "".join(f"{k[:9]:>10}" for k in keys))
    for cap in normal_caps + test_caps:
        q = P.raw_quality(cap)
        print(f"  {cap.name:<26}" + "".join(f"{q[k]:>10}" for k in keys))

    # 2-3. Clean → features ----------------------------------------------------
    banner("Step 2-3: On-device cleaning + offline contamination removal")
    print(f"  on-device : valid {P.VALID_MIN_CM:g}-{P.VALID_MAX_CM:g} cm, Hampel-{P.HAMPEL_SIZE} spike filter, "
          f"window {P.WINDOW_SIZE}, reset after {P.MAX_GAP_MS} ms gap")
    train_raw, val_raw = build_dataset(normal_caps, P.HAMPEL_SIZE, offline_clean=True)
    print(f"  train rows: {len(train_raw)}   val rows: {len(val_raw)}")

    # 4. Train -----------------------------------------------------------------
    banner("Step 4: Training autoencoder (normal data only)")
    model, layers, norm_min, norm_scale, threshold, val_mse = train_model(train_raw, val_raw)
    model.summary(print_fn=lambda s: print("  " + s) if "params" in s else None)

    # 5. Threshold + evaluation ------------------------------------------------
    banner("Step 5: Threshold & full-pipeline evaluation")
    print(f"  val MSE   : {np.mean(val_mse):.6f} ± {np.std(val_mse):.6f}")
    print(f"  threshold : {threshold:.6f}  ({THRESHOLD_PERCENTILE:g}th percentile of normal MSE; "
          f"{100 * np.mean(val_mse > threshold):.2f}% of val samples above it)")
    print(f"  alarm     : {P.ALARM_ON_COUNT} anomalous samples to raise, "
          f"{P.ALARM_OFF_COUNT} normal to clear; fault after {P.FAULT_INVALID_COUNT} invalid")
    if test_caps:
        print_eval("with cleaning (deployed)",
                   *evaluate(test_caps, layers, norm_min, norm_scale, threshold, P.HAMPEL_SIZE))
        if args.compare_raw:
            print("\n  training comparison model without cleaning...")
            b_train, b_val = build_dataset(normal_caps, filter_size=1, offline_clean=False)
            _, b_layers, b_min, b_scale, b_thr, _ = train_model(b_train, b_val)
            print_eval("WITHOUT cleaning (comparison only)",
                       *evaluate(test_caps, b_layers, b_min, b_scale, b_thr, filter_size=1))
    else:
        print("  (no labelled test captures - skipping evaluation)")

    # 6. Export + verify + copy --------------------------------------------------
    banner("Step 6: Export, verify, copy")
    paths = write_headers(layers, norm_min, norm_scale, threshold, train_raw)

    x_val = ((val_raw - norm_min) * norm_scale).astype(np.float32)
    keras_diff = float(np.max(np.abs(P.forward(layers, x_val) - model.predict(x_val, verbose=0))))
    h_layers, h_min, h_scale, h_thr = P.load_model_headers(OUTPUT_DIR)
    header_diff = float(np.nanmax(np.abs(
        P.reconstruction_mse(h_layers, h_min, h_scale, val_raw) - val_mse)))
    print(f"  C-forward vs Keras   max |diff| = {keras_diff:.2e}")
    print(f"  headers round-trip   max |diff| = {header_diff:.2e}")
    if keras_diff > 1e-4 or header_diff > 1e-6 or np.float32(h_thr) != np.float32(threshold):
        sys.exit("  ✗ Exported model does not match - not copying to firmware.")

    if args.no_copy:
        print("  (--no-copy: firmware/include left untouched)")
    elif os.path.isdir(FIRMWARE_INCLUDE):
        for src in paths:
            dst = os.path.join(FIRMWARE_INCLUDE, os.path.basename(src))
            shutil.copyfile(src, dst)
            print(f"  Copied → {os.path.normpath(dst)}")
    else:
        print("  firmware/include not found - copy the headers manually.")

    print("\n✅ Done!  Next: python verify_firmware.py   then   cd ../firmware && pio run -t upload")


if __name__ == "__main__":
    main()
