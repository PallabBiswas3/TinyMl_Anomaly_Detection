"""
Signal pipeline shared by train.py and verify_firmware.py.

ON-DEVICE section  - line-for-line mirror of firmware/src/preprocess.cpp and the
                     AlarmFilter in firmware/src/inference.cpp. Whatever the
                     ESP32 does to incoming data, this does identically, so the
                     model is trained on exactly what it will see in the field.
OFFLINE section    - extra cleaning that only makes sense on a PC with the
                     whole recording available (never runs on the ESP32).

If you change the ON-DEVICE code, change the C++ too and run verify_firmware.py.
"""

import csv
import glob
import os
import re
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):          # Windows cp1252 console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── On-device parameters (exported to model_data.h by train.py) ──────────────

SAMPLE_INTERVAL_MS  = 200     # 5 Hz. HC-SR04 needs >= 60 ms between pings
VALID_MIN_CM        = 2.0     # HC-SR04 blind zone
VALID_MAX_CM        = 400.0   # HC-SR04 rated range
HAMPEL_SIZE         = 7       # spike filter window (odd), 1.4 s
HAMPEL_K_MAD        = 3.0 * 1.4826   # outlier if |x - median| > 3 robust sigmas (MAD·1.4826)...
HAMPEL_MIN_CM       = 1.0     # ...and more than 1 cm (stops normal noise being "corrected")
WINDOW_SIZE         = 8       # feature window = 1.6 s
MAX_GAP_MS          = 1000    # longer gap without a valid reading -> restart filters
ALARM_ON_COUNT      = 3       # consecutive anomalous samples to raise the alarm
ALARM_OFF_COUNT     = 3       # consecutive normal samples to clear it
FAULT_INVALID_COUNT = 5       # consecutive invalid readings (~1 s) -> sensor fault

INPUT_DIM     = 4
FEATURE_NAMES = ["distance_cm", "jitter", "rate_of_change", "variance"]


# ══════════════════════════════════════════════════════════════════════════════
#  ON-DEVICE  (mirror of firmware/src/preprocess.cpp)
# ══════════════════════════════════════════════════════════════════════════════

class Preprocessor:
    """Validity gate → gap reset → Hampel spike filter → window features.

    Why Hampel and not a plain median filter: a median over N samples also
    cancels any vibration whose period is shorter than the window (the samples
    walk around the sine and their median sits at the centre), so vibration
    became invisible. Hampel only replaces a sample that is an outlier relative
    to the spread of its neighbours: an isolated ghost echo is replaced, a
    consistent oscillation passes through.
    """

    def __init__(self, filter_size=HAMPEL_SIZE):
        self.filter_size    = filter_size      # 1 = no filtering (comparison baseline)
        self.invalid_streak = 0
        self.reset()

    def reset(self):
        self.buf     = []      # last `filter_size` valid raw readings
        self.win     = []      # last WINDOW_SIZE filtered readings
        self.last_ts = None
        self.last_f  = 0.0

    @property
    def sensor_fault(self):
        return self.invalid_streak >= FAULT_INVALID_COUNT

    def push(self, raw_cm, ts_ms):
        """Returns (filtered_cm, jitter, rate_of_change, variance) or None if invalid."""
        # 1. Validity gate (also rejects NaN): timeouts (-1), blind zone, over-range
        if not (VALID_MIN_CM <= raw_cm <= VALID_MAX_CM):
            self.invalid_streak += 1
            return None
        self.invalid_streak = 0

        # 2. Long gap (or reboot / timestamp going backwards) -> stale state, restart.
        #    C++ does this with uint32 arithmetic, where "backwards" is a huge gap.
        if self.last_ts is not None and (ts_ms < self.last_ts or ts_ms - self.last_ts > MAX_GAP_MS):
            self.reset()

        # 3. Hampel filter (causal) over the last HAMPEL_SIZE valid readings
        self.buf.append(raw_cm)
        if len(self.buf) > self.filter_size:
            self.buf.pop(0)
        n   = len(self.buf)
        med = sorted(self.buf)[n // 2]
        mad = sorted(abs(v - med) for v in self.buf)[n // 2]
        filtered = med if abs(raw_cm - med) > max(HAMPEL_K_MAD * mad, HAMPEL_MIN_CM) else raw_cm

        # 4. Window features on the filtered signal
        self.win.append(filtered)
        if len(self.win) > WINDOW_SIZE:
            self.win.pop(0)
        n = len(self.win)
        if n >= 2:
            mean = sum(self.win) / n
            var  = sum((v - mean) ** 2 for v in self.win) / n     # population variance
        else:
            var = 0.0

        if self.last_ts is not None and ts_ms > self.last_ts:
            roc = abs(filtered - self.last_f) / ((ts_ms - self.last_ts) / 1000.0)
        else:
            roc = 0.0

        self.last_f  = filtered
        self.last_ts = ts_ms
        return filtered, var ** 0.5, roc, var


class AlarmFilter:
    """Debounce: N anomalous samples in a row raise the alarm, M normal clear it."""

    def __init__(self):
        self.active = False
        self.over   = 0
        self.under  = 0

    def update(self, over_threshold):
        if over_threshold:
            self.over += 1
            self.under = 0
        else:
            self.under += 1
            self.over = 0
        if not self.active and self.over >= ALARM_ON_COUNT:
            self.active = True
        elif self.active and self.under >= ALARM_OFF_COUNT:
            self.active = False
        return self.active


def process_stream(ts, raw, filter_size=HAMPEL_SIZE):
    """Run the on-device preprocessing over one recording.

    Returns feats (n,4) with NaN rows for invalid readings, valid (n,), fault (n,).
    """
    n     = len(ts)
    pre   = Preprocessor(filter_size)
    feats = np.full((n, INPUT_DIM), np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    fault = np.zeros(n, dtype=bool)
    for i in range(n):
        f = pre.push(float(raw[i]), int(ts[i]))
        if f is not None:
            feats[i] = f
            valid[i] = True
        fault[i] = pre.sensor_fault
    return feats, valid, fault


def run_detector(mse, valid, fault, threshold):
    """Mirror of main.cpp's decision logic. Returns alarm (n,) and LED (alarm | fault)."""
    af    = AlarmFilter()
    alarm = np.zeros(len(mse), dtype=bool)
    for i in range(len(mse)):
        if valid[i]:
            af.update(bool(mse[i] > threshold))
        alarm[i] = af.active
    return alarm, alarm | fault


# ══════════════════════════════════════════════════════════════════════════════
#  Model math (mirror of firmware/src/inference.cpp)
# ══════════════════════════════════════════════════════════════════════════════

def forward(layers, x):
    """layers = [(kernel[in][out], bias), ...]; ReLU on all but the last (sigmoid)."""
    h = x.astype(np.float32)
    with np.errstate(over="ignore"):              # expf(+large) = inf -> sigmoid 0, same as C
        for i, (k, b) in enumerate(layers):
            z = h @ k + b
            h = 1.0 / (1.0 + np.exp(-z)) if i == len(layers) - 1 else np.maximum(z, 0.0)
    return h


def reconstruction_mse(layers, norm_min, norm_scale, feats):
    """MSE per row; NaN rows (invalid readings) stay NaN."""
    x   = ((feats - norm_min) * norm_scale).astype(np.float32)
    out = np.full(len(feats), np.nan)
    ok  = ~np.isnan(x).any(axis=1)
    if ok.any():
        out[ok] = np.mean(np.square(x[ok] - forward(layers, x[ok])), axis=1)
    return out


def load_model_headers(include_dir):
    """Parse model_data.h / model_weights.h back into NumPy (used for verification)."""
    def text(name):
        with open(os.path.join(include_dir, name), encoding="latin-1") as f:
            return f.read()

    def floats(s):
        return np.array([float(v.strip().rstrip("f")) for v in s.split(",") if v.strip()],
                        dtype=np.float32)

    w, d = text("model_weights.h"), text("model_data.h")
    get_def = lambda name: floats(re.search(r"#define\s+%s\s*\{([^}]*)\}" % name, w).group(1))
    get_arr = lambda name: floats(re.search(r"%s\[4\]\s*=\s*\{([^}]*)\}" % name, d).group(1))
    shapes  = {"ENC1": (4, 8), "ENC2": (8, 2), "DEC1": (2, 8), "DEC2": (8, 4)}
    layers  = [(get_def("WEIGHTS_" + k).reshape(s), get_def("BIAS_" + k)) for k, s in shapes.items()]
    thr     = float(re.search(r"ANOMALY_THRESHOLD\s*=\s*([-0-9.eE+]+)", d).group(1))
    return layers, get_arr("NORM_MIN"), get_arr("NORM_SCALE"), thr


# ══════════════════════════════════════════════════════════════════════════════
#  Loading recordings
# ══════════════════════════════════════════════════════════════════════════════

class Capture:
    def __init__(self, name, ts, raw, label=None, scenario=None, junk_lines=0, duplicates=0):
        self.name, self.ts, self.raw = name, ts, raw
        self.label, self.scenario    = label, scenario
        self.junk_lines, self.duplicates = junk_lines, duplicates


def load_capture(path):
    """Load a CSV recorded by the firmware in LOG_RAW mode (or the generator).

    Columns: timestamp_ms, distance_cm [, label, scenario]. Tolerates boot
    messages, '#' comments, headers, blank and corrupted lines (counted as junk),
    and exact duplicate rows (dropped).
    """
    ts, raw, label, scen = [], [], [], []
    junk = dups = 0
    prev = None
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#") or row[0].strip() == "timestamp_ms":
                continue
            try:
                t = int(row[0])
                d = float(row[1])
            except (ValueError, IndexError):
                junk += 1                     # header, boot text, garbled serial line
                continue
            if (t, d) == prev:
                dups += 1
                continue
            prev = (t, d)
            ts.append(t)
            raw.append(d)
            if len(row) >= 3:
                label.append(int(row[2]))
                scen.append(row[3] if len(row) >= 4 else "")
    has_label = len(label) == len(ts) and len(ts) > 0
    return Capture(
        os.path.basename(path),
        np.array(ts, dtype=np.int64), np.array(raw, dtype=np.float64),
        np.array(label, dtype=np.int8) if has_label else None,
        np.array(scen) if has_label else None,
        junk, dups,
    )


def load_dir(directory):
    paths = sorted(glob.glob(os.path.join(directory, "*.csv")))
    return [load_capture(p) for p in paths]


def raw_quality(cap):
    """Counts of the raw-data problems in one capture."""
    r  = cap.raw
    dt = np.diff(cap.ts)
    return {
        "rows":        len(r),
        "timeouts":    int(np.sum(r < 0)),
        "blind_zone":  int(np.sum((r >= 0) & (r < VALID_MIN_CM))),
        "over_range":  int(np.sum(r > VALID_MAX_CM)),
        "gaps":        int(np.sum(dt > 1.5 * SAMPLE_INTERVAL_MS)),
        "reboots":     int(np.sum(dt < 0)),
        "junk_lines":  cap.junk_lines,
        "duplicates":  cap.duplicates,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  OFFLINE cleaning (training only)
# ══════════════════════════════════════════════════════════════════════════════

CONTAM_Z       = 6.0    # robust z-score limit on filtered distance
CONTAM_MIN_CM  = 2.0    # ...but never flag deviations smaller than this
CONTAM_PAD_PRE  = HAMPEL_SIZE                # samples before an event to drop
CONTAM_PAD_POST = WINDOW_SIZE + HAMPEL_SIZE  # window "memory" after an event


def contamination_mask(feats, valid):
    """True for rows of a 'normal' recording that are not actually normal.

    Real normal recordings always contain a few events (someone reaches into the
    beam, the object gets bumped). Training on them would teach the model that
    those events are normal. A row is contamination if its filtered distance is
    far from the recording's robust baseline (median ± 6·MAD, min 2 cm); the
    neighbouring rows whose feature window overlaps the event are dropped too.
    """
    d    = feats[:, 0]
    dv   = d[valid]
    med  = np.median(dv)
    mad  = 1.4826 * np.median(np.abs(dv - med))
    lim  = max(CONTAM_Z * mad, CONTAM_MIN_CM)
    bad  = valid & (np.abs(d - med) > lim)

    drop = bad.copy()
    for i in np.flatnonzero(bad):
        drop[max(0, i - CONTAM_PAD_PRE): i + CONTAM_PAD_POST + 1] = True
    return drop & valid


# ══════════════════════════════════════════════════════════════════════════════
#  Event-based evaluation
# ══════════════════════════════════════════════════════════════════════════════

# The alarm legitimately outlasts an event: Hampel step delay + window memory
# + ALARM_OFF_COUNT samples ≈ (3 + 8 + 3) × 200 ms = 2.8 s.
EVENT_GRACE_MS = 3000


def evaluate_events(cap, led):
    """Per-event detection + false alarms per hour on a labelled capture."""
    ts, lab, scen = cap.ts, cap.label.astype(bool), cap.scenario
    edges  = np.flatnonzero(np.diff(np.r_[0, lab.astype(int), 0]))
    events = list(zip(edges[::2], edges[1::2] - 1))     # inclusive index ranges

    in_event = np.zeros(len(ts), dtype=bool)
    results  = []
    for i0, i1 in events:
        win = (ts >= ts[i0]) & (ts <= ts[i1] + EVENT_GRACE_MS)
        in_event |= win
        hit = np.flatnonzero(win & led)
        results.append({
            "scenario": scen[i0],
            "detected": len(hit) > 0,
            "latency_ms": int(ts[hit[0]] - ts[i0]) if len(hit) else None,
        })

    rising       = np.flatnonzero(led & ~np.r_[False, led[:-1]])
    false_alarms = int(np.sum(~in_event[rising]))
    normal       = ~in_event
    normal_hours = np.sum(normal) * SAMPLE_INTERVAL_MS / 3.6e6
    return {
        "events": results,
        "false_alarms": false_alarms,
        "false_alarms_per_hour": false_alarms / normal_hours if normal_hours else 0.0,
        "normal_led_pct": 100.0 * np.mean(led[normal]) if normal.any() else 0.0,
        "normal_hours": normal_hours,
    }
