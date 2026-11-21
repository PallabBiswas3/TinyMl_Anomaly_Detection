"""
Generate realistic *raw* HC-SR04 recordings - noise, glitches and all.

The files have exactly the format the firmware prints in LOG_RAW mode
(timestamp_ms,distance_cm with -1 for "no echo"), so a real capture made with
capture_serial.py can replace any of them 1:1.

Physical / electrical effects modelled
--------------------------------------
  measurement noise      Gaussian, per-session level (surface, angle, supply ripple)
  temperature drift      speed of sound = 331.3 + 0.606·T m/s; firmware assumes 343 m/s
  echo quantization      pulseIn() returns whole microseconds (0.017 cm steps)
  dropped echoes         timeouts (-1), single and in short bursts; worse at long range
  ghost echoes / spikes  multipath double-bounce (2x distance), side reflectors, random junk
  blind-zone garbage     transducer ringing gives readings < 2 cm
  over-range             readings just above 400 cm
  timing jitter          loop jitter + occasional serial stalls in the timestamps
  lost lines             serial capture occasionally drops a line
  contamination          (normal files) people reaching into the beam / object bumped -
                         real "normal" recordings are never perfectly clean

Outputs (training/data/):
  normal/synthetic_session_N.csv   unlabeled "normal operation" captures
  test/synthetic_test_01.csv       labeled stream with anomaly events
                                   (extra columns: label, scenario)
"""

import argparse
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):          # Windows cp1252 console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")

SAMPLE_MS     = 200
CM_PER_US     = 0.01715        # firmware: distance = echo_us * 0.01715 (343 m/s)
C_ASSUMED     = 343.0
NOMINAL_CM    = 30.0           # where the monitored object normally sits
TIMEOUT_CM    = 25000 * CM_PER_US   # pulseIn timeout (~428 cm)


def speed_of_sound(temp_c):
    return 331.3 + 0.606 * temp_c


class Track:
    """Ground-truth description of one recording, sample by sample."""

    def __init__(self):
        self.dist, self.noise_mult, self.drop_mult = [], [], []
        self.no_echo, self.label, self.scenario = [], [], []

    def add(self, n, dist, noise_mult=1.0, drop_mult=1.0, no_echo=False,
            label=0, scenario="normal"):
        dist = np.broadcast_to(np.asarray(dist, dtype=float), (n,))
        self.dist.append(dist)
        self.noise_mult.append(np.full(n, noise_mult))
        self.drop_mult.append(np.full(n, drop_mult))
        self.no_echo.append(np.full(n, no_echo))
        self.label.append(np.full(n, label))
        self.scenario.append(np.full(n, scenario, dtype=object))

    def arrays(self):
        cat = np.concatenate
        return (cat(self.dist), cat(self.noise_mult), cat(self.drop_mult),
                cat(self.no_echo), cat(self.label), cat(self.scenario))


def normal_distance(rng, n, base):
    """Object at rest: tiny slow wander (settling, mount flex) around `base`."""
    walk = np.cumsum(rng.normal(0, 0.004, n))
    walk -= np.linspace(0, walk[-1], n)              # keep it bounded
    return base + np.clip(walk, -0.3, 0.3)


def render(rng, track, noise_cm, temp_c, temp_amp, start_ms=0):
    """Turn a ground-truth track into what the ESP32 would actually log."""
    dist, noise_mult, drop_mult, no_echo, label, scen = track.arrays()
    n = len(dist)
    t = np.arange(n) * SAMPLE_MS / 1000.0

    # Temperature: slow daily-ish swing + small random walk
    temp = (temp_c + temp_amp * np.sin(2 * np.pi * t / rng.uniform(1800, 5400) + rng.uniform(0, 6.3))
            + np.cumsum(rng.normal(0, 0.01, n)))
    meas = dist * speed_of_sound(temp) / C_ASSUMED

    # Measurement noise, then whole-microsecond quantization of the echo time
    meas = meas + rng.normal(0, 1, n) * noise_cm * noise_mult
    raw  = np.round(meas / CM_PER_US) * CM_PER_US

    # Ghost echoes / spikes (~0.8 %)
    spike = rng.random(n) < 0.008
    kind  = rng.integers(0, 3, n)
    raw = np.where(spike & (kind == 0), 2.0 * meas, raw)                          # double bounce
    raw = np.where(spike & (kind == 1), meas - rng.uniform(3, 15, n), raw)        # side reflector
    raw = np.where(spike & (kind == 2), rng.uniform(2, 400, n), raw)              # junk

    # Blind-zone ringing (< 2 cm) and over-range readings
    raw = np.where(rng.random(n) < 0.002, rng.uniform(0.2, 1.9, n), raw)
    raw = np.where(rng.random(n) < 0.001, rng.uniform(400.5, TIMEOUT_CM, n), raw)

    # Dropped echoes: single (~1 %) and bursts of 2-4, more frequent at long range
    drop = rng.random(n) < 0.010 * drop_mult
    for s in np.flatnonzero(rng.random(n) < 0.002 * drop_mult):
        drop[s: s + rng.integers(2, 5)] = True
    raw = np.where(drop | no_echo | (raw > TIMEOUT_CM), -1.0, raw)

    # Timestamps: 200 ms schedule + loop jitter + occasional serial stalls
    ts = start_ms + np.arange(n) * SAMPLE_MS + rng.integers(0, 3, n)
    ts = ts + np.where(rng.random(n) < 0.01, rng.integers(5, 16, n), 0)

    # Serial capture loses ~0.1 % of lines
    keep = rng.random(n) >= 0.001
    return ts[keep], np.round(raw[keep], 2), label[keep], scen[keep]


def write_csv(path, ts, raw, label=None, scen=None, comments=()):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        for c in comments:
            f.write(f"# {c}\n")
        if label is None:
            f.write("timestamp_ms,distance_cm\n")
            for t, d in zip(ts, raw):
                f.write(f"{t},{d:.2f}\n")
        else:
            f.write("timestamp_ms,distance_cm,label,scenario\n")
            for t, d, l, s in zip(ts, raw, label, scen):
                f.write(f"{t},{d:.2f},{l},{s}\n")
    print(f"  {os.path.relpath(path, SCRIPT_DIR)}  ({len(ts)} rows)")


def secs(s):
    return int(round(s * 1000 / SAMPLE_MS))


# ── Normal sessions ───────────────────────────────────────────────────────────

SESSIONS = [
    # minutes, noise_cm, temp_c, temp_amp, base_cm, boot delay before first sample
    dict(minutes=10, noise=0.15, temp=22.0, amp=1.0, base=30.0, boot_ms=1210),
    dict(minutes=10, noise=0.25, temp=27.0, amp=2.0, base=30.2, boot_ms=1185),
    dict(minutes=10, noise=0.20, temp=19.0, amp=1.5, base=29.8, boot_ms=1240),
]


def make_normal_session(rng, cfg):
    n     = secs(cfg["minutes"] * 60)
    track = Track()
    notes = []
    pos   = 0
    # 3 contamination events at random times
    events = sorted(rng.choice(np.arange(secs(30), n - secs(30)), 3, replace=False))
    for ev in events:
        track.add(ev - pos, normal_distance(rng, ev - pos, cfg["base"]))
        k = secs(rng.uniform(1.0, 3.0))
        if rng.random() < 0.6:                       # hand / person in the beam
            track.add(k, rng.uniform(8, 22) + rng.normal(0, 1.5, k))
            what = "hand in beam"
        else:                                        # object bumped, wobbles back
            tt = np.arange(k) * SAMPLE_MS / 1000
            track.add(k, cfg["base"] + rng.uniform(3, 6) * np.exp(-tt) * np.cos(2 * np.pi * 1.3 * tt))
            what = "object bumped"
        notes.append(f"injected contamination: {what} at sample {ev} ({k} samples)")
        pos = ev + k
    track.add(n - pos, normal_distance(rng, n - pos, cfg["base"]))
    ts, raw, _, _ = render(rng, track, cfg["noise"], cfg["temp"], cfg["amp"], cfg["boot_ms"])
    return ts, raw, notes


# ── Labeled test stream ───────────────────────────────────────────────────────

def make_test_stream(rng):
    track = Track()
    base  = NOMINAL_CM

    def normal(seconds):
        k = secs(seconds)
        track.add(k, normal_distance(rng, k, base))

    scenarios = ["obstruction", "object_removed", "vibration", "displaced", "sensor_fault"] * 4
    rng.shuffle(scenarios)
    normal(60)
    for sc in scenarios:
        k = secs(rng.uniform(4, 10))
        if sc == "obstruction":                      # something blocks the beam
            track.add(k, rng.uniform(5, 15) + rng.normal(0, 0.5, k), label=1, scenario=sc)
        elif sc == "object_removed":                 # echo now comes from the back wall
            track.add(k, rng.uniform(120, 200) + rng.normal(0, 0.5, k),
                      noise_mult=3.0, drop_mult=4.0, label=1, scenario=sc)
        elif sc == "vibration":                      # loose mount / machine vibration
            tt  = np.arange(k) * SAMPLE_MS / 1000
            amp = rng.uniform(1.5, 4.0)
            track.add(k, base + amp * np.sin(2 * np.pi * rng.uniform(0.7, 2.3) * tt),
                      label=1, scenario=sc)
        elif sc == "displaced":                      # object shifted and stays shifted
            track.add(k, base + rng.choice([-1, 1]) * rng.uniform(4, 10), label=1, scenario=sc)
        elif sc == "sensor_fault":                   # cable loose / transducer covered
            k = secs(rng.uniform(3, 6))
            track.add(k, base, no_echo=True, label=1, scenario=sc)
        normal(rng.uniform(25, 60))
    return render(rng, track, noise_cm=0.2, temp_c=24.0, temp_amp=2.0, start_ms=1200)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("── Generating synthetic raw HC-SR04 recordings ─────────────────")
    for i, cfg in enumerate(SESSIONS, 1):
        ts, raw, notes = make_normal_session(rng, cfg)
        write_csv(os.path.join(DATA_DIR, "normal", f"synthetic_session_{i}.csv"), ts, raw,
                  comments=[f"SYNTHETIC normal capture {i}: object at ~{cfg['base']} cm, "
                            f"noise {cfg['noise']} cm, {cfg['temp']} +/- {cfg['amp']} degC",
                            *notes])

    ts, raw, label, scen = make_test_stream(rng)
    write_csv(os.path.join(DATA_DIR, "test", "synthetic_test_01.csv"), ts, raw, label, scen,
              comments=["SYNTHETIC labeled test stream (label 1 = anomaly event)"])

    # 1-hour soak: normal only, to measure false alarms per hour meaningfully
    track = Track()
    n = secs(3600)
    track.add(n, normal_distance(rng, n, NOMINAL_CM))
    ts, raw, label, scen = render(rng, track, noise_cm=0.22, temp_c=25.0, temp_amp=2.5, start_ms=1190)
    write_csv(os.path.join(DATA_DIR, "test", "synthetic_soak_1h.csv"), ts, raw, label, scen,
              comments=["SYNTHETIC 1-hour normal soak test (no anomalies; measures false alarms)"])
    print("\nReplace these with real captures (capture_serial.py) for a real deployment.")


if __name__ == "__main__":
    main()
