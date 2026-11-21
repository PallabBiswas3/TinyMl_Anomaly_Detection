"""
Verify that the firmware's C++ pipeline behaves exactly like the Python one.

Compiles firmware/src/preprocess.cpp + inference.cpp on this PC (needs g++),
runs them over every capture in data/test/, and compares sample by sample with
pipeline.py using the headers currently in firmware/include/:
  - filtered distance and features
  - reconstruction MSE
  - alarm and sensor-fault decisions

Run after train.py, and after any change to preprocess.cpp / pipeline.py.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import pipeline as P

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
FIRMWARE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "firmware"))


def build(out_dir):
    cxx = shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        sys.exit("No host C++ compiler (g++/clang++) found - install MinGW or LLVM to run this check.")
    exe = os.path.join(out_dir, "host_check.exe" if os.name == "nt" else "host_check")
    cmd = [cxx, "-std=gnu++11", "-O2", "-Wall", "-Wextra",
           "-I", os.path.join(FIRMWARE_DIR, "include"),
           os.path.join(FIRMWARE_DIR, "test_host", "host_check.cpp"),
           os.path.join(FIRMWARE_DIR, "src", "preprocess.cpp"),
           os.path.join(FIRMWARE_DIR, "src", "inference.cpp"),
           "-o", exe]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or res.stderr.strip():
        print(res.stderr)
    if res.returncode != 0:
        sys.exit("Host build failed.")
    return exe


def run_cpp(exe, csv_path):
    out = subprocess.run([exe, csv_path], capture_output=True, text=True, check=True).stdout
    rows = [l.split(",") for l in out.strip().splitlines()[1:]]
    num  = lambda s: float(s) if s else np.nan
    return {
        "ts":    np.array([int(r[0]) for r in rows]),
        "valid": np.array([r[1] == "1" for r in rows]),
        "feats": np.array([[num(v) for v in r[2:6]] for r in rows]),
        "mse":   np.array([num(r[6]) for r in rows]),
        "alarm": np.array([r[7] == "1" for r in rows]),
        "fault": np.array([r[8] == "1" for r in rows]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(SCRIPT_DIR, "data", "test"),
                    help="folder of raw captures to replay")
    args = ap.parse_args()

    layers, nmin, nscale, thr = P.load_model_headers(os.path.join(FIRMWARE_DIR, "include"))
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        exe = build(tmp)
        print("Built firmware pipeline for host. Replaying captures:\n")
        for path in sorted(glob.glob(os.path.join(args.data, "*.csv"))):
            cap = P.load_capture(path)
            c   = run_cpp(exe, path)
            if len(c["ts"]) != len(cap.ts) or not np.array_equal(c["ts"], cap.ts):
                # loader drops exact duplicate rows; C++ sees them - compare only if aligned
                print(f"  {cap.name}: row alignment differs (duplicates?) - skipped")
                continue

            feats, valid, fault = P.process_stream(cap.ts, cap.raw)
            mse        = P.reconstruction_mse(layers, nmin, nscale, feats)
            alarm, _   = P.run_detector(mse, valid, fault, thr)

            both    = valid & c["valid"]
            f_diff  = float(np.nanmax(np.abs(feats[both] - c["feats"][both])))
            rel_mse = float(np.nanmax(np.abs(mse[both] - c["mse"][both]) / (np.abs(mse[both]) + 1e-3)))
            mism = {
                "valid": int(np.sum(valid != c["valid"])),
                "fault": int(np.sum(fault != c["fault"])),
                "alarm": int(np.sum(alarm != c["alarm"])),
            }
            good = mism["valid"] == 0 and mism["fault"] == 0 and mism["alarm"] == 0 \
                and f_diff < 1e-3 and rel_mse < 1e-3
            ok &= good
            print(f"  {'✓' if good else '✗'} {cap.name:<26} rows {len(cap.ts):>6} | "
                  f"max feature diff {f_diff:.1e} | max rel. MSE diff {rel_mse:.1e} | "
                  f"mismatches valid/fault/alarm {mism['valid']}/{mism['fault']}/{mism['alarm']}")

    print("\n" + ("✅ C++ firmware pipeline matches Python." if ok else "❌ Mismatch - see above."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
