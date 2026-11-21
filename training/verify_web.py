"""
Check the web dashboard's JavaScript pipeline (web/pipeline.js + web/model.js)
against pipeline.py, sample by sample, on every data/test/*.csv capture.

Requires Node.js on PATH. Run export_web.py first.
"""

import glob
import json
import os
import subprocess
import sys

import numpy as np

import pipeline as P

HERE        = os.path.dirname(os.path.abspath(__file__))
INCLUDE_DIR = os.path.join(HERE, "..", "firmware", "include")
WEB_DIR     = os.path.join(HERE, "..", "web")

NODE_HARNESS = r"""
const fs = require("fs"), path = require("path"), vm = require("vm");
const web = process.argv[1];
const ctx = {}; vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(web, "model.js"), "utf8"), ctx);
const Pipe = require(path.join(web, "pipeline.js"));
const { ts, raw } = JSON.parse(fs.readFileSync(0, "utf8"));
const det = new Pipe.Detector(ctx.ESP32_MODEL);
const out = { mse: [], filtered: [], led: [] };
for (let i = 0; i < ts.length; i++) {
  const r = det.step(raw[i], ts[i]);
  out.mse.push(r.valid ? r.mse : null);
  out.filtered.push(r.valid ? r.feats[0] : null);
  out.led.push(r.led ? 1 : 0);
}
process.stdout.write(JSON.stringify(out));
"""


def main():
    print("── Verify web pipeline (JS) against pipeline.py ──")
    layers, nmin, nscale, thr = P.load_model_headers(INCLUDE_DIR)
    ok = True
    for path in sorted(glob.glob(os.path.join(HERE, "data", "test", "*.csv"))):
        cap = P.load_capture(path)
        feats, valid, fault = P.process_stream(cap.ts, cap.raw)
        mse = P.reconstruction_mse(layers, nmin, nscale, feats)
        _, led = P.run_detector(mse, valid, fault, thr)

        res = subprocess.run(
            ["node", "-e", NODE_HARNESS, os.path.abspath(WEB_DIR)],
            input=json.dumps({"ts": cap.ts.tolist(), "raw": cap.raw.tolist()}),
            capture_output=True, text=True, check=True)
        js = json.loads(res.stdout)

        js_mse = np.array([np.nan if v is None else v for v in js["mse"]])
        js_f   = np.array([np.nan if v is None else v for v in js["filtered"]])
        js_led = np.array(js["led"], dtype=bool)

        valid_match = np.array_equal(~np.isnan(js_mse), valid)
        f_diff   = float(np.nanmax(np.abs(js_f - feats[:, 0])))
        rel_diff = float(np.nanmax(np.abs(js_mse - mse) / np.maximum(np.abs(mse), 1e-6)))
        led_mis  = int(np.sum(js_led != led))
        good = valid_match and f_diff < 1e-9 and rel_diff < 1e-4 and led_mis == 0
        ok &= good
        print(f"  {'✓' if good else '✗'} {os.path.basename(path)}: {len(cap.ts)} rows | "
              f"filtered max|diff| {f_diff:.1e} | mse max rel diff {rel_diff:.1e} | "
              f"LED mismatches {led_mis}")

    print("\n✅ JS web pipeline matches Python" if ok else "\n✗ JS web pipeline DIFFERS from Python")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
