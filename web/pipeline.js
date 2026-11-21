// On-device pipeline, JavaScript mirror.
//
// Line-for-line port of training/pipeline.py (Preprocessor, AlarmFilter, forward,
// run_detector), which itself mirrors firmware/src/preprocess.cpp + inference.cpp.
// Checked sample-by-sample against Python by training/verify_web.py.
// Works in the browser (window.Esp32Pipeline) and in Node (module.exports).

(function (root) {
  "use strict";

  // ── Preprocessor: validity gate → gap reset → Hampel → window features ─────
  function Preprocessor(P) {
    this.P = P;
    this.invalidStreak = 0;
    this.reset();
  }
  Preprocessor.prototype.reset = function () {
    this.buf = [];
    this.win = [];
    this.lastTs = null;
    this.lastF = 0.0;
  };
  Preprocessor.prototype.sensorFault = function () {
    return this.invalidStreak >= this.P.FAULT_INVALID_COUNT;
  };
  // Returns [filtered, jitter, rate_of_change, variance] or null if invalid.
  Preprocessor.prototype.push = function (raw, ts) {
    var P = this.P;
    // 1. Validity gate (rejects NaN too)
    if (!(raw >= P.VALID_MIN_CM && raw <= P.VALID_MAX_CM)) {
      this.invalidStreak++;
      return null;
    }
    this.invalidStreak = 0;

    // 2. Long gap / time going backwards -> restart
    this.lastReset = false;
    if (this.lastTs !== null && (ts < this.lastTs || ts - this.lastTs > P.MAX_GAP_MS)) {
      this.reset();
      this.lastReset = true;
    }

    // 3. Hampel spike filter
    this.buf.push(raw);
    if (this.buf.length > P.HAMPEL_SIZE) this.buf.shift();
    var n = this.buf.length;
    var s = this.buf.slice().sort(function (a, b) { return a - b; });
    var med = s[n >> 1];
    var dev = this.buf.map(function (v) { return Math.abs(v - med); }).sort(function (a, b) { return a - b; });
    var mad = dev[n >> 1];
    var limit = Math.max(P.HAMPEL_K_MAD * mad, P.HAMPEL_MIN_CM);
    var filtered = Math.abs(raw - med) > limit ? med : raw;
    this.lastMed = med;          // exposed for the dashboard only
    this.lastLimit = limit;

    // 4. Window features on the filtered signal
    this.win.push(filtered);
    if (this.win.length > P.WINDOW_SIZE) this.win.shift();
    var m = this.win.length, v = 0.0;
    if (m >= 2) {
      var sum = 0.0, i;
      for (i = 0; i < m; i++) sum += this.win[i];
      var mean = sum / m;
      for (i = 0; i < m; i++) v += (this.win[i] - mean) * (this.win[i] - mean);
      v /= m;
    }
    var roc = (this.lastTs !== null && ts > this.lastTs)
      ? Math.abs(filtered - this.lastF) / ((ts - this.lastTs) / 1000.0) : 0.0;

    this.lastF = filtered;
    this.lastTs = ts;
    return [filtered, Math.sqrt(v), roc, v];
  };

  // ── AlarmFilter: N anomalous in a row raise, M normal in a row clear ───────
  function AlarmFilter(P) {
    this.P = P;
    this.active = false;
    this.over = 0;
    this.under = 0;
  }
  AlarmFilter.prototype.update = function (anomalous) {
    if (anomalous) { this.over++; this.under = 0; } else { this.under++; this.over = 0; }
    if (!this.active && this.over >= this.P.ALARM_ON_COUNT) this.active = true;
    else if (this.active && this.under >= this.P.ALARM_OFF_COUNT) this.active = false;
    return this.active;
  };

  // ── Autoencoder 4-8-2-8-4 (ReLU, ReLU, ReLU, sigmoid), float32 like the ESP32 ──
  var f32 = Math.fround;
  // `trace` (optional array) receives the activations of every layer, input first.
  function forward(model, x, trace) {
    var h = x.map(f32), L = model.layers;
    if (trace) trace.push(h);
    for (var l = 0; l < L.length; l++) {
      var W = L[l].W, b = L[l].b, out = new Array(b.length);
      for (var j = 0; j < b.length; j++) {
        var z = f32(b[j]);
        for (var i = 0; i < h.length; i++) z = f32(z + f32(h[i] * W[i][j]));
        out[j] = l === L.length - 1 ? f32(1 / (1 + Math.exp(-z))) : (z > 0 ? z : 0);
      }
      h = out;
      if (trace) trace.push(h);
    }
    return h;
  }

  // ── Detector: the full main.cpp loop for one raw reading ───────────────────
  function Detector(model) {
    this.model = model;
    this.P = model.params;
    this.pre = new Preprocessor(this.P);
    this.alarm = new AlarmFilter(this.P);
  }
  Detector.prototype.step = function (raw, ts) {
    var M = this.model;
    var feats = this.pre.push(raw, ts);
    var fault = this.pre.sensorFault();
    var res = {
      ts: ts, raw: raw, valid: feats !== null, fault: fault,
      feats: null, input: null, recon: null, mse: NaN,
      anomalous: false, spike: false, alarm: this.alarm.active,
    };
    if (feats !== null) {
      var x = feats.map(function (v, i) { return f32((v - M.norm_min[i]) * M.norm_scale[i]); });
      var acts = [];
      var y = forward(M, x, acts);
      var se = 0;
      for (var i = 0; i < 4; i++) se += (x[i] - y[i]) * (x[i] - y[i]);
      res.acts = acts;
      res.median = this.pre.lastMed;
      res.limit = this.pre.lastLimit;
      res.gapReset = this.pre.lastReset;
      res.feats = feats;
      res.input = x;
      res.recon = y;
      res.mse = se / 4;
      res.anomalous = res.mse > M.threshold;
      res.spike = feats[0] !== raw;
      res.alarm = this.alarm.update(res.anomalous);
    }
    res.led = res.alarm || fault;
    res.invalidStreak = this.pre.invalidStreak;
    res.over = this.alarm.over;
    res.under = this.alarm.under;
    return res;
  };

  var api = { Preprocessor: Preprocessor, AlarmFilter: AlarmFilter, forward: forward, Detector: Detector };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Esp32Pipeline = api;
})(typeof window !== "undefined" ? window : this);
