// Live demo: feeds raw readings (simulated, replayed, uploaded or from a real
// ESP32 over Web Serial) through pipeline.js and renders every stage.
(function () {
  "use strict";

  var M = window.ESP32_MODEL, REPLAY = window.ESP32_REPLAY, Pipe = window.Esp32Pipeline;
  var P = M.params, DT = P.SAMPLE_INTERVAL_MS, WIN = 60000, GRACE = 3000;
  var SVGNS = "http://www.w3.org/2000/svg";
  var $ = function (id) { return document.getElementById(id); };

  function svg(tag, attrs, parent) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function html(tag, cls, text, parent) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function clear(e) { while (e.firstChild) e.removeChild(e.firstChild); }
  function fmt(v, d) { return (v === null || v === undefined || isNaN(v)) ? "–" : v.toFixed(d); }
  function fmtMse(v) {
    if (isNaN(v)) return "–";
    if (v >= 1e4 || v < 1e-3) return v.toExponential(2);
    return v < 1 ? v.toFixed(4) : v.toFixed(2);
  }
  var SUP = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
  function pow10(e) { return "10" + String(e).split("").map(function (c) { return SUP[c]; }).join(""); }
  window.Esp32Fmt = { pow10: pow10, svg: svg, html: html, clear: clear };

  // ── Simulator (mirror of firmware/src/sensor.cpp) ─────────────────────────
  var SCEN = {
    normal:      { name: "Normal (object at 30 cm)", base: 30, noise: 0.2, vib: 0, kind: "normal" },
    obstruction: { name: "Obstruction at 8 cm", base: 8, noise: 0.3, vib: 0, kind: "anomaly" },
    vibration:   { name: "Vibration ±3 cm", base: 30, noise: 0.2, vib: 3, kind: "anomaly" },
    removed:     { name: "Object removed (wall at 150 cm)", base: 150, noise: 0.6, vib: 0, kind: "anomaly" },
    blocked:     { name: "Sensor blocked (no echo)", base: 0, noise: 0, vib: 0, noEcho: true, kind: "fault" },
  };
  var AUTO_SEQ = ["normal", "obstruction", "normal", "vibration", "normal", "removed", "normal", "blocked"];
  var SCEN_BUTTONS = [["auto", "Auto cycle (like the firmware)"], ["normal", "Normal"], ["obstruction", "Obstruction"],
                      ["vibration", "Vibration"], ["removed", "Object removed"], ["blocked", "Sensor blocked"]];
  function gauss() { return (Math.random() + Math.random() + Math.random() + Math.random() - 2) * 1.7320508; }

  // ── State ─────────────────────────────────────────────────────────────────
  var S = {
    mode: "sim", speed: 1, paused: false, finished: false,
    det: null, buf: [], last: null, t0: null, vt: 0, lastFrame: null, dirty: true, silent: false,
    stats: null, prevValidTs: null,
    sim: { scen: "auto", nextTs: 0, ghost: false, ghostPending: false, lastName: null },
    replay: { data: null, i: 0, events: null },
    serial: { port: null, reader: null, closed: null, warned: false },
    hoverPx: null,
  };

  function resetRun() {
    S.det = new Pipe.Detector(M);
    S.buf = []; S.last = null; S.t0 = null; S.vt = 0; S.finished = false; S.prevValidTs = null;
    S.stats = { readings: 0, invalid: 0, spikes: 0, anomalous: 0, alarms: 0, faults: 0 };
    S.sim.nextTs = 0; S.sim.ghost = false; S.sim.ghostPending = false; S.sim.lastName = null;
    S.replay.i = 0;
    clear($("log"));
    S.dirty = true;
  }

  // ── Ingest one raw reading ────────────────────────────────────────────────
  function ingest(raw, ts, truth, scenName) {
    if (S.t0 === null) S.t0 = ts;
    if (S.last && ts < S.last.ts) { S.buf = []; log("info", "↺", "Timestamp went backwards (device reboot?) - chart cleared"); }
    var prev = S.last;
    var prevValidTs = S.prevValidTs;
    var r = S.det.step(raw, ts);
    r.truth = truth; r.scen = scenName;
    r.dtValid = (r.valid && prevValidTs !== null) ? ts - prevValidTs : null;
    if (r.valid) S.prevValidTs = ts;

    var st = S.stats;
    st.readings++;
    if (!r.valid) st.invalid++;
    if (r.spike) st.spikes++;
    if (r.anomalous) st.anomalous++;
    var wasAlarm = prev ? prev.alarm : false, wasFault = prev ? prev.fault : false;
    if (r.alarm && !wasAlarm) st.alarms++;
    if (r.fault && !wasFault) st.faults++;

    if (!S.silent) {
      if (scenName && scenName !== (prev && prev.scen)) log("info", "▸", "Scenario: " + scenName, ts);
      if (r.alarm && !wasAlarm) log("alarm", "!", "ALARM raised: " + S.det.P.ALARM_ON_COUNT + " anomalous samples in a row (MSE " + fmtMse(r.mse) + ")", ts);
      if (!r.alarm && wasAlarm) log("ok", "✓", "Alarm cleared: " + S.det.P.ALARM_OFF_COUNT + " normal samples in a row", ts);
      if (r.fault && !wasFault) log("fault", "⚠", "SENSOR FAULT: " + P.FAULT_INVALID_COUNT + " readings in a row without a valid echo", ts);
      if (!r.fault && wasFault) log("ok", "✓", "Valid echoes restored", ts);
      if (r.gapReset) log("info", "↺", "Gap > " + P.MAX_GAP_MS + " ms without a valid reading: filter history reset", ts);
      if (S.sim.ghostPending && S.mode === "sim") {
        S.sim.ghostPending = false;
        if (!r.valid) log("spike", "×", "Ghost echo " + fmt(raw, 1) + " cm: out of range, dropped by the validity gate", ts);
        else if (r.spike) log("spike", "×", "Ghost echo " + fmt(raw, 1) + " cm → Hampel replaced it with " + fmt(r.feats[0], 2) + " cm. The model never saw it", ts);
        else log("spike", "×", "Ghost echo " + fmt(raw, 1) + " cm was within the Hampel limit and passed through", ts);
      }
    }

    S.buf.push(r);
    var cut = ts - WIN - DT;
    while (S.buf.length && S.buf[0].ts < cut) S.buf.shift();
    S.last = r;
    S.dirty = true;
  }

  function simSample(ts) {
    var id = S.sim.scen === "auto" ? AUTO_SEQ[Math.floor(ts / 10000) % AUTO_SEQ.length] : S.sim.scen;
    var sc = SCEN[id], d;
    if (sc.noEcho) d = -1;
    else if (S.sim.ghost) {
      S.sim.ghost = false; S.sim.ghostPending = true;
      d = Math.min(sc.base * 2 + 3 * gauss(), 420);        // multipath double bounce
    } else {
      var u = Math.random();
      if (u < 0.015) d = -1;                                // dropped echo (1.5 %)
      else if (u < 0.025) d = 2 + Math.random() * 398;      // ghost / junk (1 %)
      else {
        d = sc.base + sc.noise * gauss();
        if (sc.vib) d += sc.vib * Math.sin(2 * Math.PI * 1.7 * ts / 1000);
      }
    }
    ingest(Math.round(d * 100) / 100, ts, sc.kind, sc.name);
  }

  function replayTruth(d, i) {
    if (!d.label) return [null, null];
    var name = d.scenarios[d.scen[i]];
    var kind = d.label[i] ? (name === "sensor_fault" ? "fault" : "anomaly") : "normal";
    return [kind, name.replace(/_/g, " ")];
  }

  function replayRow(i) {
    var d = S.replay.data, t = replayTruth(d, i);
    ingest(d.raw[i], d.ts[i], t[0], t[1]);
  }

  // ── Frame loop ────────────────────────────────────────────────────────────
  function frame(now) {
    if (S.lastFrame === null) S.lastFrame = now;
    var dt = Math.min(now - S.lastFrame, 250);
    S.lastFrame = now;
    if (!S.paused && !S.finished && (S.mode === "sim" || S.mode === "replay")) {
      S.vt += dt * S.speed;
      var n = 0;
      if (S.mode === "sim") {
        while (S.sim.nextTs <= S.vt && n++ < 1000) { simSample(S.sim.nextTs); S.sim.nextTs += DT; }
      } else if (S.replay.data) {
        var d = S.replay.data;
        while (S.replay.i < d.ts.length && d.ts[S.replay.i] - d.ts[0] <= S.vt && n++ < 1000) replayRow(S.replay.i++);
        if (S.replay.i >= d.ts.length) {
          S.finished = true;
          log("info", "■", "Replay finished. Press Restart to run it again", S.last ? S.last.ts : 0);
          updateButtons();
        }
      }
    }
    if (S.dirty) { render(); S.dirty = false; }
    requestAnimationFrame(frame);
  }

  // ── Log ───────────────────────────────────────────────────────────────────
  function tRel(ts) { return S.t0 === null ? "" : ((ts - S.t0) / 1000).toFixed(1) + " s"; }
  function log(kind, icon, text, ts) {
    var ul = $("log");
    var li = html("li", kind);
    html("span", "t", ts === undefined ? "" : tRel(ts), li);
    html("span", "ic", icon, li);
    html("span", "", text, li);
    ul.insertBefore(li, ul.firstChild);
    while (ul.children.length > 60) ul.removeChild(ul.lastChild);
  }

  // ── Static UI pieces ──────────────────────────────────────────────────────
  var TILES = [["readings", "Readings"], ["invalid", "Invalid (dropped)"], ["spikes", "Spikes removed"],
               ["anomalous", "Anomalous samples"], ["alarms", "Alarms raised"], ["faults", "Sensor faults"]];
  var tileEls = {};
  TILES.forEach(function (t) {
    var d = html("div", "tile", null, $("tiles"));
    tileEls[t[0]] = html("div", "tile-val", "0", d);
    html("div", "tile-label", t[1], d);
  });

  var STAGES = [
    ["gate", "① Validity gate", "preprocess.cpp"],
    ["gap", "② Gap reset", "preprocess.cpp"],
    ["hampel", "③ Hampel spike filter", "preprocess.cpp"],
    ["feat", "④ Window features", "preprocess.cpp"],
    ["model", "⑤ Autoencoder", "inference.cpp"],
    ["debounce", "⑥ Alarm debounce", "inference.cpp"],
    ["led", "⑦ LED", "main.cpp"],
  ];
  var stageEls = {};
  STAGES.forEach(function (s) {
    var li = html("li", "stage", null, $("stages"));
    html("div", "stage-title", s[1], li);
    var v = html("div", "stage-val", "–", li);
    var n = html("div", "stage-note", "", li);
    html("div", "stage-file", s[2], li);
    stageEls[s[0]] = { li: li, v: v, n: n };
  });
  function setStage(id, cls, val, note) {
    var e = stageEls[id];
    e.li.className = "stage " + (cls || "");
    e.v.textContent = val;
    e.n.textContent = note;
  }

  var scenSeg = $("seg-scen");
  SCEN_BUTTONS.forEach(function (b) {
    var btn = html("button", "", b[1], scenSeg);
    btn.type = "button";
    btn.dataset.scen = b[0];
    btn.setAttribute("aria-pressed", b[0] === "auto" ? "true" : "false");
    btn.addEventListener("click", function () {
      S.sim.scen = b[0];
      pressOne(scenSeg, btn);
      if (S.mode !== "sim") setMode("sim");
    });
  });
  function pressOne(group, btn) {
    Array.prototype.forEach.call(group.querySelectorAll("button"), function (b) {
      b.setAttribute("aria-pressed", b === btn ? "true" : "false");
    });
  }

  // ── Autoencoder panel ─────────────────────────────────────────────────────
  var FEAT = [["distance", "cm", 2], ["jitter", "cm", 3], ["rate of change", "cm/s", 2], ["variance", "cm²", 3]];
  var net = { nodes: [] };
  (function buildNet() {
    var root = $("net"), sizes = [4, 8, 2, 8, 4], W = 360, H = 190, top = 22, bottom = 14;
    var xs = sizes.map(function (_, i) { return 46 + i * (W - 92) / (sizes.length - 1); });
    var pos = sizes.map(function (n, c) {
      var out = [];
      for (var k = 0; k < n; k++) out.push([xs[c], top + (k + 0.5) * (H - top - bottom) / n]);
      return out;
    });
    var gE = svg("g", {}, root), gN = svg("g", {}, root);
    for (var c = 0; c < sizes.length - 1; c++)
      pos[c].forEach(function (a) { pos[c + 1].forEach(function (b) {
        svg("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: "net-edge" }, gE);
      }); });
    var titles = ["input", "enc 8", "latent 2", "dec 8", "output"];
    pos.forEach(function (col, c) {
      var t = svg("text", { x: xs[c], y: 12, "text-anchor": "middle", class: "net-label" }, root);
      t.textContent = titles[c];
      net.nodes.push(col.map(function (p) {
        return svg("circle", { cx: p[0], cy: p[1], r: c === 2 ? 8 : 6.5, class: "net-node" + (c === 4 ? " out" : ""), "fill-opacity": 0.1 }, gN);
      }));
    });
    ["dist", "jitter", "roc", "var"].forEach(function (n, k) {
      var a = svg("text", { x: xs[0] - 12, y: pos[0][k][1] + 3, "text-anchor": "end", class: "net-label" }, root);
      a.textContent = n;
      var b = svg("text", { x: xs[4] + 12, y: pos[4][k][1] + 3, class: "net-label" }, root);
      b.textContent = n;
    });
  })();
  var aeRows = [];
  (function buildAeTable() {
    var tb = $("ae-table").querySelector("tbody");
    FEAT.forEach(function (f) {
      var tr = html("tr", "", null, tb);
      html("td", "", f[0], tr);
      var val = html("td", "", "–", tr);
      var tdb = html("td", "", null, tr);
      var bars = html("div", "ae-bars", null, tdb);
      html("div", "ae-track", null, bars);
      var z = html("div", "ae-zero", null, bars), o = html("div", "ae-one", null, bars);
      z.style.left = pct(0); o.style.left = pct(1);
      var bi = html("div", "ae-bar in", null, bars), bo = html("div", "ae-bar out", null, bars);
      var ovL = html("span", "ae-over", "", bars), ovR = html("span", "ae-over", "", bars);
      ovL.style.left = "-10px"; ovR.style.right = "-10px";
      var nums = html("div", "ae-nums", "", tdb);
      var err = html("td", "", "–", tr);
      aeRows.push({ val: val, bi: bi, bo: bo, ovL: ovL, ovR: ovR, nums: nums, err: err, unit: f[1], d: f[2] });
    });
  })();
  // normalized domain shown in the bars: [-0.25, 1.25]
  function frac(v) { return (v + 0.25) / 1.5; }
  function pct(v) { return (Math.max(0, Math.min(1, frac(v))) * 100).toFixed(2) + "%"; }
  function placeBar(el, v) {
    var a = Math.max(0, Math.min(1, frac(0))), b = Math.max(0, Math.min(1, frac(v)));
    el.style.left = (Math.min(a, b) * 100) + "%";
    el.style.width = Math.max(Math.abs(b - a) * 100, 0.8) + "%";
  }

  function renderAe(r) {
    var acts = r && r.valid ? r.acts : null;
    net.nodes.forEach(function (col, c) {
      col.forEach(function (node, k) {
        var o = 0.08;
        if (acts) {
          var v = acts[c][k];
          o = c === 4 ? 0.1 + 0.9 * v : 0.1 + 0.9 * (1 - Math.exp(-Math.abs(v) * 1.5));
        }
        node.setAttribute("fill-opacity", o.toFixed(3));
      });
    });
    aeRows.forEach(function (row, i) {
      if (!acts) {
        row.val.textContent = "–"; row.err.textContent = "–"; row.nums.textContent = r && !r.valid ? "skipped: invalid reading" : "";
        row.bi.style.width = "0"; row.bo.style.width = "0"; row.ovL.textContent = ""; row.ovR.textContent = "";
        return;
      }
      var x = r.input[i], y = r.recon[i];
      row.val.textContent = r.feats[i].toFixed(row.d) + " " + row.unit;
      placeBar(row.bi, x); placeBar(row.bo, y);
      row.ovL.textContent = x < -0.25 ? "‹" : "";
      row.ovR.textContent = x > 1.25 ? "›" : "";
      row.nums.textContent = "in " + fmtSmart(x) + " · out " + y.toFixed(3);
      row.err.textContent = fmtMse((x - y) * (x - y));
    });
  }
  function fmtSmart(v) { return Math.abs(v) >= 100 ? v.toExponential(1) : v.toFixed(3); }

  // ── Charts ────────────────────────────────────────────────────────────────
  var MARGIN = { l: 48, r: 14, t: 16, b: 24 };
  function Chart(el, height) {
    this.el = el; this.H = height; this.W = 600;
    this.gBand = svg("g", {}, el); this.gGrid = svg("g", {}, el); this.gData = svg("g", {}, el);
    this.cross = svg("line", { class: "crosshair", display: "none" }, el);
    this.gTop = svg("g", {}, el);
  }
  Chart.prototype.resize = function () {
    this.W = Math.max(280, Math.floor(this.el.parentNode.clientWidth));
    this.el.setAttribute("width", this.W);
    this.el.setAttribute("height", this.H);
    this.el.setAttribute("viewBox", "0 0 " + this.W + " " + this.H);
  };
  var chartD = new Chart($("chart-dist"), 240), chartM = new Chart($("chart-mse"), 190);
  function resizeCharts() { chartD.resize(); chartM.resize(); S.dirty = true; }
  window.addEventListener("resize", resizeCharts);
  resizeCharts();

  function niceStep(span, n) {
    var raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
    return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
  }
  function circlePath(x, y, r) {
    return "M" + (x - r).toFixed(1) + " " + y.toFixed(1) + "a" + r + " " + r + " 0 1 0 " + 2 * r + " 0a" + r + " " + r + " 0 1 0 " + -2 * r + " 0";
  }
  function runs(buf, key) {
    var out = [], start = null;
    for (var i = 0; i < buf.length; i++) {
      var on = typeof key === "function" ? key(buf[i]) : buf[i][key];
      if (on && start === null) start = buf[i].ts;
      if (!on && start !== null) { out.push([start, buf[i].ts]); start = null; }
    }
    if (start !== null) out.push([start, buf[buf.length - 1].ts + DT]);
    return out;
  }
  function drawFrame(c, tEnd, yTicks, yFn, yLabelFn) {
    clear(c.gBand); clear(c.gGrid); clear(c.gData); clear(c.gTop);
    var W = c.W, H = c.H, pw = W - MARGIN.l - MARGIN.r, t0 = tEnd - WIN;
    var x = function (ts) { return MARGIN.l + (ts - t0) / WIN * pw; };
    yTicks.forEach(function (v) {
      var y = yFn(v);
      svg("line", { x1: MARGIN.l, x2: W - MARGIN.r, y1: y, y2: y, class: "grid-line" }, c.gGrid);
      var t = svg("text", { x: MARGIN.l - 6, y: y + 3.5, "text-anchor": "end", class: "tick-label" }, c.gGrid);
      t.textContent = yLabelFn(v);
    });
    svg("line", { x1: MARGIN.l, x2: W - MARGIN.r, y1: H - MARGIN.b, y2: H - MARGIN.b, class: "axis-line" }, c.gGrid);
    for (var s = 0; s <= 60; s += 10) {
      var tx = x(tEnd - s * 1000);
      var t = svg("text", { x: tx, y: H - 6, "text-anchor": s === 0 ? "end" : "middle", class: "tick-label" }, c.gGrid);
      t.textContent = s === 0 ? "now" : "−" + s + " s";
    }
    // state bands (alarm / fault) + ground truth strip
    var y0 = MARGIN.t, y1 = H - MARGIN.b;
    var clampX = function (v) { return Math.max(MARGIN.l, Math.min(W - MARGIN.r, v)); };
    runs(S.buf, "alarm").forEach(function (b) {
      svg("rect", { x: clampX(x(b[0])), y: y0, width: Math.max(1, clampX(x(b[1])) - clampX(x(b[0]))), height: y1 - y0, class: "band-alarm" }, c.gBand);
    });
    runs(S.buf, "fault").forEach(function (b) {
      svg("rect", { x: clampX(x(b[0])), y: y0, width: Math.max(1, clampX(x(b[1])) - clampX(x(b[0]))), height: y1 - y0, class: "band-fault" }, c.gBand);
    });
    runs(S.buf, function (r) { return r.truth && r.truth !== "normal"; }).forEach(function (b) {
      svg("rect", { x: clampX(x(b[0])), y: 4, width: Math.max(1, clampX(x(b[1])) - clampX(x(b[0]))), height: 4, rx: 2, class: "band-truth" }, c.gBand);
    });
    return x;
  }

  function renderDistance(tEnd) {
    var c = chartD, H = c.H, lo = Infinity, hi = -Infinity;
    S.buf.forEach(function (r) { if (r.valid) { lo = Math.min(lo, r.feats[0]); hi = Math.max(hi, r.feats[0]); } });
    if (lo === Infinity) { lo = 0; hi = 50; }
    var span = Math.max(hi - lo, 6), mid = (hi + lo) / 2;
    lo = mid - span * 0.6; hi = mid + span * 0.6;
    if (lo < 0) { hi -= lo; lo = 0; }
    var step = niceStep(hi - lo, 4);
    lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
    var ph = H - MARGIN.t - MARGIN.b;
    var y = function (v) { return MARGIN.t + (1 - (v - lo) / (hi - lo)) * ph; };
    var yc = function (v) { return Math.max(MARGIN.t, Math.min(H - MARGIN.b, y(v))); };
    var ticks = [];
    for (var v = lo; v <= hi + 1e-9; v += step) ticks.push(v);
    var x = drawFrame(c, tEnd, ticks, y, function (v) { return (step < 1 ? v.toFixed(1) : v.toFixed(0)) + " cm"; });

    var dots = "", rings = "", rug = "", line = "", prevTs = null;
    S.buf.forEach(function (r) {
      var px = x(r.ts);
      if (px < MARGIN.l - 1) return;
      if (!r.valid) { rug += "M" + px.toFixed(1) + " " + (H - MARGIN.b) + "v-9"; return; }
      if (r.spike) rings += circlePath(px, yc(r.raw), 4);
      else dots += circlePath(px, yc(r.raw), 1.8);
      var py = y(r.feats[0]).toFixed(1);
      line += (prevTs === null || r.ts - prevTs > P.MAX_GAP_MS ? "M" : "L") + px.toFixed(1) + " " + py;
      prevTs = r.ts;
    });
    svg("path", { d: rug, class: "rug" }, c.gData);
    svg("path", { d: dots, class: "raw-dots" }, c.gData);
    svg("path", { d: line, class: "line-s1" }, c.gData);
    svg("path", { d: rings, class: "spike-rings" }, c.gData);
    c.x = x;
  }

  var LOG_LO = -4, LOG_HI = 9;
  function renderMse(tEnd) {
    var c = chartM, H = c.H, ph = H - MARGIN.t - MARGIN.b;
    var y = function (lv) { return MARGIN.t + (1 - (lv - LOG_LO) / (LOG_HI - LOG_LO)) * ph; };
    var ly = function (v) { return y(Math.max(LOG_LO, Math.min(LOG_HI, Math.log10(Math.max(v, 1e-12))))); };
    var x = drawFrame(c, tEnd, [-4, -2, 0, 2, 4, 6, 8], y, function (v) { return v === 0 ? "1" : pow10(v); });

    var line = "", over = "", prevTs = null;
    S.buf.forEach(function (r) {
      var px = x(r.ts);
      if (px < MARGIN.l - 1 || !r.valid) return;
      var py = ly(r.mse);
      line += (prevTs === null || r.ts - prevTs > P.MAX_GAP_MS ? "M" : "L") + px.toFixed(1) + " " + py.toFixed(1);
      if (r.anomalous) over += circlePath(px, py, 3);
      prevTs = r.ts;
    });
    svg("path", { d: line, class: "line-s1" }, c.gData);
    svg("path", { d: over, class: "over-dots" }, c.gData);
    var ty = ly(M.threshold);
    svg("line", { x1: MARGIN.l, x2: c.W - MARGIN.r, y1: ty, y2: ty, class: "thr-line" }, c.gTop);
    var t = svg("text", { x: MARGIN.l + 6, y: ty - 5, class: "thr-label" }, c.gTop);
    t.textContent = "threshold " + M.threshold.toFixed(3) + " (99.9th pct of normal)";
    c.x = x;
  }

  // ── Hover: shared crosshair + tooltip ─────────────────────────────────────
  function nearest(ts) {
    var b = S.buf, lo = 0, hi = b.length - 1;
    if (!b.length) return null;
    while (hi - lo > 1) { var m = (lo + hi) >> 1; if (b[m].ts < ts) lo = m; else hi = m; }
    return Math.abs(b[lo].ts - ts) <= Math.abs(b[hi].ts - ts) ? b[lo] : b[hi];
  }
  function renderHover() {
    var tip = $("tooltip");
    if (S.hoverPx === null || !S.last || !chartD.x) {
      tip.hidden = true; chartD.cross.setAttribute("display", "none"); chartM.cross.setAttribute("display", "none");
      return;
    }
    var pw = chartD.W - MARGIN.l - MARGIN.r;
    var ts = S.last.ts - WIN + (S.hoverPx - MARGIN.l) / pw * WIN;
    var r = nearest(ts);
    if (!r) return;
    var px = chartD.x(r.ts);
    [chartD, chartM].forEach(function (c) {
      c.cross.setAttribute("display", "inline");
      c.cross.setAttribute("x1", px); c.cross.setAttribute("x2", px);
      c.cross.setAttribute("y1", MARGIN.t); c.cross.setAttribute("y2", c.H - MARGIN.b);
    });
    clear(tip);
    html("div", "tt-head", "t = " + tRel(r.ts) + (r.scen ? " · " + r.scen : ""), tip);
    function row(val, label, key) {
      var d = html("div", "tt-row", null, tip);
      if (key) html("span", "key " + key, null, d);
      html("b", "", val, d);
      html("span", "", label, d);
    }
    row(r.raw < 0 ? "no echo (−1)" : fmt(r.raw, 2) + " cm", r.valid ? "raw" : "raw · invalid, dropped", "key-dot");
    if (r.valid) {
      row(fmt(r.feats[0], 2) + " cm", r.spike ? "filtered · spike replaced" : "filtered", "key-line s1");
      row(fmtMse(r.mse), "MSE · " + (r.anomalous ? "above" : "below") + " threshold", r.anomalous ? "key-dot crit" : "key-line s1");
    }
    row(r.fault ? "SENSOR FAULT" : r.alarm ? "ALARM" : "normal", "device state" + (r.led ? " · LED on" : ""));
    tip.hidden = false;
    var stack = $("chart-stack"), w = tip.offsetWidth;
    var left = px + 14 + w > stack.clientWidth ? px - 14 - w : px + 14;
    tip.style.left = Math.max(0, left) + "px";
    tip.style.top = "30px";
  }
  [chartD, chartM].forEach(function (c) {
    c.el.addEventListener("pointermove", function (e) {
      var rect = c.el.getBoundingClientRect(), px = e.clientX - rect.left;
      S.hoverPx = (px >= MARGIN.l && px <= c.W - MARGIN.r) ? px : null;
      S.dirty = true;
    });
    c.el.addEventListener("pointerleave", function () { S.hoverPx = null; S.dirty = true; });
  });

  // ── Status + stages ───────────────────────────────────────────────────────
  var MODE_LABEL = { sim: "Live simulator", replay: "Replay", serial: "ESP32 over USB" };
  function renderStatus(r) {
    var st = $("state"), led = $("led");
    if (!r) {
      st.className = "state"; $("state-icon").textContent = "·"; $("state-text").textContent = "Waiting for data";
      $("state-why").textContent = ""; led.className = "led";
    } else if (r.fault) {
      st.className = "state fault"; $("state-icon").textContent = "⚠"; $("state-text").textContent = "SENSOR FAULT";
      $("state-why").textContent = r.invalidStreak + " readings in a row without a valid echo (limit " + P.FAULT_INVALID_COUNT + ")";
      led.className = "led on-fault";
    } else if (r.alarm) {
      st.className = "state alarm"; $("state-icon").textContent = "!"; $("state-text").textContent = "ANOMALY ALARM";
      $("state-why").textContent = "Reconstruction error above threshold for " + P.ALARM_ON_COUNT + "+ samples in a row";
      led.className = "led on-alarm";
    } else {
      st.className = "state normal"; $("state-icon").textContent = "✓"; $("state-text").textContent = "NORMAL";
      $("state-why").textContent = "The autoencoder reconstructs the signal well (error below threshold)";
      led.className = "led";
    }
    var src = MODE_LABEL[S.mode] || "";
    if (S.mode === "replay" && S.replay.data) src += ": " + S.replay.data.name;
    if (S.mode !== "serial") src += " · " + S.speed + "×" + (S.paused ? " · paused" : "");
    if (r && r.truth) src += " · truth: " + r.scen;
    $("state-src").textContent = src;

    var st2 = S.stats;
    tileEls.readings.textContent = st2.readings;
    tileEls.invalid.textContent = st2.invalid + (st2.readings ? " (" + (100 * st2.invalid / st2.readings).toFixed(1) + "%)" : "");
    tileEls.spikes.textContent = st2.spikes;
    tileEls.anomalous.textContent = st2.anomalous;
    tileEls.alarms.textContent = st2.alarms;
    tileEls.faults.textContent = st2.faults;
  }

  function renderStages(r) {
    if (!r) { STAGES.forEach(function (s) { setStage(s[0], "", "–", ""); }); return; }
    var lim = P.VALID_MIN_CM + "–" + P.VALID_MAX_CM + " cm";
    if (!r.valid) {
      setStage("gate", "bad", r.raw < 0 ? "no echo (−1)" : fmt(r.raw, 2) + " cm",
        "outside " + lim + ": dropped · invalid streak " + r.invalidStreak + "/" + P.FAULT_INVALID_COUNT);
      ["gap", "hampel", "feat", "model"].forEach(function (id) { setStage(id, "skip", "skipped", "invalid readings never reach this stage"); });
      setStage("debounce", "skip", r.alarm ? "ALARM held" : "unchanged", "invalid readings don't count either way");
    } else {
      setStage("gate", "ok", fmt(r.raw, 2) + " cm", "inside " + lim + ": kept");
      setStage("gap", r.gapReset ? "warn" : "ok", r.gapReset ? "history reset" : (r.dtValid === null ? "first reading" : "Δt " + r.dtValid + " ms"),
        "reset if > " + P.MAX_GAP_MS + " ms since last valid reading");
      setStage("hampel", r.spike ? "warn" : "ok", r.spike ? fmt(r.raw, 1) + " → " + fmt(r.feats[0], 2) + " cm" : "kept " + fmt(r.feats[0], 2) + " cm",
        "median " + fmt(r.median, 2) + " · limit ±" + fmt(r.limit, 2) + " cm (" + P.HAMPEL_SIZE + " samples)");
      setStage("feat", "ok", "σ " + fmt(r.feats[1], 3) + " cm",
        "|Δ| " + fmt(r.feats[2], 2) + " cm/s · var " + fmt(r.feats[3], 3) + " over " + P.WINDOW_SIZE + " samples");
      setStage("model", r.anomalous ? "bad" : "ok", "MSE " + fmtMse(r.mse),
        (r.anomalous ? "above" : "below") + " threshold " + M.threshold.toFixed(3) + " (" + fmtRatio(r.mse / M.threshold) + ")");
      setStage("debounce", r.alarm ? "bad" : "ok", r.alarm ? "ALARM" : "no alarm",
        r.alarm ? "normal streak " + r.under + "/" + P.ALARM_OFF_COUNT + " to clear"
                : "anomalous streak " + r.over + "/" + P.ALARM_ON_COUNT + " to raise");
    }
    setStage("led", r.led ? (r.fault ? "warn" : "bad") : "ok", r.led ? "ON" : "off",
      r.fault ? "sensor fault" : r.alarm ? "anomaly alarm" : "alarm OR fault turns it on");
  }
  function fmtRatio(q) { return q >= 100 ? q.toExponential(1) + "×" : q.toFixed(q < 1 ? 2 : 1) + "×"; }

  function renderReplayBar() {
    if (S.mode !== "replay" || !S.replay.data) return;
    var d = S.replay.data, n = d.ts.length;
    $("replay-bar").style.width = (100 * S.replay.i / n).toFixed(2) + "%";
    var cur = S.replay.i ? (d.ts[S.replay.i - 1] - d.ts[0]) / 1000 : 0, tot = (d.ts[n - 1] - d.ts[0]) / 1000;
    $("replay-pos").textContent = fmtClock(cur) + " / " + fmtClock(tot);
  }
  function fmtClock(s) { var m = Math.floor(s / 60); return m + ":" + String(Math.floor(s % 60)).padStart(2, "0"); }

  function render() {
    var tEnd = S.last ? S.last.ts : WIN;
    renderDistance(tEnd);
    renderMse(tEnd);
    renderHover();
    renderStatus(S.last);
    renderStages(S.last);
    renderAe(S.last);
    renderReplayBar();
  }

  // ── Replay evaluation (mirror of pipeline.evaluate_events) ─────────────────
  function evaluate(d) {
    var det = new Pipe.Detector(M), n = d.ts.length, led = new Array(n), i, k;
    for (i = 0; i < n; i++) led[i] = det.step(d.raw[i], d.ts[i]).led;
    var events = [], inEvent = new Array(n).fill(false), start = null;
    for (i = 0; i <= n; i++) {
      var on = i < n && d.label[i] === 1;
      if (on && start === null) start = i;
      if (!on && start !== null) { events.push([start, i - 1]); start = null; }
    }
    var res = events.map(function (e) {
      var a = d.ts[e[0]], b = d.ts[e[1]] + GRACE, hit = -1;
      for (k = 0; k < n; k++) if (d.ts[k] >= a && d.ts[k] <= b) { inEvent[k] = true; if (hit < 0 && led[k]) hit = k; }
      return { i0: e[0], scenario: d.scenarios[d.scen[e[0]]], detected: hit >= 0, latency: hit >= 0 ? d.ts[hit] - a : null };
    });
    var fa = 0, normal = 0;
    for (i = 0; i < n; i++) {
      if (led[i] && !(i > 0 && led[i - 1]) && !inEvent[i]) fa++;
      if (!inEvent[i]) normal++;
    }
    return { events: res, falseAlarms: fa, normalMin: normal * DT / 60000 };
  }

  function renderEventsCard() {
    var card = $("events-card"), d = S.replay.data;
    if (S.mode !== "replay" || !d || !d.label) { card.hidden = true; return; }
    card.hidden = false;
    var ev = S.replay.events || (S.replay.events = evaluate(d));
    var det = ev.events.filter(function (e) { return e.detected; }).length;
    $("events-sub").textContent = "· " + det + "/" + ev.events.length + " detected · " + ev.falseAlarms +
      " false alarms in " + ev.normalMin.toFixed(1) + " min of normal data";
    var tb = $("events-table").querySelector("tbody");
    clear(tb);
    ev.events.forEach(function (e, idx) {
      var tr = html("tr", "", null, tb);
      html("td", "", String(idx + 1), tr);
      html("td", "", e.scenario.replace(/_/g, " "), tr);
      html("td", "", fmtClock((d.ts[e.i0] - d.ts[0]) / 1000), tr);
      html("td", e.detected ? "yes" : "no", e.detected ? "✓ yes" : "✗ missed", tr);
      html("td", "", e.latency === null ? "–" : (e.latency / 1000).toFixed(1) + " s", tr);
      var td = html("td", "", null, tr);
      var b = html("button", "btn", "Watch", td);
      b.type = "button";
      b.addEventListener("click", function () { seek(Math.max(0, e.i0 - Math.round(5000 / DT))); });
    });
  }

  function seek(idx) {
    var d = S.replay.data;
    resetRun();
    // Run the detector over everything before the jump point so its state is exact
    // (the chart buffer only keeps the last minute anyway).
    S.silent = true;
    for (var i = 0; i < idx; i++) replayRow(i);
    S.silent = false;
    S.t0 = d.ts[0];
    S.replay.i = idx;
    S.vt = d.ts[idx] - d.ts[0];
    S.paused = false; S.finished = false;
    setSpeed(1);
    log("info", "▸", "Jumped to " + fmtClock(S.vt / 1000) + " (5 s before the event)", d.ts[idx]);
    updateButtons();
    $("live").scrollIntoView({ behavior: "smooth" });
  }

  // ── CSV upload (same rules as pipeline.load_capture) ──────────────────────
  function parseCsv(text, name) {
    var ts = [], raw = [], label = [], scen = [], names = [], prev = null;
    text.split(/\r?\n/).forEach(function (line) {
      var row = line.split(",");
      if (!line.trim() || row[0].trim().charAt(0) === "#" || row[0].trim() === "timestamp_ms") return;
      if (!/^\s*-?\d+\s*$/.test(row[0])) return;
      var t = parseInt(row[0], 10), d = parseFloat(row[1]);
      if (isNaN(d)) return;
      if (prev && prev[0] === t && prev[1] === d) return;
      prev = [t, d];
      ts.push(t); raw.push(d);
      if (row.length >= 3) {
        label.push(parseInt(row[2], 10) ? 1 : 0);
        var s = (row[3] || "").trim();
        var k = names.indexOf(s); if (k < 0) { names.push(s); k = names.length - 1; }
        scen.push(k);
      }
    });
    var labelled = label.length === ts.length && ts.length > 0;
    return { name: name, ts: ts, raw: raw, label: labelled ? label : null, scen: labelled ? scen : null, scenarios: names };
  }
  $("file-input").addEventListener("change", function (e) {
    var f = e.target.files[0];
    if (!f) return;
    f.text().then(function (text) {
      var d = parseCsv(text, f.name);
      if (!d.ts.length) { log("fault", "⚠", "No data rows found in " + f.name + " (expected timestamp_ms,distance_cm[,label,scenario])"); return; }
      S.replay.data = d; S.replay.events = null;
      setMode("replay", true);
      log("info", "▸", "Loaded " + f.name + ": " + d.ts.length + " readings" + (d.label ? ", labelled" : ""), d.ts[0]);
    });
    e.target.value = "";
  });

  // ── Web Serial (real ESP32) ───────────────────────────────────────────────
  var hasSerial = "serial" in navigator;
  if (!hasSerial) {
    $("btn-serial").disabled = true;
    $("btn-serial").title = "Web Serial needs Chrome or Edge on desktop";
    $("btn-serial").textContent = "Connect ESP32 (Chrome/Edge only)";
  }
  function connectSerial() {
    var port;
    navigator.serial.requestPort().then(function (p) {
      port = p;
      return port.open({ baudRate: 115200 });
    }).then(function () {
      setMode("serial", true);
      S.serial.port = port;
      $("serial-status").textContent = "Connected · waiting for data…";
      var dec = new TextDecoderStream();
      S.serial.closed = port.readable.pipeTo(dec.writable).catch(function () {});
      var reader = dec.readable.getReader();
      S.serial.reader = reader;
      var pending = "";
      function pump() {
        return reader.read().then(function (res) {
          if (res.done) return;
          pending += res.value;
          var lines = pending.split(/\r?\n/);
          pending = lines.pop();
          lines.forEach(serialLine);
          return pump();
        });
      }
      return pump();
    }).catch(function (err) {
      if (err && err.name !== "NotFoundError") log("fault", "⚠", "Serial: " + err.message);
    }).then(function () {
      if (S.serial.port === port) disconnectSerial();
    });
  }
  function serialLine(line) {
    line = line.trim();
    if (!line) return;
    if (line.charAt(0) === "#" || line.charAt(0) === "[") { log("info", "›", "ESP32: " + line.replace(/^#\s*/, ""), S.last ? S.last.ts : undefined); return; }
    var c = line.split(",");
    if (!/^\d+$/.test(c[0])) return;
    var ts = parseInt(c[0], 10), d = parseFloat(c[1]);
    if (isNaN(d)) return;
    if (c.length >= 10 && !S.serial.warned) {
      S.serial.warned = true;
      log("info", "i", "Device is in feature-log mode: invalid readings aren't sent, so sensor faults won't show. Use LOG_RAW=1 for the full picture.", ts);
    }
    $("serial-status").textContent = "Connected · " + (S.stats.readings + 1) + " readings received";
    ingest(d, ts, null, null);
  }
  function disconnectSerial() {
    var s = S.serial, port = s.port;
    s.port = null;
    var p = s.reader ? s.reader.cancel().catch(function () {}) : Promise.resolve();
    s.reader = null;
    p.then(function () { return s.closed; }).then(function () { return port && port.close(); }).catch(function () {});
    $("serial-status").textContent = "Disconnected";
  }

  // ── Mode / speed / buttons ────────────────────────────────────────────────
  function setMode(mode, keepData) {
    if (S.mode === "serial" && mode !== "serial" && S.serial.port) disconnectSerial();
    S.mode = mode;
    if (mode === "replay" && !keepData) { S.replay.data = REPLAY; S.replay.events = null; }
    resetRun();
    if (mode === "replay") { setSpeed(5); $("replay-name").textContent = S.replay.data.name; }
    if (mode === "sim") setSpeed(1);
    S.paused = false;
    $("panel-sim").hidden = mode !== "sim";
    $("panel-replay").hidden = mode !== "replay";
    $("panel-serial").hidden = mode !== "serial";
    var shown = (mode === "replay" && S.replay.data !== REPLAY) ? "upload" : mode;
    Array.prototype.forEach.call($("seg-mode").querySelectorAll("button"), function (b) {
      b.setAttribute("aria-pressed", b.dataset.mode === shown ? "true" : "false");
    });
    renderEventsCard();
    updateButtons();
    S.dirty = true;
  }
  function setSpeed(v) {
    S.speed = v;
    Array.prototype.forEach.call($("seg-speed").querySelectorAll("button"), function (b) {
      b.setAttribute("aria-pressed", Number(b.dataset.speed) === v ? "true" : "false");
    });
    S.dirty = true;
  }
  function updateButtons() {
    var live = S.mode === "serial";
    $("btn-pause").disabled = live;
    $("btn-pause").textContent = S.paused ? "Resume" : "Pause";
    Array.prototype.forEach.call($("seg-speed").querySelectorAll("button"), function (b) { b.disabled = live; });
  }

  $("seg-mode").addEventListener("click", function (e) {
    var b = e.target.closest("button");
    if (!b) return;
    var m = b.dataset.mode;
    if (m === "upload") $("file-input").click();
    else if (m === "serial") connectSerial();
    else setMode(m);
  });
  $("seg-speed").addEventListener("click", function (e) {
    var b = e.target.closest("button");
    if (b) setSpeed(Number(b.dataset.speed));
  });
  $("btn-pause").addEventListener("click", function () {
    if (S.finished) { setMode(S.mode, true); return; }
    S.paused = !S.paused; updateButtons(); S.dirty = true;
  });
  $("btn-restart").addEventListener("click", function () {
    if (S.mode === "serial") { resetRun(); return; }
    setMode(S.mode, true);
  });
  $("btn-ghost").addEventListener("click", function () { S.sim.ghost = true; });
  $("btn-disconnect").addEventListener("click", function () { if (S.serial.port) disconnectSerial(); });

  // Pause the simulation clock while the tab is hidden (no catch-up burst)
  document.addEventListener("visibilitychange", function () { S.lastFrame = null; });

  // ── Hero KPIs ─────────────────────────────────────────────────────────────
  (function heroKpis() {
    var PR = window.ESP32_PROJECT, ev = 0, det = 0, fa = 0, mins = 0;
    if (PR) PR.results.forEach(function (r) {
      ev += r.events.length; det += r.events.filter(function (e) { return e.detected; }).length;
      fa += r.false_alarms; mins += r.normal_minutes;
    });
    var items = [
      ["118", "model parameters (472 B of flash)"],
      ["0 B", "heap used on the ESP32"],
      [PR ? det + " / " + ev : "–", "anomaly events detected (synthetic test set)"],
      [PR ? String(fa) : "–", "false alarms in " + Math.round(mins) + " min of normal data"],
      ["5 Hz", "sampling, 80 MHz CPU, < 1 % load"],
    ];
    items.forEach(function (k) {
      var d = html("div", "kpi", null, $("kpis"));
      html("div", "kpi-val", k[0], d);
      html("div", "kpi-label", k[1], d);
    });
  })();

  setMode("sim");
  requestAnimationFrame(frame);
})();
