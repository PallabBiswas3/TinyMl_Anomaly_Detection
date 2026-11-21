# ESP32 HC-SR04 TinyML Anomaly Detector

Real-time anomaly detection on an ESP32 from a cheap ultrasonic sensor, built
for **messy real-world data**: noise, dropped echoes, ghost reflections and
temperature drift are cleaned on the device by the *same* pipeline the model
was trained on. The model is a **pure C++ autoencoder**: no TensorFlow runtime and no
libraries. Its weights are compiled into flash as C arrays.

| | |
|---|---|
| On-device pipeline (cleaning + model + alarm) | **≈1.7 KB flash, 80 B RAM, 0 B heap** |
| Model | 4-8-2-8-4 autoencoder, 118 parameters (472 B), 96 MACs |
| Detection (synthetic test set) | **20/20 anomaly events**, 0.8–1.3 s latency |
| False alarms | **0 in 75 min** of test data; 1 in 6 h of extra soak runs |
| CPU | 80 MHz (<1 % load), sleeps between samples |

---

## Web dashboard

`web/` is a static page that runs **the same pipeline with the same trained weights** in the
browser and explains every part of the project (hardware, data, cleaning, training, model,
firmware, verification, results). Data sources:

- **Live simulator**, mirroring the firmware's scenarios. You can switch scenarios and inject ghost echoes.
- **Replay** of the labelled test capture, with ground truth and per-event detection. *Watch* jumps to any event.
- **Upload** any CSV recorded with `capture_serial.py`.
- **Connect a real ESP32** over USB (Web Serial, Chrome/Edge). Flash with `-DLOG_RAW=1` so the browser gets every raw reading.

Open `web/index.html` locally, or deploy it with the included GitHub Actions workflow
(`.github/workflows/pages.yml`). After retraining, refresh and check it:

```bash
cd training
python export_web.py     # web/model.js, replay.js, project.js from firmware/include + data
python verify_web.py     # JS pipeline == Python, sample by sample (needs Node.js)
```

---

## The pipeline

```
 HC-SR04 ──▶ raw reading ──▶ ① validity gate ──▶ ② gap reset ──▶ ③ Hampel spike filter ──▶ ④ window features
  (5 Hz)     (-1 = no echo)    2–400 cm, NaN     >1 s without      replace a reading only     distance, jitter,
                               → else dropped    a valid echo →    if it is an outlier vs.    rate of change,
                                                 restart filters   its 7 neighbours           variance (8 samples)
                                                                                                   │
            LED ◀── ⑥ alarm debounce ◀── MSE > threshold? ◀── ⑤ autoencoder (reconstruction error) ◀┘
                    3 anomalous in a row = ALARM, 3 normal = clear
                    5 invalid readings in a row = SENSOR FAULT
```

Steps ①–④ and ⑥ exist **twice**, in `firmware/src/preprocess.cpp` and in
`training/pipeline.py`. The settings are generated into `model_data.h`, so the
two copies can't drift apart. `verify_firmware.py` compiles the C++ on your PC
and checks it against Python, sample by sample.

### Why each cleaning step exists

| Real-world problem | What the device does |
|---|---|
| No echo: timeout returns `-1` (soft or angled target, out of range) | ① dropped. 5 in a row (~1 s) raises a **sensor fault** |
| Transducer ringing (< 2 cm), over-range readings | ① dropped |
| Ghost echoes: multipath double-bounce, side reflectors, electrical junk | ③ **Hampel filter** replaces isolated outliers with the local median |
| Long gaps (sensor blocked, reboot) | ② stale history is discarded, so features restart cleanly |
| Measurement noise, temperature drift (speed of sound ±0.6 %/°C) | learned as *normal*, because the training data contains it |

**Why Hampel instead of a plain median filter?** A median over N samples also
cancels any vibration faster than its window: the samples walk around the sine
wave and their median sits at the centre. With a median-5 filter, a ±3 cm
vibration at 1.7 Hz was detected only **21 %** of the time. The Hampel filter
only replaces readings that are outliers relative to the spread of their neighbours,
so a consistent oscillation passes through. The same vibration is now detected
**99 %** of the time.

### Training-only cleaning (on the PC)

`train.py` also removes **contamination** from the "normal" recordings, such as a hand
in the beam or a bumped object. A row is dropped when its filtered distance leaves
the recording's robust baseline (median ± 6·MAD, minimum 2 cm), along with the
neighbouring rows whose feature window overlaps the event. Otherwise the model would
learn those events as normal.

### What the cleaning is worth

`python train.py --compare-raw` also trains a model on the same data with no
cleaning at all:

| | With cleaning | Without cleaning |
|---|---|---|
| Events detected | **20 / 20** | 4 / 20 (sensor faults only) |
| False alarms | **0 / hour** | 3.2 / hour |

Without cleaning, the spikes stretch the normalization range, so real anomalies look tiny to the model.

---

## Features the model uses

| Feature | Description | Anomaly signal |
|---|---|---|
| `distance_cm` | Spike-filtered distance | Too close / object gone / displaced |
| `jitter` | Std dev over the last 8 samples | Vibration, instability |
| `rate_of_change` | \|Δdistance\| per second | Sudden movement |
| `variance` | Variance over the last 8 samples | Erratic behaviour |

## Anomaly scenarios detected

| Scenario | Example | Detected by |
|---|---|---|
| Obstruction | Something enters the beam | model (distance) |
| Object removed | Part fell off; echo now from the back wall | model (distance) |
| Displaced | Object shifted 4–10 cm and stays | model (distance) |
| Vibration | Loose mount, machine vibration (≥ ~1 cm) | model (jitter, variance, rate) |
| Sensor fault | Cable loose, transducer covered | invalid-reading counter |

**Limits of a 5 Hz ultrasonic sensor.** Vibration is only visible through aliasing,
and vibration at an exact multiple of 5 Hz aliases to a constant, so no 5 Hz sensor can see it.
A sensor that freezes on a constant valid value looks like a perfectly still object.

---

## How the model fits in such a small space

Normally you would train, convert to TFLite, and ship the TFLite Micro runtime (~300 KB
flash plus a tensor arena). Here the Keras weights are exported as C arrays and the
forward pass is about 30 lines of C++:

```
Input (4 floats)
      ↓  Dense(4→8, ReLU)     ← 32 MACs
      ↓  Dense(8→2, ReLU)     ← 16 MACs   } encoder
      ↓  Dense(2→8, ReLU)     ← 16 MACs
      ↓  Dense(8→4, Sigmoid)  ← 32 MACs   } decoder
MSE(input, output) > threshold → anomalous sample
```

Measured from the compiled firmware (`xtensa-esp32-elf-nm`):

| Component | Size | Where |
|---|---|---|
| Weights + biases | 464 B | flash (`0x3F40xxxx`, read-only data) |
| Inference code | ~0.5 KB | flash |
| Cleaning + features code | ~0.6 KB | flash |
| Preprocessor state | 80 B | RAM |
| Heap | 0 B | none |

Embedded design choices:
- **Single-precision floats only.** The ESP32 FPU has no double-precision support.
- **Normalization uses a precomputed `1/range`**, so the device multiplies instead of dividing.
- **Templated layers** (`dense<IN,OUT,ACT>`) let the compiler check every shape and unroll the loops.
- **80 MHz CPU clock.** One sample needs a few µs of maths plus at most 25 ms waiting for the echo,
  out of each 200 ms period. The datasheet gives roughly half the current draw of 240 MHz (radio off).
- **`delay()` until the next sample slot** instead of spinning on `millis()`, so the FreeRTOS idle task halts the core.
- `pulseIn()` counts CPU cycles scaled by the current clock, so distance accuracy doesn't depend on the CPU frequency.

---

## Hardware

| HC-SR04 Pin | ESP32 Pin | Note |
|---|---|---|
| VCC | **5V / VIN** | Must be 5 V, not 3.3 V |
| GND | GND | |
| TRIG | GPIO5 | |
| ECHO | GPIO18 | Through a voltage divider (below) |

```
ECHO (5V) ──┬── 1kΩ ──→ GPIO18        HC-SR04 drives 5 V; ESP32 GPIOs are 3.3 V max
            └── 2kΩ ──→ GND
```

The onboard LED on GPIO2 lights for an **alarm** or a **sensor fault**.

---

## Project structure

```
esp32-ultrasonic-anomaly/
├── training/
│   ├── generate_data.py      ← realistic synthetic raw recordings (noise, dropouts, spikes, drift…)
│   ├── capture_serial.py     ← record REAL raw data from the ESP32
│   ├── pipeline.py           ← Python mirror of the on-device cleaning + evaluation helpers
│   ├── train.py              ← clean → train → threshold → evaluate → export headers
│   ├── verify_firmware.py    ← compiles the firmware C++ on the PC, compares with Python
│   ├── export_web.py         ← model + data for the web dashboard
│   ├── verify_web.py         ← checks the dashboard's JS pipeline against Python
│   ├── data/
│   │   ├── normal/*.csv      ← normal-operation captures (training)
│   │   └── test/*.csv        ← labelled captures (evaluation) + 1 h soak
│   └── output/               ← generated model_data.h / model_weights.h
└── firmware/
    ├── platformio.ini        ← SENSOR_SIMULATION, LOG_RAW, CPU_FREQ_MHZ
    ├── include/              ← model_data.h / model_weights.h are copied here by train.py
    ├── src/
    │   ├── main.cpp          ← scheduling, alarm/fault handling, logging
    │   ├── sensor.cpp        ← HC-SR04 driver + realistic simulator (raw readings only)
    │   ├── preprocess.cpp    ← cleaning + features (pure C++, no Arduino)
    │   └── inference.cpp     ← autoencoder + alarm debounce (pure C++, no Arduino)
    └── test_host/host_check.cpp  ← runs preprocess + inference on the PC
web/                          ← static dashboard (pipeline.js = JS mirror; model/replay/project.js generated)
```

---

## Quick start (synthetic data)

```bash
cd training
pip install -r requirements.txt
python generate_data.py          # writes data/normal/*.csv and data/test/*.csv
python train.py                  # trains, evaluates, copies headers into firmware/include
python verify_firmware.py        # optional: needs g++ - checks C++ == Python

cd ../firmware
pio run --target upload          # simulation mode by default (no sensor needed)
pio device monitor --baud 115200
```

## Real deployment

1. **Wire the sensor** and set `-DSENSOR_SIMULATION=0 -DLOG_RAW=1` in `platformio.ini`. Flash it.
2. **Record normal operation** with the scene set up exactly as it will run. Close the serial monitor first.
   Record a few sessions at different times of day so temperature varies:
   ```bash
   python capture_serial.py --port COM5 --minutes 10 --out data/normal/site_01.csv
   ```
   Occasional disturbances are fine, because the contamination filter removes them. Delete the
   `synthetic_*` files or point `--normal-dir` at your own folder.
3. **Optional but recommended: record a test file.** Provoke each anomaly and add `label,scenario`
   columns, using `data/test/synthetic_test_01.csv` as the template. Without one, `train.py` skips evaluation.
4. **Train and verify:** `python train.py --normal-dir data/normal` then `python verify_firmware.py`.
5. **Deploy:** set `-DLOG_RAW=0` and flash.

---

## Serial output

Everything after the header is CSV, and status lines start with `#`
(`pandas.read_csv(path, comment="#")` works directly).

**Feature mode** (`LOG_RAW=0`). The example rows below are real output of the firmware
code run on the PC against the test data. `inference_us` is measured on the device.
```
timestamp_ms,raw_cm,filtered_cm,jitter,rate_of_change,variance,mse,anomaly,alarm,inference_us
20200,30.32,30.32,0.1966,0.4020,0.0386,0.004805,0,0,…          ← normal
75200,67.54,30.22,0.1704,0.2000,0.0290,0.006120,0,0,…          ← double-bounce echo removed
76002,356.95,30.37,0.1141,0.0000,0.0130,0.020550,0,0,…         ← junk reading removed
124400,150.49,150.49,58.5086,37.3367,3423.2520,92512840.000000,1,1,… ← ALARM: object removed
```

**Raw mode** (`LOG_RAW=1`, used for recording training data): `timestamp_ms,distance_cm`, with `-1` meaning no echo.

Every 60 s a `#` summary reports readings, invalid readings, spikes removed, alarms, faults,
inference time (avg/min/max at the current clock), estimated power and free heap.

## Simulation mode

`-DSENSOR_SIMULATION=1` (the default) cycles through 10-second scenarios:
normal, obstruction, vibration, object removed and sensor fault, with normal periods in between.
Every scenario includes 1.5 % dropped echoes and 1 % ghost spikes, so the cleaning stage is exercised too.

---

## Benchmarks

Measured on hardware **before** the cleaning pipeline was added (ESP32-D0WDQ5 rev3, 240 MHz):
inference **19 µs**, free heap 377 KB. The current firmware defaults to 80 MHz and prints the
measured inference time in its 60 s summary. Re-measure before quoting numbers.

Firmware size (current, real-sensor build): 216 KB flash (6.9 % of 3 MB), 13.6 KB static RAM.
Almost all of it is the Arduino/ESP-IDF framework.

**Note on the evaluation numbers:** they come from synthetic recordings made with a physical noise model
(`generate_data.py`). Expect different numbers on real data, and retrain on your own captures.

---

## Why this is portfolio-worthy

| Requirement | How we meet it |
|---|---|
| Beyond standard ONNX/TFLite | No TFLite runtime: pure C++ forward pass |
| Real-world data handling | On-device validity gate, Hampel filter, fault detection; offline contamination removal |
| Train/deploy consistency | One pipeline in two languages, with parameters generated from training and an automated C++ vs Python check |
| Honest evaluation | Event-based detection and latency, false alarms per hour, with vs. without cleaning |
| Memory-constrained deployment | ≈1.7 KB flash for the whole on-device pipeline, 80 B RAM, zero heap |
| Power awareness | 80 MHz clock, idle sleeping between samples |
