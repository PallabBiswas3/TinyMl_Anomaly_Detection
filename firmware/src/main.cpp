#include <Arduino.h>
#include "sensor.h"
#include "preprocess.h"
#include "inference.h"
#include "model_data.h"

// ── Build options (set in platformio.ini) ─────────────────────────────────────

#ifndef LOG_RAW
#define LOG_RAW 0          // 1 = print raw readings only (record training data with capture_serial.py)
#endif
#ifndef CPU_FREQ_MHZ
#define CPU_FREQ_MHZ 80    // the workload uses <1 % of an 80 MHz core
#endif
static_assert(CPU_FREQ_MHZ == 80 || CPU_FREQ_MHZ == 160 || CPU_FREQ_MHZ == 240,
              "CPU_FREQ_MHZ must be 80, 160 or 240 (lower clocks slow the APB/UART)");

// ── Config ────────────────────────────────────────────────────────────────────

#define SERIAL_BAUD      115200
#define SAMPLE_INTERVAL  MODEL_SAMPLE_INTERVAL_MS   // from training - must match
#define ANOMALY_LED_PIN  2                          // onboard LED: alarm or sensor fault
#define SUMMARY_INTERVAL 60000                      // ms between summaries

// ── Globals (all static allocation - no heap use anywhere) ──────────────────

SensorManager      sensors;
SignalPreprocessor preproc;
InferenceEngine    engine;
AlarmFilter        alarm_filter;

struct Stats {
    uint32_t readings = 0, invalid = 0, spikes = 0, anomalous = 0, alarms = 0, faults = 0;
    uint32_t inferences = 0;
    float    avg_us = 0, min_us = 1e9f, max_us = 0;

    void add_inference(uint32_t us) {
        inferences++;
        avg_us += (us - avg_us) / inferences;
        if (us < min_us) min_us = us;
        if (us > max_us) max_us = us;
    }

    void print(const PowerReading& p) const {
        Serial.println("# ── Summary ─────────────────────────────────────");
        Serial.printf("#   readings %lu | invalid %lu (%.1f%%) | spikes removed %lu\n",
            (unsigned long)readings, (unsigned long)invalid,
            readings ? 100.0f * invalid / readings : 0.0f, (unsigned long)spikes);
        Serial.printf("#   anomalous samples %lu | alarms raised %lu | sensor faults %lu\n",
            (unsigned long)anomalous, (unsigned long)alarms, (unsigned long)faults);
        Serial.printf("#   inference avg %.1f us (min %.0f / max %.0f) @ %lu MHz\n",
            avg_us, min_us, max_us, (unsigned long)getCpuFrequencyMhz());
        Serial.printf("#   power ~%.0f mW%s | free heap %lu B\n", p.power_mW,
            p.available ? "" : " (datasheet estimate)", (unsigned long)ESP.getFreeHeap());
    }
} stats;

// ── Setup ─────────────────────────────────────────────────────────────────────

void setup() {
    // Lower the clock first so the UART baud rate is configured for it
    setCpuFrequencyMhz(CPU_FREQ_MHZ);

    Serial.begin(SERIAL_BAUD);
    delay(500);

    Serial.println("\n╔══════════════════════════════════════════╗");
    Serial.println("║  ESP32 HC-SR04 TinyML Anomaly Detector  ║");
    Serial.println("╚══════════════════════════════════════════╝\n");
    Serial.printf("[System] Chip %s rev%d | CPU %lu MHz | free heap %lu B\n",
        ESP.getChipModel(), ESP.getChipRevision(),
        (unsigned long)getCpuFrequencyMhz(), (unsigned long)ESP.getFreeHeap());
    Serial.printf("[System] Mode: %s sensor, %s log\n",
        SENSOR_SIMULATION ? "SIMULATED" : "REAL", LOG_RAW ? "RAW" : "FEATURE");

    pinMode(ANOMALY_LED_PIN, OUTPUT);
    digitalWrite(ANOMALY_LED_PIN, LOW);

    sensors.begin();

    Serial.printf("[Model] threshold %.6f | Hampel-%d, window %d, %d ms period | "
                  "alarm %d on / %d off, fault after %d invalid\n",
        InferenceEngine::threshold(), MODEL_HAMPEL_SIZE, MODEL_WINDOW_SIZE,
        MODEL_SAMPLE_INTERVAL_MS, MODEL_ALARM_ON_COUNT, MODEL_ALARM_OFF_COUNT,
        MODEL_FAULT_INVALID_COUNT);

    // Everything after this header is CSV; status lines start with '#'
    Serial.println(LOG_RAW
        ? "timestamp_ms,distance_cm"
        : "timestamp_ms,raw_cm,filtered_cm,jitter,rate_of_change,variance,mse,anomaly,alarm,inference_us");
}

// ── Loop ──────────────────────────────────────────────────────────────────────

void loop() {
    static uint32_t next_ms      = millis();
    static uint32_t last_summary = millis();
    static bool     was_fault    = false;

    // Sleep until the next sample slot. delay() yields to FreeRTOS, whose idle
    // task halts the core (WAITI) - far less power than spinning on millis().
    const int32_t wait = (int32_t)(next_ms - millis());
    if (wait > 0) delay(wait);
    next_ms += SAMPLE_INTERVAL;
    if ((int32_t)(millis() - next_ms) >= 0) next_ms = millis() + SAMPLE_INTERVAL;  // overran: resync

    // 1. Acquire (raw, uncleaned)
    const RawReading raw = sensors.read();
    stats.readings++;
#if LOG_RAW
    Serial.printf("%lu,%.2f\n", (unsigned long)raw.timestamp_ms, raw.distance_cm);
#endif

    // 2. Clean + extract features (identical to training)
    Features f;
    const bool valid = preproc.push(raw.distance_cm, raw.timestamp_ms, f);
    const bool fault = preproc.sensor_fault();
    if (fault != was_fault) {
        if (fault) stats.faults++;
        Serial.println(fault ? "# [Sensor] FAULT: no valid echo for ~1 s "
                               "(blocked, unplugged or object far away)"
                             : "# [Sensor] valid echoes restored");
        was_fault = fault;
    }

    if (!valid) {
        stats.invalid++;
        digitalWrite(ANOMALY_LED_PIN, (alarm_filter.active() || fault) ? HIGH : LOW);
    } else {
        if (f.filtered_cm != raw.distance_cm) stats.spikes++;   // Hampel replaced this reading

        // 3. Inference (timed)
        const uint32_t t0 = micros();
        const InferenceResult r = engine.run(f);
        const uint32_t us = micros() - t0;
        stats.add_inference(us);

        // 4. Decision: debounced alarm drives the LED
        const bool was_alarm = alarm_filter.active();
        const bool alarm     = alarm_filter.update(r.is_anomaly);
        if (r.is_anomaly) stats.anomalous++;
        if (alarm && !was_alarm) stats.alarms++;
        digitalWrite(ANOMALY_LED_PIN, alarm ? HIGH : LOW);

#if !LOG_RAW
        Serial.printf("%lu,%.2f,%.2f,%.4f,%.4f,%.4f,%.6f,%d,%d,%lu\n",
            (unsigned long)raw.timestamp_ms, raw.distance_cm, f.filtered_cm,
            f.jitter, f.rate_of_change, f.variance, r.reconstruction_mse,
            r.is_anomaly ? 1 : 0, alarm ? 1 : 0, (unsigned long)us);
#endif
    }

    if ((uint32_t)(millis() - last_summary) >= SUMMARY_INTERVAL) {
        last_summary = millis();
        stats.print(sensors.read_power());
    }
}
