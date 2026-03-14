#include <Arduino.h>
#include "sensor.h"
#include "inference.h"

// ── Config ────────────────────────────────────────────────────────────────────

#define SERIAL_BAUD      115200
#define SAMPLE_INTERVAL  200     // ms — HC-SR04 needs ~60ms min between reads
#define ANOMALY_LED_PIN  2       // onboard LED
#define LOG_CSV          true    // CSV output for plotting

// ── Globals ───────────────────────────────────────────────────────────────────

SensorManager   sensors;
InferenceEngine engine;

struct Stats {
    uint32_t total        = 0;
    uint32_t anomalies    = 0;
    float    avg_inf_us   = 0;
    float    min_inf_us   = 1e9f;
    float    max_inf_us   = 0;
    float    avg_mse      = 0;
    float    avg_power_mw = 0;

    void update(const InferenceResult& r, float power_mw) {
        total++;
        if (r.is_anomaly) anomalies++;
        avg_inf_us   = (avg_inf_us   * (total-1) + r.inference_us)         / total;
        avg_mse      = (avg_mse      * (total-1) + r.reconstruction_mse)   / total;
        avg_power_mw = (avg_power_mw * (total-1) + power_mw)               / total;
        if (r.inference_us < min_inf_us) min_inf_us = r.inference_us;
        if (r.inference_us > max_inf_us) max_inf_us = r.inference_us;
    }

    void print_summary() const {
        Serial.println("\n══════════════════════════════════════════");
        Serial.println("  Benchmark Summary (HC-SR04 Anomaly Detector)");
        Serial.println("══════════════════════════════════════════");
        Serial.printf("  Total readings    : %lu\n",   total);
        Serial.printf("  Anomalies detected: %lu (%.1f%%)\n",
            anomalies, 100.0f * anomalies / total);
        Serial.printf("  Avg inference     : %.1f μs\n", avg_inf_us);
        Serial.printf("  Min / Max         : %.1f / %.1f μs\n", min_inf_us, max_inf_us);
        Serial.printf("  Avg MSE           : %.6f\n",   avg_mse);
        Serial.printf("  Avg power draw    : %.1f mW\n", avg_power_mw);
        Serial.println("══════════════════════════════════════════\n");
    }
} stats;

// ── Setup ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(SERIAL_BAUD);
    delay(500);

    Serial.println("\n╔══════════════════════════════════════════╗");
    Serial.println("║  ESP32 HC-SR04 TinyML Anomaly Detector  ║");
    Serial.println("╚══════════════════════════════════════════╝\n");

    Serial.printf("[System] Chip    : %s rev%d\n",
        ESP.getChipModel(), ESP.getChipRevision());
    Serial.printf("[System] CPU freq: %d MHz\n", ESP.getCpuFreqMHz());
    Serial.printf("[System] Free heap: %lu bytes\n", ESP.getFreeHeap());

    pinMode(ANOMALY_LED_PIN, OUTPUT);
    digitalWrite(ANOMALY_LED_PIN, LOW);

    if (!sensors.begin()) {
        Serial.println("[ERROR] Sensor init failed — check HC-SR04 wiring!");
    }

    if (!engine.begin()) {
        Serial.println("[ERROR] TFLite init failed! Halting.");
        while (true) delay(1000);
    }

    Serial.printf("[System] Free heap after init: %lu bytes\n\n",
        ESP.getFreeHeap());

    if (LOG_CSV) {
        Serial.println(
            "timestamp_ms,distance_cm,jitter,rate_of_change,variance,"
            "mse,is_anomaly,inference_us,power_mw"
        );
    }
}

// ── Loop ──────────────────────────────────────────────────────────────────────

void loop() {
    static uint32_t last_sample  = 0;
    static uint32_t last_summary = 0;
    uint32_t now = millis();

    if (now - last_sample >= SAMPLE_INTERVAL) {
        last_sample = now;

        // Read HC-SR04 + extract features
        SensorFeatures feat  = sensors.read();
        PowerReading   power = sensors.read_power();

        // Run TFLite inference
        InferenceResult result = engine.run(
            feat.distance_cm,
            feat.jitter,
            feat.rate_of_change,
            feat.variance
        );

        stats.update(result, power.power_mW);

        // Blink LED on anomaly
        digitalWrite(ANOMALY_LED_PIN, result.is_anomaly ? HIGH : LOW);

        if (LOG_CSV) {
            Serial.printf(
                "%lu,%.2f,%.4f,%.4f,%.4f,%.6f,%d,%lu,%.1f\n",
                feat.timestamp_ms,
                feat.distance_cm,
                feat.jitter,
                feat.rate_of_change,
                feat.variance,
                result.reconstruction_mse,
                result.is_anomaly ? 1 : 0,
                result.inference_us,
                power.power_mW
            );
        } else {
            SensorManager::print(feat);
            InferenceEngine::print(result);
            SensorManager::print_power(power);
            Serial.println();
        }
    }

    // Print benchmark summary every 60s
    if (now - last_summary >= 60000 && stats.total > 0) {
        last_summary = now;
        stats.print_summary();
    }
}
