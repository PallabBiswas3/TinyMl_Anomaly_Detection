#include "sensor.h"
#include <math.h>

// ── SensorManager::begin ──────────────────────────────────────────────────────

bool SensorManager::begin() {
    Serial.println("[Sensor] Initializing HC-SR04...");

    pinMode(HCSR04_TRIG_PIN, OUTPUT);
    pinMode(HCSR04_ECHO_PIN, INPUT);
    digitalWrite(HCSR04_TRIG_PIN, LOW);
    delayMicroseconds(2);

    RawReading r = read_hcsr04();
    if (r.distance_cm > 0.0f) {
        Serial.printf("[Sensor] HC-SR04 OK — first reading: %.1f cm\n", r.distance_cm);
    } else {
        Serial.println("[Sensor] HC-SR04 WARNING: no echo on first reading (check wiring)");
    }

#if SENSOR_SIMULATION
    Serial.println("[Sensor] SIMULATION MODE: synthetic readings incl. noise, dropouts, spikes");
#endif
    Serial.println("[Sensor] INA219: not connected (power is a datasheet estimate)");
    return true;
}

// ── HC-SR04 single-shot distance read ────────────────────────────────────────
// 10 µs TRIG pulse, then measure the ECHO high time.
// pulseIn() counts CPU cycles and scales by the current CPU frequency, so it
// stays correct at 80/160/240 MHz. Worst case it blocks for HCSR04_TIMEOUT_US.

RawReading SensorManager::read_hcsr04() {
    RawReading r;
    r.timestamp_ms = millis();

    digitalWrite(HCSR04_TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(HCSR04_TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(HCSR04_TRIG_PIN, LOW);

    const uint32_t echo_us = pulseIn(HCSR04_ECHO_PIN, HIGH, HCSR04_TIMEOUT_US);
    r.distance_cm = (echo_us == 0) ? -1.0f
                                   : echo_us * 0.01715f;   // 343 m/s, round trip / 2
    return r;   // validated later by SignalPreprocessor (single place, same as training)
}

// ── Simulator ─────────────────────────────────────────────────────────────────
// Cycles through 10-second scenarios and adds the same kinds of real-world
// defects the cleaning stage must handle: noise, dropped echoes, ghost spikes.

namespace {

struct SimScenario {
    const char* name;
    float base_cm;
    float noise_cm;
    float vibration_cm;   // sine amplitude (0 = none)
    bool  no_echo;        // simulate a dead/blocked sensor
};

const SimScenario kScenarios[] = {
    {"NORMAL (object at 30 cm)",                   30.0f, 0.20f, 0.0f, false},
    {"ANOMALY: obstruction at 8 cm",                8.0f, 0.30f, 0.0f, false},
    {"NORMAL (object at 30 cm)",                   30.0f, 0.20f, 0.0f, false},
    {"ANOMALY: vibration +/-3 cm",                 30.0f, 0.20f, 3.0f, false},
    {"NORMAL (object at 30 cm)",                   30.0f, 0.20f, 0.0f, false},
    {"ANOMALY: object removed (wall at 150 cm)",  150.0f, 0.60f, 0.0f, false},
    {"NORMAL (object at 30 cm)",                   30.0f, 0.20f, 0.0f, false},
    {"FAULT: no echo (sensor blocked/unplugged)",   0.0f, 0.00f, 0.0f, true },
};
constexpr uint8_t  kNumScenarios = sizeof(kScenarios) / sizeof(kScenarios[0]);
constexpr uint32_t kPhaseMs      = 10000;

inline float frand() { return (float)rand() / (float)RAND_MAX; }               // [0,1]
inline float gauss() { return (frand() + frand() + frand() + frand() - 2.0f)   // ≈ N(0,1)
                              * 1.7320508f; }

} // namespace

float SensorManager::simulated_distance(uint32_t now_ms) {
    const uint8_t phase = (now_ms / kPhaseMs) % kNumScenarios;
    if (phase != sim_phase_) {
        sim_phase_ = phase;
        Serial.printf("# [SIMULATION] %s\n", kScenarios[phase].name);
    }
    const SimScenario& s = kScenarios[phase];
    if (s.no_echo) return -1.0f;

    const float r = frand();
    if (r < 0.015f) return -1.0f;                        // dropped echo (1.5 %)
    if (r < 0.025f) return 2.0f + frand() * 398.0f;      // ghost echo / spike (1 %)

    float d = s.base_cm + s.noise_cm * gauss();
    if (s.vibration_cm > 0.0f)
        d += s.vibration_cm * sinf(2.0f * PI * 1.7f * (now_ms / 1000.0f));
    return d;
}

// ── Public read ───────────────────────────────────────────────────────────────

RawReading SensorManager::read() {
#if SENSOR_SIMULATION
    RawReading r;
    r.timestamp_ms = millis();
    r.distance_cm  = simulated_distance(r.timestamp_ms);
    return r;
#else
    return read_hcsr04();
#endif
}

// ── Power (estimate) ──────────────────────────────────────────────────────────
// Without an INA219 this is the ESP32 datasheet's upper bound for the chip with
// radios off (modem-sleep) at the current CPU clock - not a measurement.

PowerReading SensorManager::read_power() const {
    PowerReading p;
    p.available  = false;
    p.voltage_mV = 3300.0f;
    const uint32_t mhz = getCpuFrequencyMhz();
    p.current_mA = (mhz >= 240) ? 68.0f : (mhz >= 160) ? 44.0f : 31.0f;
    p.power_mW   = p.voltage_mV * p.current_mA / 1000.0f;
    return p;
}
