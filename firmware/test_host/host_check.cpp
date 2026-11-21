// Host-side harness: runs the firmware's real preprocessing + inference code
// (src/preprocess.cpp, src/inference.cpp - no Arduino needed) over a raw CSV
// capture and prints the per-sample results. Built and compared against the
// Python pipeline by training/verify_firmware.py.
//
//   g++ -std=gnu++11 -O2 -I../include host_check.cpp ../src/preprocess.cpp ../src/inference.cpp

#include <cstdio>
#include <cstdlib>
#include "preprocess.h"
#include "inference.h"

int main(int argc, char** argv) {
    FILE* in = (argc > 1) ? std::fopen(argv[1], "r") : stdin;
    if (!in) { std::perror("open"); return 1; }

    SignalPreprocessor pre;
    InferenceEngine    engine;
    AlarmFilter        alarm;
    char line[512];

    std::printf("timestamp_ms,valid,filtered_cm,jitter,rate_of_change,variance,mse,alarm,fault\n");
    while (std::fgets(line, sizeof line, in)) {
        char* end = nullptr;
        const unsigned long ts = std::strtoul(line, &end, 10);
        if (end == line || *end != ',') continue;          // comment / header / junk
        char* dend = nullptr;
        const float raw = std::strtof(end + 1, &dend);
        if (dend == end + 1) continue;

        Features f;
        if (!pre.push(raw, (uint32_t)ts, f)) {
            std::printf("%lu,0,,,,,,%d,%d\n", ts, alarm.active(), pre.sensor_fault());
            continue;
        }
        const InferenceResult r = engine.run(f);
        const bool a = alarm.update(r.is_anomaly);
        std::printf("%lu,1,%.6f,%.6f,%.6f,%.6f,%.9g,%d,%d\n", ts, f.filtered_cm, f.jitter,
                    f.rate_of_change, f.variance, r.reconstruction_mse, a, pre.sensor_fault());
    }
    return 0;
}
