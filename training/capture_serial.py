"""
Record real HC-SR04 data from the ESP32 for training.

1. Flash the firmware with  -DSENSOR_SIMULATION=0  and  -DLOG_RAW=1
2. Set the scene up as it NORMALLY is (object in place, normal environment)
3. python capture_serial.py --port COM5 --minutes 10 --out data/normal/site_01.csv
   (close the PlatformIO serial monitor first - only one program can hold the port)

Record several sessions at different times (morning/afternoon temperature, etc.).
Occasional disturbances during a recording are fine - train.py removes them.
For a labelled test file, add columns label,scenario by hand (see data/test/).
"""

import argparse
import os
import sys
import time

try:
    import serial                      # pyserial (installed with PlatformIO)
except ImportError:
    sys.exit("pyserial is required: pip install pyserial")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="e.g. COM5 or /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--out", required=True, help="output CSV, e.g. data/normal/site_01.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows, invalid, started = 0, 0, False
    end = time.time() + args.minutes * 60

    # Opening the port resets the ESP32 (DTR), so we see the boot banner + header
    with serial.Serial(args.port, args.baud, timeout=1) as ser, \
            open(args.out, "w", encoding="ascii", newline="\n") as f:
        f.write(f"# real capture {time.strftime('%Y-%m-%d %H:%M:%S')} from {args.port}\n")
        print(f"Recording {args.minutes:g} min from {args.port} → {args.out}  (Ctrl+C to stop early)")
        try:
            while time.time() < end:
                line = ser.readline().decode("ascii", errors="replace").strip()
                if not line:
                    continue
                if line == "timestamp_ms,distance_cm":
                    started = True
                    f.write(line + "\n")
                    continue
                if line.startswith("timestamp_ms,raw_cm"):
                    sys.exit("Firmware is in FEATURE log mode - rebuild with -DLOG_RAW=1")
                if not started or line.startswith("#"):
                    continue
                parts = line.split(",")
                try:
                    int(parts[0]); d = float(parts[1])
                except (ValueError, IndexError):
                    continue                              # garbled line
                f.write(f"{parts[0]},{parts[1]}\n")
                rows += 1
                invalid += d < 0
                if rows % 50 == 0:
                    left = int(end - time.time())
                    print(f"\r  {rows} rows, {invalid} no-echo, {left // 60}:{left % 60:02d} left ",
                          end="", flush=True)
        except KeyboardInterrupt:
            pass
    print(f"\nSaved {rows} rows to {args.out}")
    if not started:
        print("WARNING: never saw the CSV header - is the firmware in LOG_RAW mode?")


if __name__ == "__main__":
    main()
