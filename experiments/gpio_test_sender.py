import time
import datetime

OUTPUT_PIN   = 17    # BCM pin on THIS Pi wired to BCM 18 on the receiver Pi
PULSE_HIGH_SEC = 2.0
PULSE_LOW_SEC  = 2.0

try:
    from gpiozero import LED
except ImportError:
    raise SystemExit("gpiozero not found — install with: pip install gpiozero")

pin = LED(OUTPUT_PIN)
pin.off()   # ensure we start LOW

print(f"[SENDER] Pulsing BCM {OUTPUT_PIN}  "
      f"(HIGH {PULSE_HIGH_SEC}s / LOW {PULSE_LOW_SEC}s). Ctrl+C to stop.")

try:
    while True:
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}] Pin BCM {OUTPUT_PIN} → HIGH ↑")
        pin.on()
        time.sleep(PULSE_HIGH_SEC)

        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}] Pin BCM {OUTPUT_PIN} → LOW  ↓")
        pin.off()
        time.sleep(PULSE_LOW_SEC)

except KeyboardInterrupt:
    print("\n[SENDER] Stopped.")
finally:
    pin.off()
    pin.close()
