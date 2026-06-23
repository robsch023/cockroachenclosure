import time
import datetime

INPUT_PIN = 18   # BCM pin — must match REMOTE_GPIO_PIN in door controller

try:
    from gpiozero import Button
except ImportError:
    raise SystemExit("gpiozero not found — activate your venv first.")

pin = Button(INPUT_PIN, pull_up=None, active_state=True)

print(f"[RECEIVER] Listening on BCM {INPUT_PIN}. Ctrl+C to stop.")
print(f"[RECEIVER] Current state: {'HIGH' if pin.is_pressed else 'LOW'}")

last_state = pin.is_pressed

try:
    while True:
        current_state = pin.is_pressed
        if current_state != last_state:
            ts    = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            level = "HIGH ↑" if current_state else "LOW  ↓"
            print(f"[{ts}] Pin BCM {INPUT_PIN} → {level}")
            last_state = current_state
        time.sleep(0.01)   # 10 ms poll — fast enough to catch 2-second pulses
except KeyboardInterrupt:
    print("\n[RECEIVER] Stopped.")
finally:
    pin.close()
