"""
Robert Schaefer
25/05/2026

Cockroach detection + door controller for Raspberry Pi with Camera Module 3.

Extends cockroach_detector.py with:
  • Robotis AX-12A servo door actuation via USB2Dynamixel (dynamixel_sdk)
  • Day / night operating mode (NIGHT_MODE flag — flip to reverse all logic)
  • Remote-Pi open signal polled from a GPIO input pin
  • Clip recording triggered on any door-open event (same ClipWriter as original)
  • MCP3008 capacitive moisture sensor logged every 10 minutes to a daily CSV

Door logic summary
──────────────────
  NIGHT_MODE = True  (default — cockroaches are nocturnal)
    • Night  → door OPEN  when cockroach detected OR remote signal received
    • Day    → door CLOSED unconditionally
  NIGHT_MODE = False  (flip to reverse, e.g. diurnal species)
    • Day    → door OPEN  when cockroach detected OR remote signal received
    • Night  → door CLOSED unconditionally

The active period is defined by ACTIVE_START_HOUR / ACTIVE_END_HOUR (24-h).
Door only actuates on state transitions, not every frame.

Dependencies
────────────
  pip install ai-edge-litert opencv-python-headless picamera2 imutils
  pip install dynamixel-sdk          # Robotis official Python SDK
  pip install gpiozero               # for remote-Pi input pin (Pi 5 compatible)
  pip install spidev                 # MCP3008 moisture sensor via SPI
  pip install adafruit-circuitpython-sht4x adafruit-blinka  # SHT4x temp/humidity
"""

import cv2
import numpy as np
import datetime
import time
import os
import threading
from collections import deque
from picamera2 import Picamera2
import imutils

# ── dynamixel_sdk ─────────────────────────────────────────────────────────────
try:
    from dynamixel_sdk import (
        PortHandler, PacketHandler,
        COMM_SUCCESS,
    )
    DYNAMIXEL_AVAILABLE = True
except ImportError:
    DYNAMIXEL_AVAILABLE = False
    print("[WARN] dynamixel_sdk not found — servo will be simulated (dry run).")

# ── gpiozero (Pi 5 compatible) ────────────────────────────────────────────────
try:
    from gpiozero import Button, LED
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[WARN] gpiozero not found — remote-Pi polling and signal output disabled.")

# ── spidev (MCP3008 moisture sensor) ─────────────────────────────────────────
try:
    import spidev
    SPIDEV_AVAILABLE = True
except ImportError:
    SPIDEV_AVAILABLE = False
    print("[WARN] spidev not found — moisture sensor logging disabled.")

# ── adafruit SHT4x (temperature + humidity) ───────────────────────────────────
try:
    import board
    import adafruit_sht4x
    SHT4X_AVAILABLE = True
except ImportError:
    SHT4X_AVAILABLE = False
    print("[WARN] adafruit-circuitpython-sht4x not found — temp/humidity logging disabled.")

# ── Vision / detection ────────────────────────────────────────────────────────
MODEL_PATH        = "/home/rosch023/enclosure/model_float.tflite"
CONFIDENCE_THRESH = 0.11
TARGET_LABEL      = "cockroach"
FRAME_WIDTH       = 1640
FRAME_HEIGHT      = 922
FPS               = 16
SHOW_VIDEO        = True          # set True to preview on a connected display
SAVEPATH          = "/home/rosch023/local_mount/labtestdata/doorcam"
BUF_SIZE          = 32            # pre-roll frames kept before a trigger
IDLE_TIMEOUT_FRAMES = int(FPS * 10)  # frames of no-detection before clip closes

# ── Day / night schedule ──────────────────────────────────────────────────────
# NIGHT_MODE = True  → door logic is active at *night* (cockroach default).
# NIGHT_MODE = False → door logic is active during the *day* (flip to reverse).
NIGHT_MODE       = False
ACTIVE_START_HOUR = 20   # 8 pm  — start of active period
ACTIVE_END_HOUR   = 6    # 6 am  — end   of active period

# ── AX-12A servo (USB2Dynamixel) ─────────────────────────────────────────────
SERVO_PORT        = "/dev/ttyUSB0"   # adjust if different (check: ls /dev/ttyUSB*)
SERVO_BAUDRATE    = 1_000_000        # AX-12A default baud
SERVO_ID          = 1                # Dynamixel ID set on the servo
PROTOCOL_VERSION  = 1.0              # AX-12A uses Protocol 1.0

# AX-12A position range: 0–1023 → 0°–300°
# Tune these two values to match your physical door geometry.
DOOR_OPEN_POSITION   = 512   # ~150° (centred) — adjust to your open position
DOOR_CLOSED_POSITION = 200   # ~59°            — adjust to your closed position

# AX-12A control table addresses (Protocol 1.0)
ADDR_TORQUE_ENABLE   = 24
ADDR_GOAL_POSITION   = 30
ADDR_MOVING_SPEED    = 32
ADDR_PRESENT_POSITION = 36

SERVO_MOVING_SPEED   = 100   # 0–1023; lower = slower / gentler
SERVO_MOVE_TIMEOUT   = 3.0   # seconds to wait for move to complete

# ── Remote-Pi GPIO input ──────────────────────────────────────────────────────
USE_REMOTE_GPIO   = False    # set False to disable remote-Pi polling
REMOTE_GPIO_PIN   = 18      # BCM pin wired to the other Pi's output
# The other Pi should drive this pin HIGH to request door open.
# Wire a common GND between both Pis.

# ── Detection signal output (from original script) ────────────────────────────
USE_SIGNAL_GPIO   = False   # pulse a pin HIGH on cockroach detection (optional)
SIGNAL_GPIO_PIN   = 17

# ── Moisture sensor (MCP3008 via SPI) ────────────────────────────────────────
USE_MOISTURE      = True          # set False to disable entirely
MOISTURE_CHANNEL  = 0             # MCP3008 channel wired to sensor (0–7)
MOISTURE_INTERVAL = 10 * 60       # seconds between readings (default 10 min)
MOISTURE_LOG_DIR  = "/home/rosch023/local_mount/labtestdata/sensordata"    # daily CSV files written here
# DFRobot capacitive sensor calibration — adjust for your sensor/soil:
MOISTURE_DRY      = 750           # raw ADC value in dry air
MOISTURE_WET      = 400           # raw ADC value submerged in water
MOISTURE_VREF     = 3.3           # reference voltage (match your MCP3008 wiring)

# ── Temperature / humidity sensor (SHT4x via I2C) ────────────────────────────
USE_SHT4X         = True          # set False to disable entirely
# Precision mode — options: NOHEAT_HIGHPRECISION, NOHEAT_MEDPRECISION,
#                            NOHEAT_LOWPRECISION, LOWHEAT_100MS, HIGHHEAT_100MS
SHT4X_MODE        = "NOHEAT_HIGHPRECISION"

# ══════════════════════════════════════════════════════════════════════════════
#  GPIO INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

# ── GPIO pin objects (created once at startup) ────────────────────────────────
# gpiozero uses BCM numbering by default on the Pi 5.
# Button wraps an input with a built-in pull-down resistor.
# LED wraps an output (on/off) — named LED but works for any digital output.

_remote_pin: "Button | None" = None
_signal_pin: "LED | None"    = None

if GPIO_AVAILABLE:
    if USE_REMOTE_GPIO:
        # pull_up=False → internal pull-down; active when other Pi drives HIGH
        _remote_pin = Button(REMOTE_GPIO_PIN, pull_up=None, active_state=True)
    if USE_SIGNAL_GPIO:
        _signal_pin = LED(SIGNAL_GPIO_PIN)

# ══════════════════════════════════════════════════════════════════════════════
#  SERVO CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class DoorServo:
    """
    Thin wrapper around the AX-12A via USB2Dynamixel.

    Falls back to a dry-run (print-only) mode when dynamixel_sdk is absent or
    the port cannot be opened — useful for testing on a desktop.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._dry_run = not DYNAMIXEL_AVAILABLE

        if not self._dry_run:
            self._port    = PortHandler(SERVO_PORT)
            self._packet  = PacketHandler(PROTOCOL_VERSION)

            if not self._port.openPort():
                print(f"[SERVO] Cannot open {SERVO_PORT} — switching to dry-run.")
                self._dry_run = True
            elif not self._port.setBaudRate(SERVO_BAUDRATE):
                print("[SERVO] Cannot set baud rate — switching to dry-run.")
                self._dry_run = True
            else:
                self._write1(ADDR_TORQUE_ENABLE, 1)
                self._write2(ADDR_MOVING_SPEED,  SERVO_MOVING_SPEED)
                print(f"[SERVO] AX-12A ready on {SERVO_PORT} (ID {SERVO_ID})")

    # ── low-level helpers ─────────────────────────────────────────────────────

    def _write1(self, address: int, value: int):
        result, error = self._packet.write1ByteTxRx(
            self._port, SERVO_ID, address, value)
        self._check(result, error, address)

    def _write2(self, address: int, value: int):
        result, error = self._packet.write2ByteTxRx(
            self._port, SERVO_ID, address, value)
        self._check(result, error, address)

    def _read2(self, address: int) -> int:
        value, result, error = self._packet.read2ByteTxRx(
            self._port, SERVO_ID, address)
        self._check(result, error, address)
        return value

    def _check(self, result, error, address):
        if result != COMM_SUCCESS:
            print(f"[SERVO] Comm error at addr {address}: "
                  f"{self._packet.getTxRxResult(result)}")
        elif error != 0:
            print(f"[SERVO] Hardware error at addr {address}: "
                  f"{self._packet.getRxPacketError(error)}")

    # ── public API ────────────────────────────────────────────────────────────

    def move_to(self, position: int, label: str = ""):
        """Send goal position and block until the servo stops moving."""
        with self._lock:
            if self._dry_run:
                print(f"[SERVO] (dry-run) move → {position}  {label}")
                return

            self._write2(ADDR_GOAL_POSITION, position)
            deadline = time.time() + SERVO_MOVE_TIMEOUT
            while time.time() < deadline:
                present = self._read2(ADDR_PRESENT_POSITION)
                if abs(present - position) <= 5:   # within ±5 counts ≈ 1.5°
                    break
                time.sleep(0.05)
            print(f"[SERVO] Moved to {position}  {label}")

    def open_door(self):
        print("[DOOR] Opening door.")
        self.move_to(DOOR_OPEN_POSITION, "(OPEN)")

    def close_door(self):
        print("[DOOR] Closing door.")
        self.move_to(DOOR_CLOSED_POSITION, "(CLOSED)")

    def shutdown(self):
        if not self._dry_run:
            self._write1(ADDR_TORQUE_ENABLE, 0)   # release torque
            self._port.closePort()
            print("[SERVO] Port closed.")


# ══════════════════════════════════════════════════════════════════════════════
#  DAY / NIGHT SCHEDULE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def in_active_period() -> bool:
    """
    Return True when the current hour falls in the configured active window.
    Handles windows that wrap midnight (e.g. 20:00–06:00).
    The meaning of "active" depends on NIGHT_MODE:
      NIGHT_MODE=True  → active period is night-time (door may open at night)
      NIGHT_MODE=False → active period is day-time   (door may open by day)
    """
    hour = datetime.datetime.now().hour
    if ACTIVE_START_HOUR < ACTIVE_END_HOUR:
        # e.g. 08:00–18:00 — does not wrap midnight
        active = ACTIVE_START_HOUR <= hour < ACTIVE_END_HOUR
    else:
        # e.g. 20:00–06:00 — wraps midnight
        active = hour >= ACTIVE_START_HOUR or hour < ACTIVE_END_HOUR

    # When NIGHT_MODE is True the "active" window IS the night window above.
    # When NIGHT_MODE is False we want the complementary (day) window.
    return active if NIGHT_MODE else not active


# ══════════════════════════════════════════════════════════════════════════════
#  REMOTE GPIO POLL
# ══════════════════════════════════════════════════════════════════════════════

def remote_pi_requesting_open() -> bool:
    """Return True if the other Pi is driving REMOTE_GPIO_PIN HIGH."""
    if not USE_REMOTE_GPIO or _remote_pin is None:
        return False
    return _remote_pin.is_pressed   # True when pin is HIGH


# ══════════════════════════════════════════════════════════════════════════════
#  TFLITE IMPORT
# ══════════════════════════════════════════════════════════════════════════════

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

# These are populated inside main() after the camera is running
interpreter            = None
input_details          = None
output_details         = None
input_h                = None
input_w                = None
input_dtype            = None

# Hardcoded output tensor indices — these are specific to each exported
# model and MUST be re-checked whenever a new model is trained/exported.
# Use the tensor-inspection snippet (see chat history) to find new indices.
#
# INT8 model (model_int8.tflite):   boxes=498  scores+classes=495
# float32 model (model_float.tflite): boxes=431  scores+classes=429
IDX_BOXES              = 431
IDX_SCORES_AND_CLASSES = 429

LABELS     = {0: "background", 1: TARGET_LABEL}
ZONE_LEFT  = FRAME_WIDTH / 3
ZONE_RIGHT = 2 * FRAME_WIDTH / 3


# ══════════════════════════════════════════════════════════════════════════════
#  CLIP WRITER  (unchanged from cockroach_detector.py)
# ══════════════════════════════════════════════════════════════════════════════

class ClipWriter:
    """Rolling pre-roll buffer + OpenCV VideoWriter. Thread-safe."""

    def __init__(self, buf_size: int = BUF_SIZE):
        self._buf: deque = deque(maxlen=buf_size)
        self._writer = None
        self._lock   = threading.Lock()
        self.recording = False

    def update(self, frame: np.ndarray):
        with self._lock:
            if self._writer is not None:
                self._writer.write(frame)
            else:
                self._buf.append(frame.copy())

    def start(self, path: str, fourcc, fps: float, frame_size: tuple):
        with self._lock:
            self._writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
            for buffered in self._buf:
                self._writer.write(buffered)
            self._buf.clear()
            self.recording = True
            print(f"[RECORD] Clip started → {path}")

    def finish(self):
        with self._lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            self.recording = False
            print("[RECORD] Clip saved.")


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE HELPERS  (unchanged from cockroach_detector.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(frame_bgr: np.ndarray):
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_w, input_h))

    if input_dtype == np.uint8:
        input_data = resized[np.newaxis, ...]
    else:
        # float32 model — normalise to [0, 1]. If your export instead expects
        # [-1, 1], change this line to: (resized.astype(np.float32) - 127.5) / 127.5
        input_data = resized.astype(np.float32) / 255.0
        input_data = input_data[np.newaxis, ...]

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes_tensor = interpreter.get_tensor(IDX_BOXES)
    sc_tensor    = interpreter.get_tensor(IDX_SCORES_AND_CLASSES)

    if boxes_tensor.dtype == np.uint8:
        # Quantised INT8 model — dequantise using each tensor's scale/zero_point.
        bq = interpreter.get_output_details()[
            next(i for i, d in enumerate(interpreter.get_output_details())
                 if d['index'] == IDX_BOXES)
        ]['quantization']
        boxes_raw = (boxes_tensor[0].astype(np.float32) - bq[1]) * bq[0]

        scq = interpreter.get_output_details()[
            next(i for i, d in enumerate(interpreter.get_output_details())
                 if d['index'] == IDX_SCORES_AND_CLASSES)
        ]['quantization']
        sc_raw = (sc_tensor[0].astype(np.float32) - scq[1]) * scq[0]
    else:
        # Float32 model — outputs are already in their natural range, no
        # dequantisation needed.
        boxes_raw = boxes_tensor[0].astype(np.float32)
        sc_raw    = sc_tensor[0].astype(np.float32)

    scores  = sc_raw[:, 0]   # score in [0, 1]
    classes = sc_raw[:, 1]   # class id (round to int)

    h, w = frame_bgr.shape[:2]
    detections = []
    for i, score in enumerate(scores):
        if score < CONFIDENCE_THRESH:
            continue
        cls_id = int(round(classes[i])) + 1
        label  = LABELS.get(cls_id, f"class_{cls_id}")
        ymin, xmin, ymax, xmax = boxes_raw[i]
        detections.append({
            "label": label,
            "score": float(score),
            "box":   [int(xmin * w), int(ymin * h),
                      int(xmax * w), int(ymax * h)],
        })
    return detections


def in_middle_third(box) -> bool:
    xmin, _, xmax, _ = box
    cx = (xmin + xmax) / 2
    return ZONE_LEFT <= cx <= ZONE_RIGHT


def signal_detection_gpio(duration: float = 0.1):
    """Pulse the detection-signal GPIO pin (optional, from original script)."""
    if not USE_SIGNAL_GPIO or _signal_pin is None:
        return
    _signal_pin.on()
    time.sleep(duration)
    _signal_pin.off()


# ══════════════════════════════════════════════════════════════════════════════
#  SENSOR READINGS
# ══════════════════════════════════════════════════════════════════════════════

class MoistureSensor:
    """
    Reads the MCP3008 capacitive moisture sensor (SPI) and the SHT4x
    temperature/humidity sensor (I2C) on the same schedule, writing a
    single combined row to a daily CSV log file.

    CSV columns:
        timestamp, raw, voltage_V, moisture_pct, temp_C, humidity_pct
    Temperature and humidity columns are empty strings if the SHT4x is
    unavailable or disabled, so the CSV stays parseable either way.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # ── MCP3008 (SPI) ─────────────────────────────────────────────────────
        self._spi = None
        self._moisture_ok = SPIDEV_AVAILABLE and USE_MOISTURE
        if self._moisture_ok:
            try:
                self._spi = spidev.SpiDev()
                self._spi.open(0, 0)
                self._spi.max_speed_hz = 1_000_000
                self._spi.mode = 0
                print(f"[MOISTURE] MCP3008 ready on SPI0/CE0 "
                      f"(channel {MOISTURE_CHANNEL})")
            except Exception as exc:
                print(f"[MOISTURE] SPI init failed: {exc} — moisture logging disabled.")
                self._moisture_ok = False

        # ── SHT4x (I2C) ───────────────────────────────────────────────────────
        self._sht = None
        self._sht_ok = SHT4X_AVAILABLE and USE_SHT4X
        if self._sht_ok:
            try:
                i2c = board.I2C()
                self._sht = adafruit_sht4x.SHT4x(i2c)
                self._sht.mode = getattr(adafruit_sht4x.Mode, SHT4X_MODE)
                print(f"[SHT4X] Ready — serial {hex(self._sht.serial_number)}  "
                      f"mode={SHT4X_MODE}")
            except Exception as exc:
                print(f"[SHT4X] Init failed: {exc} — temp/humidity logging disabled.")
                self._sht_ok = False

    @property
    def available(self) -> bool:
        return self._moisture_ok or self._sht_ok

    # ── MCP3008 helpers ───────────────────────────────────────────────────────

    def _read_channel(self) -> int:
        cmd    = [1, (8 + MOISTURE_CHANNEL) << 4, 0]
        result = self._spi.xfer2(cmd)
        return ((result[1] & 3) << 8) | result[2]

    def _to_voltage(self, raw: int) -> float:
        return (raw / 1023.0) * MOISTURE_VREF

    def _to_percent(self, raw: int) -> float:
        pct = (MOISTURE_DRY - raw) / (MOISTURE_DRY - MOISTURE_WET) * 100
        return max(0.0, min(100.0, pct))

    # ── Log path ──────────────────────────────────────────────────────────────

    def _log_path(self) -> str:
        day = datetime.datetime.now().strftime("%Y-%m-%d")
        return os.path.join(MOISTURE_LOG_DIR, f"environment_{day}.csv")

    # ── Combined read + log ───────────────────────────────────────────────────

    def read_and_log(self):
        """
        Read both sensors and append one row to today's CSV. Thread-safe.
        Either sensor can be absent — its columns will be empty strings.
        """
        if not self.available:
            return

        with self._lock:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Moisture
            raw = voltage = pct = None
            if self._moisture_ok:
                try:
                    raw     = self._read_channel()
                    voltage = self._to_voltage(raw)
                    pct     = self._to_percent(raw)
                except Exception as exc:
                    print(f"[MOISTURE] Read error: {exc}")

            # Temperature + humidity
            temp = humidity = None
            if self._sht_ok:
                try:
                    temp, humidity = self._sht.measurements
                except Exception as exc:
                    print(f"[SHT4X] Read error: {exc}")

            # Format row — empty string for any missing value
            def fmt(v, spec):
                return format(v, spec) if v is not None else ""

            row = (
                f"{ts},"
                f"{fmt(raw, 'd')},"
                f"{fmt(voltage, '.3f')},"
                f"{fmt(pct, '.1f')},"
                f"{fmt(temp, '.2f')},"
                f"{fmt(humidity, '.2f')}\n"
            )

            path   = self._log_path()
            is_new = not os.path.exists(path)
            try:
                with open(path, "a") as f:
                    if is_new:
                        f.write("timestamp,raw,voltage_V,"
                                "moisture_pct,temp_C,humidity_pct\n")
                    f.write(row)
            except Exception as exc:
                print(f"[ENV LOG] Write error: {exc}")
                return

            # Console summary
            parts = [f"[ENV] {ts}"]
            if raw     is not None: parts.append(f"moisture={pct:.1f}% (raw={raw})")
            if temp    is not None: parts.append(f"temp={temp:.2f}°C")
            if humidity is not None: parts.append(f"humidity={humidity:.2f}%RH")
            parts.append(f"→ {os.path.basename(path)}")
            print("  ".join(parts))

    def close(self):
        if self._spi is not None:
            self._spi.close()
            print("[MOISTURE] SPI closed.")


def moisture_loop(sensor: MoistureSensor, stop_event: threading.Event):
    """
    Background thread: takes an immediate first reading on startup then waits
    MOISTURE_INTERVAL seconds between subsequent readings.
    Uses stop_event.wait() so shutdown is immediate rather than waiting for
    the next sleep to expire.
    """
    print(f"[ENV] Logging every {MOISTURE_INTERVAL // 60} min "
          f"to {MOISTURE_LOG_DIR}/environment_YYYY-MM-DD.csv")
    while not stop_event.is_set():
        sensor.read_and_log()
        stop_event.wait(timeout=MOISTURE_INTERVAL)
    print("[ENV] Loop stopped.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global interpreter, input_details, output_details
    global input_h, input_w, input_dtype

    os.makedirs(SAVEPATH, exist_ok=True)
    os.makedirs(MOISTURE_LOG_DIR, exist_ok=True)

    # ── Step 1: load model BEFORE camera to avoid ISP thread contention ───────
    print("[INFO] Loading model...", flush=True)
    interpreter = Interpreter(model_path=MODEL_PATH, num_threads=4)
    print("[INFO] Interpreter created", flush=True)
    interpreter.allocate_tensors()
    print("[INFO] Tensors allocated", flush=True)

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_h        = input_details[0]['shape'][1]
    input_w        = input_details[0]['shape'][2]
    input_dtype    = input_details[0]['dtype']

    print(f"[MODEL] Using hardcoded output tensors: boxes={IDX_BOXES}  "
          f"scores+classes={IDX_SCORES_AND_CLASSES}", flush=True)

    # ── Step 2: camera ────────────────────────────────────────────────────────
    print("[INFO] Starting camera...", flush=True)
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()
    print("[INFO] Camera warming up...", flush=True)
    time.sleep(2.5)
    print("[INFO] Camera ready", flush=True)

    # ── Step 3: servo, clip writer, moisture sensor ───────────────────────────
    servo      = DoorServo()
    clip       = ClipWriter(buf_size=BUF_SIZE)
    fourcc     = cv2.VideoWriter_fourcc(*'MJPG')
    idle_frames = 0

    moisture_stop   = threading.Event()
    moisture_sensor = MoistureSensor()
    moisture_thread = threading.Thread(
        target=moisture_loop,
        args=(moisture_sensor, moisture_stop),
        daemon=True,
        name="moisture-logger",
    )
    moisture_thread.start()

    door_open = False
    servo.close_door()

    print("[INFO] Starting detection loop. Press 'q' to quit.", flush=True)
    mode_str = "NIGHT" if NIGHT_MODE else "DAY"
    print(f"[INFO] Operating mode: {mode_str}-active  "
          f"({ACTIVE_START_HOUR:02d}:00 – {ACTIVE_END_HOUR:02d}:00)", flush=True)

    try:
        while True:
            frame     = picam2.capture_array()
            frame     = imutils.resize(frame, width=FRAME_WIDTH)
            h, w      = frame.shape[:2]
            timestamp = datetime.datetime.now()

            # ── Middle-third zone overlay ─────────────────────────────────────
            zone_l = int(w / 3)
            zone_r = int(2 * w / 3)
            cv2.rectangle(frame, (zone_l, 0), (zone_r, h), (255, 180, 0), 1)

            # ── TFLite inference ──────────────────────────────────────────────  
            detections = run_inference(frame)

            if detections:
                print(f"[DEBUG] {len(detections)} raw detections, "
                    f"top score: {max(d['score'] for d in detections):.3f}", flush=True)

            trigger_detections = [
                d for d in detections
                if d["label"] == TARGET_LABEL
            ]
            cockroach_in_zone = len(trigger_detections) > 0

            # ── Remote-Pi poll ────────────────────────────────────────────────
            remote_open_signal = remote_pi_requesting_open()

            # ── Day/night schedule check ──────────────────────────────────────
            active_now = in_active_period()

            # ── Door state machine ────────────────────────────────────────────
            # OPEN:  active period AND (cockroach OR remote signal)
            #        OR outside active period AND cockroach (entry always allowed)
            # CLOSE: no trigger present
            should_open = (
                (active_now and (cockroach_in_zone or remote_open_signal))
                or (not active_now and cockroach_in_zone)
            )

            if should_open and not door_open:
                reason = []
                if cockroach_in_zone:  reason.append("cockroach detected")
                if remote_open_signal: reason.append("remote Pi signal")
                print(f"[DOOR ] OPEN  ← {', '.join(reason)}  "
                      f"| {timestamp.strftime('%H:%M:%S')}", flush=True)
                threading.Thread(target=servo.open_door, daemon=True).start()
                door_open = True

                if not clip.recording:
                    clip_name = (f"{SAVEPATH}/"
                                 f"{timestamp.strftime('%Y%m%d-%H%M%S')}_door.avi")
                    clip.start(clip_name, fourcc, FPS,
                               (frame.shape[1], frame.shape[0]))

            elif not should_open and door_open:
                reason = "schedule ended" if not active_now else "no trigger"
                print(f"[DOOR ] CLOSE ← {reason}  "
                      f"| {timestamp.strftime('%H:%M:%S')}", flush=True)
                threading.Thread(target=servo.close_door, daemon=True).start()
                door_open = False

            # ── Detection signal & clip timeout ───────────────────────────────
            if cockroach_in_zone:
                idle_frames = 0
                print(f"[SIGNAL] COCKROACH DETECTED  "
                      f"| {timestamp.strftime('%H:%M:%S')}", flush=True)
                threading.Thread(target=signal_detection_gpio, daemon=True).start()
            else:
                idle_frames += 1
                if clip.recording and idle_frames >= IDLE_TIMEOUT_FRAMES:
                    clip.finish()

            clip.update(frame)

            # ── Draw detections ───────────────────────────────────────────────
            for d in detections:
                xmin, ymin, xmax, ymax = d["box"]
                in_zone = d["label"] == TARGET_LABEL and in_middle_third(d["box"])
                colour  = (0, 255, 0) if in_zone else (0, 165, 255)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), colour, 2)
                cv2.putText(
                    frame,
                    f"{d['label']} {d['score']:.2f}",
                    (xmin, max(ymin - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1,
                )

            # ── OSD overlay ───────────────────────────────────────────────────
            ts_str = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
            cv2.putText(frame, ts_str, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            door_label  = "DOOR: OPEN" if door_open else "DOOR: CLOSED"
            door_colour = (0, 255, 0)  if door_open else (0, 0, 255)
            cv2.putText(frame, door_label, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, door_colour, 2)

            period_label  = "ACTIVE" if active_now else "INACTIVE"
            period_colour = (0, 255, 200) if active_now else (80, 80, 80)
            cv2.putText(frame, period_label, (10, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, period_colour, 1)

            if clip.recording:
                cv2.circle(frame, (w - 18, 18), 8, (0, 0, 255), -1)

            if SHOW_VIDEO:
                cv2.imshow("Cockroach Door Controller", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        if clip.recording:
            clip.finish()
        if door_open:
            servo.close_door()
        servo.shutdown()
        moisture_stop.set()
        moisture_thread.join(timeout=5)
        moisture_sensor.close()
        picam2.stop()
        cv2.destroyAllWindows()
        if _remote_pin is not None:
            _remote_pin.close()
        if _signal_pin is not None:
            _signal_pin.close()
        print("[INFO] Shutdown complete.")


if __name__ == "__main__":
    main()