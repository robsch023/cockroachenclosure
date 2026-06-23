#!/usr/bin/env python3
"""
Enclosure Monitor - Raspberry Pi Camera
Captures an image every 5 minutes, compiles 6-hourly timelapses, and runs
continuous MOG2-based motion detection with GPIO alerting on a defined zone.

Requires: picamera2 (pre-installed on PiOS Trixie)
          sudo apt install -y ffmpeg python3-opencv python3-rpi.gpio

Debug preview: set DEBUG_PREVIEW = True and run from the desktop terminal.
               cv2.imshow renders via the wayland backend (set below).
               Overlays drawn on the live lores feed:
                 - GREEN rectangle  = alert zone
                 - ORANGE rectangle = motion contour inside zone
                 - RED rectangle    = motion contour outside zone
                 - YELLOW dot       = contour centroid
                 - Top-left HUD     = timestamp, contour count, confirm, GPIO
"""

import os

# Must be set before cv2 or any Qt library is imported.
# Tells OpenCV/Qt to use the Wayland backend (PiOS Trixie default).
os.environ.setdefault("QT_QPA_PLATFORM", "wayland")

import time
import threading
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from libcamera import controls

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR      = Path("/home/rosch023/local_mount/labtestdata/oversightcam")
IMAGES_DIR    = BASE_DIR / "images"
TIMELAPSE_DIR = BASE_DIR / "timelapses"
LOG_FILE      = Path("/home/rosch023/enclosure/monitor.log")

CAPTURE_INTERVAL_SEC = 5 * 60
TIMELAPSE_FPS        = 10
TIMELAPSE_PERIODS    = [0, 6, 12, 18]

CAMERA_WIDTH   = 1920
CAMERA_HEIGHT  = 1080
CAMERA_QUALITY = 85

LORES_WIDTH  = 320
LORES_HEIGHT = 240

USE_AUTOFOCUS = True

# ─── Motion Detection Configuration ──────────────────────────────────────────

MOTION_MIN_AREA       = 100
MOTION_CONFIRM_FRAMES = 2
MOTION_GPIO_HOLD_SEC  = 8.0
MOTION_DEAD_SEC       = 1.0
MOTION_GPIO_PIN       = 17

ZONE = (80, 85, 120, 165)   # (x1, y1, x2, y2) in lores coordinates

# ─── Debug Preview ────────────────────────────────────────────────────────────

# Set True to open a cv2.imshow window. Run from the desktop terminal.
# Set False for headless/service deployment.
DEBUG_PREVIEW = False

# Preview display scale — lores is 320x240, scale 3 = 960x720 on screen
PREVIEW_SCALE = 3

# ─── Logging ──────────────────────────────────────────────────────────────────

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─── GPIO setup ───────────────────────────────────────────────────────────────

GPIO.setmode(GPIO.BCM)
GPIO.setup(MOTION_GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)

# ─── Period helpers ───────────────────────────────────────────────────────────

def period_label(dt: datetime) -> str:
    for i, start in enumerate(TIMELAPSE_PERIODS):
        end = TIMELAPSE_PERIODS[(i + 1) % len(TIMELAPSE_PERIODS)]
        if start < end:
            if start <= dt.hour < end:
                return f"{start:02d}-{end:02d}"
        else:
            if dt.hour >= start or dt.hour < end:
                return f"{start:02d}-{end:02d}"
    return "00-06"

def image_dir_for(dt: datetime) -> Path:
    return IMAGES_DIR / dt.strftime("%Y-%m-%d") / period_label(dt)

# ─── Camera initialisation ────────────────────────────────────────────────────

def init_camera() -> Picamera2:
    cam = Picamera2()
    config = cam.create_still_configuration(
        main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"},
        lores={"size": (LORES_WIDTH, LORES_HEIGHT), "format": "YUV420"},
        display=None,
        buffer_count=4,
    )
    cam.configure(config)
    cam.options["quality"] = CAMERA_QUALITY

    cam_controls = {"AeEnable": True, "AwbEnable": True}
    if USE_AUTOFOCUS:
        cam_controls["AfMode"] = controls.AfModeEnum.Continuous
    cam.set_controls(cam_controls)

    cam.start()
    time.sleep(2)   # ISP settling time
    log.info(
        "Camera initialised -- %dx%d  quality=%d  autofocus=%s",
        CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_QUALITY, USE_AUTOFOCUS,
    )
    return cam

# ─── Capture (snapshot) ───────────────────────────────────────────────────────

def capture_image(cam: Picamera2) -> "Path | None":
    now      = datetime.now()
    dest     = image_dir_for(now)
    dest.mkdir(parents=True, exist_ok=True)
    filepath = dest / now.strftime("%Y%m%d_%H%M%S.jpg")
    try:
        cam.capture_file(str(filepath))
        log.info("Captured: %s", filepath)
        return filepath
    except Exception as exc:
        log.error("Capture failed: %s", exc)
    return None

# ─── Timelapse ────────────────────────────────────────────────────────────────

def build_timelapse(day: str, period: str) -> "Path | None":
    src_dir = IMAGES_DIR / day / period
    images  = sorted(src_dir.glob("*.jpg"))
    if not images:
        log.warning("No images for %s/%s -- skipping timelapse.", day, period)
        return None

    TIMELAPSE_DIR.mkdir(parents=True, exist_ok=True)
    out_file  = TIMELAPSE_DIR / f"{day}_{period}.mp4"
    list_file = src_dir / "filelist.txt"

    with open(list_file, "w") as f:
        for img in images:
            f.write(f"file '{img}'\n")
            f.write(f"duration {1 / TIMELAPSE_FPS}\n")
        f.write(f"file '{images[-1]}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-vf", f"scale={CAMERA_WIDTH}:{CAMERA_HEIGHT}:flags=lanczos",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(out_file),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        log.info("Timelapse saved: %s  (%d frames)", out_file, len(images))
        list_file.unlink(missing_ok=True)
        return out_file
    except subprocess.CalledProcessError as exc:
        log.error("ffmpeg failed:\n%s", exc.stderr[-500:])
    except subprocess.TimeoutExpired:
        log.error("ffmpeg timed out.")
    return None

# ─── Motion Detection + Preview Thread ───────────────────────────────────────

def point_in_zone(cx: int, cy: int) -> bool:
    x1, y1, x2, y2 = ZONE
    return x1 <= cx <= x2 and y1 <= cy <= y2

def gpio_release_thread(hold_sec: float, state: dict) -> None:
    time.sleep(hold_sec)
    GPIO.output(MOTION_GPIO_PIN, GPIO.LOW)
    state["gpio_low_time"] = time.monotonic()
    log.debug("GPIO pin %d LOW -- dead period starts (%.1fs)", MOTION_GPIO_PIN, MOTION_DEAD_SEC)

def draw_debug_frame(grey: np.ndarray, contours, confirm_count: int,
                     gpio_active: bool, in_dead_period: bool = False) -> np.ndarray:
    """
    Build the debug display frame from the greyscale lores image.
    Returns a scaled BGR image with all overlays drawn on it.
    """
    s = PREVIEW_SCALE
    display = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    display = cv2.resize(display, (LORES_WIDTH * s, LORES_HEIGHT * s),
                         interpolation=cv2.INTER_LINEAR)

    # ── Zone rectangle ────────────────────────────────────────────────────
    x1, y1, x2, y2 = ZONE
    zone_colour = (0, 255, 128) if gpio_active else (0, 220, 0)
    cv2.rectangle(display, (x1*s, y1*s), (x2*s, y2*s), zone_colour, 2)
    cv2.putText(display, "ZONE", (x1*s, max(0, y1*s - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_colour, 1)

    # ── Motion contours ───────────────────────────────────────────────────
    valid = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MOTION_MIN_AREA:
            continue
        valid += 1
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        cy = y + h // 2
        in_zone    = point_in_zone(cx, cy)
        box_colour = (0, 140, 255) if in_zone else (0, 0, 200)

        cv2.rectangle(display, (x*s, y*s), ((x+w)*s, (y+h)*s), box_colour, 1)
        cv2.circle(display, (cx*s, cy*s), 4, (0, 255, 255), -1)
        cv2.putText(display, f"{int(area)}", (x*s, max(0, y*s - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, box_colour, 1)

    # ── HUD ───────────────────────────────────────────────────────────────
    gpio_str    = "GPIO: HIGH" if gpio_active else "GPIO: low"
    gpio_colour = (0, 255, 80) if gpio_active else (160, 160, 160)
    hud = [
        (datetime.now().strftime("%H:%M:%S"),                   (210, 210, 210)),
        (f"Contours : {valid}",                                 (210, 210, 210)),
        (f"Confirm  : {confirm_count}/{MOTION_CONFIRM_FRAMES}", (210, 210, 210)),
        (gpio_str,                                              gpio_colour),
        (f"MinArea  : {MOTION_MIN_AREA}",                       (150, 150, 150)),
        (f"Zone     : {ZONE}",                                  (150, 150, 150)),
        ("-- DEAD PERIOD --" if in_dead_period else "",         (0, 180, 255)),
    ]
    for i, (text, colour) in enumerate(hud):
        cv2.putText(display, text, (8, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    return display


def motion_detection_loop(cam: Picamera2, stop_event: threading.Event) -> None:
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=False,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    confirm_count = 0
    gpio_active   = False
    gpio_timer: "threading.Thread | None" = None
    # Shared mutable state passed into gpio_release_thread so it can record
    # when GPIO went LOW without needing a global variable.
    gpio_state = {"gpio_low_time": 0.0}

    log.info(
        "Motion detection started -- lores %dx%d  zone=%s  min_area=%d  "
        "gpio_pin=%d  preview=%s",
        LORES_WIDTH, LORES_HEIGHT, ZONE, MOTION_MIN_AREA,
        MOTION_GPIO_PIN, DEBUG_PREVIEW,
    )

    while not stop_event.is_set():
        # Grab lores Y plane (greyscale) for motion detection
        yuv  = cam.capture_array("lores")
        grey = yuv[:LORES_HEIGHT, :LORES_WIDTH]

        fg_mask = subtractor.apply(grey)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        zone_motion_this_frame = False

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MOTION_MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2

            log.debug(
                "Motion -- bbox=(%d,%d,%d,%d)  centroid=(%d,%d)  area=%.0f",
                x, y, w, h, cx, cy, area,
            )

            if point_in_zone(cx, cy):
                zone_motion_this_frame = True
                log.info("Zone motion -- centroid=(%d,%d)  area=%.0f", cx, cy, area)

        # ── GPIO logic ────────────────────────────────────────────────────
        # Suppress new triggers during the dead period after GPIO goes LOW
        in_dead_period = (
            not gpio_active
            and (time.monotonic() - gpio_state["gpio_low_time"]) < MOTION_DEAD_SEC
        )
        if in_dead_period and zone_motion_this_frame:
            log.debug("Motion suppressed -- dead period active")
            zone_motion_this_frame = False
            confirm_count = 0

        if zone_motion_this_frame:
            confirm_count += 1
            if confirm_count >= MOTION_CONFIRM_FRAMES:
                if not gpio_active:
                    GPIO.output(MOTION_GPIO_PIN, GPIO.HIGH)
                    gpio_active = True
                    log.info("GPIO pin %d HIGH", MOTION_GPIO_PIN)
                gpio_timer = threading.Thread(
                    target=gpio_release_thread,
                    args=(MOTION_GPIO_HOLD_SEC, gpio_state),
                    daemon=True,
                )
                gpio_timer.start()
        else:
            confirm_count = 0
            if gpio_active and (gpio_timer is None or not gpio_timer.is_alive()):
                gpio_active = False

        # ── Debug preview ─────────────────────────────────────────────────
        if DEBUG_PREVIEW:
            frame = draw_debug_frame(grey, contours, confirm_count, gpio_active, in_dead_period)
            cv2.imshow("Enclosure Monitor", frame)
            # q to quit the preview (stops the whole script)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

        time.sleep(0.065)

    if DEBUG_PREVIEW:
        cv2.destroyAllWindows()
    log.info("Motion detection loop stopped.")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Enclosure Monitor started ===")
    log.info("Images     -> %s", IMAGES_DIR)
    log.info("Timelapses -> %s", TIMELAPSE_DIR)
    log.info("Debug preview: %s", DEBUG_PREVIEW)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TIMELAPSE_DIR.mkdir(parents=True, exist_ok=True)

    cam        = init_camera()
    stop_event = threading.Event()

    # Motion detection (and preview) runs in a background thread
    motion_thread = threading.Thread(
        target=motion_detection_loop,
        args=(cam, stop_event),
        daemon=True,
        name="motion-detection",
    )
    motion_thread.start()

    last_period: "str | None" = None

    try:
        while not stop_event.is_set():
            loop_start     = datetime.now()
            current_period = period_label(loop_start)

            if last_period is None:
                last_period = current_period

            if current_period != last_period:
                prev_moment = loop_start - timedelta(seconds=1)
                prev_day    = prev_moment.strftime("%Y-%m-%d")
                log.info(
                    "Period changed -> compiling timelapse %s/%s",
                    prev_day, last_period,
                )
                build_timelapse(prev_day, last_period)
                last_period = current_period

            capture_image(cam)

            elapsed   = (datetime.now() - loop_start).total_seconds()
            sleep_for = max(0, CAPTURE_INTERVAL_SEC - elapsed)
            log.debug("Next capture in %.1fs", sleep_for)
            time.sleep(sleep_for)

    finally:
        stop_event.set()
        motion_thread.join(timeout=2)
        cam.stop()
        cam.close()
        GPIO.cleanup()
        log.info("Shutdown complete.")


if __name__ == "__main__":
    main()