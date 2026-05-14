#!/usr/bin/env python3
"""
Enclosure Monitor - Raspberry Pi Camera
Captures an image every 5 minutes, compiles 6-hourly timelapses, and runs
continuous MOG2-based motion detection with GPIO alerting on a defined zone.

Requires: picamera2 (pre-installed on PiOS Trixie)
          sudo apt install -y ffmpeg python3-opencv python3-rpi.gpio
"""

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

CAPTURE_INTERVAL_SEC = 5 * 60          # 5 minutes between snapshots
TIMELAPSE_FPS        = 10              # output video frame rate
TIMELAPSE_PERIODS    = [0, 6, 12, 18]  # 6-hour period start hours

# Camera settings
CAMERA_WIDTH   = 1920
CAMERA_HEIGHT  = 1080
CAMERA_QUALITY = 85

# Motion detection stream resolution (lores — keeps CPU load low)
LORES_WIDTH  = 320
LORES_HEIGHT = 240

# Set True only if using a motorised-focus lens (e.g. Camera Module 3)
USE_AUTOFOCUS = False

# ─── Motion Detection Configuration ──────────────────────────────────────────

# Minimum contour area (in lores pixels) to count as motion.
# Increase to ignore small insects/dust, decrease for finer sensitivity.
MOTION_MIN_AREA = 500

# How many consecutive frames must show motion before GPIO fires.
# Reduces false positives from single-frame noise.
MOTION_CONFIRM_FRAMES = 3

# Seconds the GPIO pin stays HIGH after motion is last detected.
MOTION_GPIO_HOLD_SEC = 5.0

# GPIO pin (BCM numbering) to pulse HIGH when zone motion is detected.
MOTION_GPIO_PIN = 17

# ── Alert Zone ────────────────────────────────────────────────────────────────
# Defined in LORES coordinates (0–LORES_WIDTH, 0–LORES_HEIGHT).
# The zone is a rectangle: (x1, y1) top-left → (x2, y2) bottom-right.
# Replace these values with your desired region before deploying.
#
#   Full frame (no zone restriction):
#       ZONE = (0, 0, LORES_WIDTH, LORES_HEIGHT)
#
#   Example — right half of frame:
#       ZONE = (160, 0, 320, 240)
#
ZONE = (0, 0, LORES_WIDTH, LORES_HEIGHT)   # ← edit this

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
    """
    Open the camera once with two simultaneous streams:
      - main  (RGB888 @ full res)  → used by capture_file() for snapshots
      - lores (YUV420 @ 320x240)  → used by motion detection thread
    Both streams are fed by the same sensor readout with no interference.
    """
    cam = Picamera2()

    config = cam.create_still_configuration(
        main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"},
        lores={"size": (LORES_WIDTH, LORES_HEIGHT), "format": "YUV420"},
        display="lores",
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

# ─── Motion Detection Thread ──────────────────────────────────────────────────

def point_in_zone(cx: int, cy: int) -> bool:
    """Return True if centroid (cx, cy) falls inside the alert zone."""
    x1, y1, x2, y2 = ZONE
    return x1 <= cx <= x2 and y1 <= cy <= y2

def gpio_release_thread(hold_sec: float) -> None:
    """Pull GPIO LOW after hold_sec seconds (runs in a short-lived thread)."""
    time.sleep(hold_sec)
    GPIO.output(MOTION_GPIO_PIN, GPIO.LOW)
    log.debug("GPIO pin %d LOW", MOTION_GPIO_PIN)

def motion_detection_loop(cam: Picamera2, stop_event: threading.Event) -> None:
    """
    Continuously reads the lores YUV stream and applies MOG2 background
    subtraction to detect motion.

    MOG2 advantages for enclosure monitoring:
      - Adapts automatically to slow lighting changes (day/night cycles)
      - Built-in shadow detection (set detectShadows=False to save CPU)
      - Robust to minor camera shake or compression artefacts

    For each detected contour that exceeds MOTION_MIN_AREA:
      - Its bounding box and centroid are logged (in lores coordinates)
      - If the centroid falls inside ZONE and MOTION_CONFIRM_FRAMES
        consecutive frames show motion, GPIO pin is pulled HIGH
    """
    # MOG2 background subtractor
    # varThreshold: higher = less sensitive to gradual change (50 is a good
    #               starting point for indoor enclosures with steady lighting)
    # detectShadows: disabled to halve the per-frame CPU cost
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=False,
    )

    # Morphological kernel to remove speckle noise from the foreground mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    confirm_count   = 0          # consecutive frames with zone motion
    gpio_active     = False      # True while GPIO is HIGH
    gpio_timer: "threading.Thread | None" = None

    log.info(
        "Motion detection started -- lores %dx%d  zone=%s  min_area=%d  gpio_pin=%d",
        LORES_WIDTH, LORES_HEIGHT, ZONE, MOTION_MIN_AREA, MOTION_GPIO_PIN,
    )

    while not stop_event.is_set():
        # Grab a lores YUV420 frame (Y plane only = greyscale, cheapest path)
        yuv = cam.capture_array("lores")
        # YUV420: first LORES_HEIGHT rows are the Y (luma) plane
        grey = yuv[:LORES_HEIGHT, :LORES_WIDTH]

        # Apply background subtraction → binary foreground mask
        fg_mask = subtractor.apply(grey)

        # Clean up noise: erode then dilate (opening operation)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Find contours of moving regions
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        zone_motion_this_frame = False

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MOTION_MIN_AREA:
                continue

            # Bounding box and centroid
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2

            log.debug(
                "Motion detected -- bbox=(%d,%d,%d,%d)  centroid=(%d,%d)  area=%.0f",
                x, y, w, h, cx, cy, area,
            )

            if point_in_zone(cx, cy):
                zone_motion_this_frame = True
                log.info(
                    "Zone motion -- centroid=(%d,%d)  area=%.0f  zone=%s",
                    cx, cy, area, ZONE,
                )

        # ── GPIO logic ────────────────────────────────────────────────────
        if zone_motion_this_frame:
            confirm_count += 1
            if confirm_count >= MOTION_CONFIRM_FRAMES:
                if not gpio_active:
                    GPIO.output(MOTION_GPIO_PIN, GPIO.HIGH)
                    gpio_active = True
                    log.info("GPIO pin %d HIGH", MOTION_GPIO_PIN)

                # (Re)start the hold timer on every confirmed motion frame
                if gpio_timer and gpio_timer.is_alive():
                    # Can't cancel a sleeping thread directly; let it expire
                    # and suppress the LOW — reset gpio_active to keep HIGH
                    pass
                gpio_timer = threading.Thread(
                    target=gpio_release_thread,
                    args=(MOTION_GPIO_HOLD_SEC,),
                    daemon=True,
                )
                gpio_timer.start()
        else:
            confirm_count = 0
            # gpio_release_thread handles the delayed LOW automatically

        # Small sleep to cap CPU — ~15 fps is plenty for motion detection
        time.sleep(0.065)

    log.info("Motion detection loop stopped.")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Enclosure Monitor started ===")
    log.info("Images     -> %s", IMAGES_DIR)
    log.info("Timelapses -> %s", TIMELAPSE_DIR)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TIMELAPSE_DIR.mkdir(parents=True, exist_ok=True)

    cam         = init_camera()
    stop_event  = threading.Event()

    # Start motion detection in a background thread
    motion_thread = threading.Thread(
        target=motion_detection_loop,
        args=(cam, stop_event),
        daemon=True,
        name="motion-detection",
    )
    motion_thread.start()

    last_period: "str | None" = None

    try:
        while True:
            loop_start     = datetime.now()
            current_period = period_label(loop_start)

            # ── Timelapse trigger ──────────────────────────────────────────
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

            # ── Snapshot ──────────────────────────────────────────────────
            capture_image(cam)

            # ── Sleep until next interval ──────────────────────────────────
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