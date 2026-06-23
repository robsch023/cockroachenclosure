"""
Robert Schaefer 
10/04/2026

Script to record behavioural data of giant burrowing cockroach in container within lab environment for behavioural study and analysis. Activates camera to record while motion is detected within the container using weighted average background subtraction method. Saves video files with timestamp for later analysis. Press 'q' to stop recording.

Need to pip install picamera and imutils for image processing simplification.
"""

import cv2
from picamera import Picamera
from picamera.array import PiRGBArray
from experiments.keyclipwriter import KeyClipWriter
from imutils.video import VideoStream as vs
import datetime
import imutils
import time


# Configuration parameters
show_video = True
min_motion_frames = 8
camera_warmup_time = 2.5
delta_threshold = 5
resolution = (640, 480)
fps = 16
min_area = 5000
savepath = "/home/rosch023/local_mount/behaviouraldata"

# Initialize the Picamera2
picam2 = Picamera()
picam2.resolution = resolution
picam2.framerate = fps
rawCapture = PiRGBArray(picam2, size=resolution)

# allow the camera to warmup, initialise average frame counter, last uploaded timestamp, and frame motion counter
print("[INFO] Camera warming up...")
time.sleep(camera_warmup_time)
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0

kcw = KeyClipWriter(bufSize=32, timeout=1.0)
consecFrames = 0

# Capture frames from the camera
for f in picam2.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = f.array
    timestamp = datetime.datetime.now()
    text = "Unoccupied"
    frame = imutils.resize(frame, width=500)
    updateConsecFrames = True

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if avg is None:
        print("[INFO] Starting background model...")
        avg = gray.copy().astype("float")
        rawCapture.truncate(0)
        continue

    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    thresh = cv2.threshold(frameDelta, delta_threshold, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"  

    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame, f"Room Status: {text}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    if text == "Occupied":
        consecFrames = 0
        
        if not kcw.recording:
            timestamp = datetime.datetime.now()
            p = f"{savepath}/{timestamp.strftime('%Y%m%d-%H%M%S')}.avi"
            kcw.start(p, cv2.VideoWriter_fourcc(*'MJPG'), fps)

    if updateConsecFrames:
        consecFrames += 1

    kcw.update(frame)

    if kcw.recording and consecFrames == buffSize:
        kcw.finish()

    rawCapture.truncate(0)

    if key == ord("q"):
        break

if kcw.recording:
    kcw.finish()

cv2.destroyAllWindows()
vs.stop()