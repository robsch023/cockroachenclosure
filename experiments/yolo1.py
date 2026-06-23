"""
Robert Schaefer 
08/04/2026

Script to operate YOLO26 model on Raspberry Pi using Picamera2. This script captures video frames from the camera, performs object detection using the YOLO26 nano model, and displays the annotated video feed in real-time. Press 'q' to exit the video feed.

Function data_capture causes the camera to save the frame if the insect is identified as crossing centre of frame using bounding box estimation -- may need to upgrade to active tracking library but this is nice for now as is lightweight. 

Requires ultralytics library and as specified runs the YOLO26 nano model in NCNN format. Ensure export occurs to correct path and pytorch version is correct for export process.
"""

import cv2
from picamera import PiCamera
import time

# Load YOLOv5 nano model
from ultralytics import YOLO

def data_capture(frame, results, tolerance=50):
    for det in results:
        x1, y1, x2, y2 = det.box
        cls = det.class_id
        conf = det.confidence

        # Determine centre of the bounding box
        bbox_centre_x = (x1 + x2) / 2
        bbox_centre_y = (y1 + y2) / 2

        # Determine center of the frame
        frame_centre_x = frame.shape[1] / 2
        frame_centre_y = frame.shape[0] / 2

        path = "/home/rosch023/Images/gbc_training_images"

        # Add tolerance to the centre condition
        x_min = frame_centre_x - tolerance
        x_max = frame_centre_x + tolerance
        y_min = frame_centre_y - tolerance
        y_max = frame_centre_y + tolerance

        # If object is detected within the center area and confidence is above threshold, save the frame
        if x_min < bbox_centre_x < x_max and y_min < bbox_centre_y < y_max and conf > 0.8:    # Adjust confidence threshold as needed
            # Save the frame with a timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(f"{path}/gbc_{timestamp}.jpg", frame)
            print(f"Frame saved: gbc_{timestamp}.jpg")

# Initialize the Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

 # Load the YOLO26 nano model
model = YOLO('/.yolo26n_ncnn_model') 

# Capture frames from the camera
while True:
    frame = picam2.capture_array()
    
    # Perform inference
    results = model(frame)

    # Run data capture function
    data_capture(frame, results)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    inference_time = results[0].speed['inference']
    fps = 1000/inference_time
    text = f'FPS:{fps:.2f}'

    # Configure FPS text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_y = annotated_frame.shape[1] - text_size[0] - 10
    text_x = text_size[1] + 10
    cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("Camera", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()