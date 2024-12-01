import cv2
import torch
import pyttsx3
import pandas as pd

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Set up the camera for real-time video capture
cap = cv2.VideoCapture(0)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Confidence threshold for object detection
confidence_threshold = 0.5

# Function to provide audio feedback
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Loop for real-time object recognition and audio feedback
while True:
    # Capture each frame from the camera
    ret, frame = cap.read()
    
    # If the frame was not captured correctly, skip this iteration
    if not ret:
        continue

    # Perform object detection
    results = model(frame)

    # Extract the detected objects as a pandas DataFrame
    detections = results.pandas().xywh[0]

    # Print column names to inspect the DataFrame
    print(detections.columns)
    print(detections.head())

    # Loop over each detection
    for _, detection in detections.iterrows():
        # Confidence and class name for detection
        confidence = detection['confidence']
        class_name = detection['name']

        # Only process detections with confidence above the threshold
        if confidence > confidence_threshold:
            # Calculate bounding box coordinates from center, width, and height
            xcenter, ycenter, width, height = detection['xcenter'], detection['ycenter'], detection['width'], detection['height']
            
            # Convert to corner coordinates
            xmin = int(xcenter - width / 2)
            ymin = int(ycenter - height / 2)
            xmax = int(xcenter + width / 2)
            ymax = int(ycenter + height / 2)

            # Draw bounding box on the frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            
            # Add text label for the detected object
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Provide audio feedback about the detected object
            speak(f"Detected {class_name} at {round(confidence * 100, 2)} percent")

    # Show the frame with detections
    cv2.imshow('Object Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()
