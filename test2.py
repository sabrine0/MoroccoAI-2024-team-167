import torch
import cv2
import pyttsx3
import time

# Initialize text-to-speech
engine = pyttsx3.init()

# Function to provide audio feedback
def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error in TTS: {e}")

# Function to estimate distance (requires calibration)
def estimate_distance(height_in_pixels, known_height=170, focal_length=500):
    
    if height_in_pixels > 0:
        distance = (known_height * focal_length) / height_in_pixels
        return round(distance, 2)  # Return distance in the same unit as known_height
    return None

# Load the YOLOv8 model from Ultralytics
print("Loading YOLOv8 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov8s.pt', trust_repo=True)

# Open the webcam
print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("Press 'Q' to quit.")

# Dictionary to track detected objects and their timestamps
last_detected = {}

# Process frames from the webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Perform object detection
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Extract detection details

    # Draw bounding boxes and labels on the frame
    for _, row in detections.iterrows():
        x_min, y_min, x_max, y_max, label = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['name']
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Estimate distance
        object_height_pixels = y_max - y_min
        distance = estimate_distance(object_height_pixels)

        # Check if the object was recently announced
        current_time = time.time()
        if label not in last_detected or (current_time - last_detected[label] > 5):  # 5 seconds cooldown
            if distance:
                description = f"I see a {label} approximately {distance} centimeters away."
            else:
                description = f"I see a {label}."
            print(description)  # Log description
            speak(description)  # Speak description
            last_detected[label] = current_time  # Update timestamp

    # Show the frame in a window
    cv2.imshow('Real-Time Object Detection', frame)

    # Break the loop if 'Q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Program ended.")
