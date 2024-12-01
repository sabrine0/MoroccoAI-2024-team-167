import torch  
import cv2  
from gtts import gTTS  
import os  

# Load YOLO model  
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  

# Open webcam  
cap = cv2.VideoCapture(0)  
while True:  
    ret, frame = cap.read()  
    if not ret:  
        break  

    # Inference  
    results = model(frame)  
    detections = results.pandas().xyxy[0]  # Bounding boxes  

    # Draw boxes  
    for _, row in detections.iterrows():  
        cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])),  
                      (int(row['xmax']), int(row['ymax'])), (0, 255, 0), 2)  
        cv2.putText(frame, row['name'],  
                    (int(row['xmin']), int(row['ymin']) - 10),  
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  

    cv2.imshow('Object Detection', frame)  

    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  

cap.release()  
cv2.destroyAllWindows()  

def speak(text):  
    tts = gTTS(text, lang='en')  
    tts.save("output.mp3")  
    os.system("start output.mp3")  # Use 'xdg-open' on Linux  
    
# Example usage  
speak("Person detected in front of you")  
