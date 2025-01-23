import os
import cv2
from ultralytics import YOLO
import easyocr
from PIL import Image
import numpy as np
import datetime as dt

def save_text_in_file(text):
    filename = "ocr_on_video.txt"
    current_date_and_time = dt.datetime.now()

    # Validar si el archivo existe
    if os.path.exists(filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not

    # Guardar texto en archivo plano
    file = open(filename, append_write)
    file.write(f"{current_date_and_time}: {text}" + "\n")
    file.close()

# Initialize EasyOCR reader
reader = easyocr.Reader(['es'], gpu=False)

# Load your YOLO model
model = YOLO('./Models/yolo11n.pt', task='detect')

# Open the video file
cap = cv2.VideoCapture("./Data/Video_1.mp4")

# Frame skipping factor
frame_skip = 3  # Skip every 3rd frame
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break  # Exit loop if there are no frames left

    # Skip frames
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue  # Skip processing this frame

    # Resize the frame
    frame = cv2.resize(frame, (640, 480))  # Resize to 640x480

    # Make predictions on the current frame
    results = model.predict(source=frame)

    # Iterate over results and draw predictions
    for result in results:
        boxes = result.boxes  # Get the boxes predicted by the model
        
        for box in boxes:
            class_id = int(box.cls)  # Get the class ID
            confidence = box.conf.item()  # Get confidence score
            coordinates = box.xyxy[0]  # Get box coordinates as a tensor

            # Extract and convert box coordinates to integers
            x1, y1, x2, y2 = map(int, coordinates.tolist())  # Convert tensor to list and then to int

            # Draw the box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
            
            # Try to apply OCR on detected region
            try:
                # Ensure coordinates are within frame bounds
                r0 = max(0, x1)
                r1 = max(0, y1)
                r2 = min(frame.shape[1], x2)
                r3 = min(frame.shape[0], y2)

                # Crop region
                region = frame[r1:r3, r0:r2]

                # Convert to format compatible with EasyOCR
                image = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
                array = np.array(image)

                # Use EasyOCR to read text
                text = reader.readtext(array)
                text_concat = ' '.join([number[1] for number in text])

                # Llamar función para guardar texto en archivo plano
                save_text_in_file(text_concat)

            except Exception as e:
                print(f"OCR Error: {e}")
                pass

    # Show the frame with detections
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit loop if 'q' is pressed

    frame_count += 1  # Increment frame count
    
# Release resources
cap.release()
cv2.destroyAllWindows()