from ultralytics import YOLO
import cvzone
import cv2
import math

# Running real time from webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam, change to the appropriate index if you have multiple cameras

model_human = YOLO('detection.pt')
model_fire = YOLO('fire.pt')

# Reading the classes
classnames_human = ['person']
classnames_fire = ['fire']

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    result_human = model_human(frame, stream=True)
    result_fire = model_fire(frame, stream=True)

    
    for info in result_human:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if Class < len(classnames_human):  
                if confidence > 50:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cvzone.putTextRect(frame, f'{classnames_human[Class]} {confidence}%', (x1 + 8, y1 + 100),
                                       scale=1.5, thickness=2)

    
    for info in result_fire:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if Class < len(classnames_fire):  
                if confidence > 50:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cvzone.putTextRect(frame, f'{classnames_fire[Class]} {confidence}%', (x1 + 8, y1 + 100),
                                       scale=1.5, thickness=2)

    cv2.imshow('Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
