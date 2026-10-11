import cv2
import numpy as np
import time
import threading
import os
import pygame

# Initialize pygame mixer
pygame.mixer.init()
alarm_sound = "alarm-danger-danger.mp3"  # Ensure this file exists
alarm_channel = None

# Load DNN face detection model
net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',
    'res10_300x300_ssd_iter_140000.caffemodel'
)

# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Parameters
closed_frames = 0
threshold = 15
drowsy = False
frame_count = 0
face_box = None
alarm_on = False
snapshot_taken = False

# Create folder if not exists
os.makedirs("snapshots", exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]

    # Run face detection every 5 frames
    if frame_count % 5 == 0:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        face_box = None
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                face_box = box.astype("int")
                break

    eyes_detected = False
    if face_box is not None:
        (startX, startY, endX, endY) = face_box
        cv2.rectangle(display_frame, (startX, startY), (endX, endY), (0, 255, 255), 2)

        roi_gray = gray[startY:endY, startX:endX]
        roi_color = display_frame[startY:endY, startX:endX]
        roi_gray = cv2.equalizeHist(roi_gray)

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(15, 15),
            maxSize=(80, 80)
        )

        # Filter eyes only from upper half of face ROI
        roi_h = roi_gray.shape[0]
        filtered_eyes = []
        for (ex, ey, ew, eh) in eyes:
            if ey + eh / 2 < roi_h * 0.6:
                filtered_eyes.append((ex, ey, ew, eh))
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        if len(filtered_eyes) >= 1:
            eyes_detected = True
            closed_frames = 0
            drowsy = False
            snapshot_taken = False

    if not eyes_detected:
        closed_frames += 1
        if closed_frames >= threshold:
            drowsy = True
    else:
        closed_frames = 0

    # Alarm control
    if drowsy and not alarm_on:
        pygame.mixer.music.load(alarm_sound)
        pygame.mixer.music.play(-1)  # Loop until stopped
        alarm_on = True

    if not drowsy and alarm_on:
        pygame.mixer.music.stop()
        alarm_on = False
        snapshot_taken = False

    # UI and alert
    if drowsy:
        cv2.putText(display_frame, "DROWSINESS ALERT!", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        if int(time.time() * 2) % 2 == 0:
            cv2.rectangle(display_frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)

        if not snapshot_taken:
            filename = f"snapshots/drowsy_{int(time.time())}.jpg"
            cv2.imwrite(filename, display_frame)
            snapshot_taken = True

    # Eye status display
    eye_status = "EYES: CLOSED" if drowsy else "EYES: OPEN"
    color = (0, 0, 255) if drowsy else (0, 255, 0)
    cv2.putText(display_frame, eye_status, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("DDD - Driver Drowsiness Detection", display_frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
pygame.mixer.music.stop()
cap.release()
cv2.destroyAllWindows()