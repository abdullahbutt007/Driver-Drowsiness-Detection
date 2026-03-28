import cv2
import time
import pygame
import pickle
import numpy as np
import mediapipe as mp
from math import hypot, degrees, atan2
from xgboost import XGBClassifier

# === Load Trained Model and Scaler ===
with open("best_drowsiness_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# === Initialize MediaPipe FaceMesh ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# === Pygame alert setup ===
pygame.mixer.init()
pygame.mixer.music.load("alert.wav")

# === Feature Calculation Functions ===
def euclidean(p1, p2):
    return hypot(p2[0] - p1[0], p2[1] - p1[1])

def get_EAR(landmarks, eye_indices):
    A = euclidean(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    B = euclidean(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    C = euclidean(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    return (A + B) / (2.0 * C)

def get_MAR(landmarks):
    top = landmarks[13]
    bottom = landmarks[14]
    left = landmarks[78]
    right = landmarks[308]
    return euclidean(top, bottom) / euclidean(left, right)

def get_head_tilt(landmarks):
    nose = landmarks[1]
    chin = landmarks[152]
    dx = chin[0] - nose[0]
    dy = chin[1] - nose[1]
    angle = abs(degrees(atan2(dy, dx)) - 90)
    return angle

# === Constants and Thresholds ===
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MAR_THRESHOLD = 0.75
EAR_THRESHOLD = 0.28
TILT_THRESHOLD = 25
EYE_CLOSED_DURATION = 2  # seconds

# === Start Webcam ===
cap = cv2.VideoCapture(0)
eye_closed_start = None
status = "Alert"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = [(int(p.x * w), int(p.y * h)) for p in results.multi_face_landmarks[0].landmark]

        left_ear = get_EAR(landmarks, LEFT_EYE)
        right_ear = get_EAR(landmarks, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        mar = get_MAR(landmarks)
        tilt = get_head_tilt(landmarks)

        is_eye_closed = ear < EAR_THRESHOLD
        is_mouth_open = mar > MAR_THRESHOLD
        is_head_tilted = tilt > TILT_THRESHOLD

        # === Eye Closure Timer Logic ===
        if is_eye_closed:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            eye_closed_duration = time.time() - eye_closed_start
        else:
            eye_closed_duration = 0
            eye_closed_start = None

        # === Feature Vector (for ML Model) ===
        features = np.array([[left_ear, right_ear, ear, mar, tilt, int(is_eye_closed)]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        # === Drowsiness Classification Logic ===
        if eye_closed_duration >= EYE_CLOSED_DURATION:
            status = "Drowsy (Eyes Closed)"
        elif is_mouth_open:
            status = "Drowsy (Yawning)"
        elif is_eye_closed and is_head_tilted:
            status = "Drowsy (Head Tilt)"
        else:
            status = "Alert"

        # === Sound Alert ===
        if "Drowsy" in status:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1)
        else:
            pygame.mixer.music.stop()

        # === Display Status ===
        color = (0, 0, 255) if "Drowsy" in status else (0, 255, 0)
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Tilt: {tilt:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, status, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    else:
        pygame.mixer.music.stop()
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Drowsiness Detection (Model-Based)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()