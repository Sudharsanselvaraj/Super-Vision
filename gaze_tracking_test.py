import cv2
import mediapipe as mp
import time
from collections import deque
import math

cap = cv2.VideoCapture(0)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True
)

LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
EYE_TOP = 159
EYE_BOTTOM = 145
IRIS = 468

X_THRESHOLD = 0.35
Y_THRESHOLD = 0.35

look_start_time = None
ALERT_TIME = 4

confusion_start_time = None
CONFUSION_TIME = 3

timeline = deque(maxlen=300)

confidence_score = 50
last_frame_time = time.time()

blink_counter = 0
blink_state = False

def distance(p1, p2):
    return math.dist(p1, p2)

def eye_aspect_ratio(landmarks, eye):
    p1 = landmarks.landmark[eye[0]]
    p2 = landmarks.landmark[eye[1]]
    vertical = abs(landmarks.landmark[EYE_TOP].y - landmarks.landmark[EYE_BOTTOM].y)
    horizontal = abs(p2.x - p1.x)
    return vertical / horizontal

def get_gaze(landmarks):
    left_x = landmarks.landmark[LEFT_EYE[0]].x
    right_x = landmarks.landmark[LEFT_EYE[1]].x
    iris_x = landmarks.landmark[IRIS].x
    x_pos = (iris_x - left_x) / (right_x - left_x)

    top_y = landmarks.landmark[EYE_TOP].y
    bottom_y = landmarks.landmark[EYE_BOTTOM].y
    iris_y = landmarks.landmark[IRIS].y
    y_pos = (iris_y - top_y) / (bottom_y - top_y)

    if x_pos < X_THRESHOLD:
        return "LEFT"
    if x_pos > 1 - X_THRESHOLD:
        return "RIGHT"
    if y_pos < Y_THRESHOLD:
        return "UP"
    if y_pos > 1 - Y_THRESHOLD:
        return "DOWN"
    return "CENTER"

def look_away_alert(gaze):
    global look_start_time
    if gaze != "CENTER":
        if look_start_time is None:
            look_start_time = time.time()
        elif time.time() - look_start_time >= ALERT_TIME:
            return True
    else:
        look_start_time = None
    return False

def eyebrow_furrow(lm):
    return abs(((lm.landmark[65].y + lm.landmark[295].y) / 2) - lm.landmark[159].y) < 0.015

def mouth_neutral(lm):
    left = lm.landmark[61].x
    right = lm.landmark[291].x
    top = lm.landmark[13].y
    bottom = lm.landmark[14].y
    return (bottom - top) / (right - left) < 0.25

def head_tilt(lm):
    return abs(lm.landmark[152].y - lm.landmark[1].y) > 0.35

def detect_emotion(lm):
    score = 0
    if eyebrow_furrow(lm):
        score += 1
    if mouth_neutral(lm):
        score += 1
    if head_tilt(lm):
        score += 1

    if score >= 2:
        return "CONFUSED"
    if not mouth_neutral(lm):
        return "HAPPY"
    return "FOCUSED"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    dt = now - last_frame_time
    last_frame_time = now

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    status = ""
    color = (255, 255, 255)

    if not result.multi_face_landmarks:
        status = "PROCTOR ALERT: NO FACE"
        confidence_score -= 10

    elif len(result.multi_face_landmarks) > 1:
        status = "PROCTOR ALERT: MULTIPLE FACES"
        confidence_score -= 10

    else:
        face = result.multi_face_landmarks[0]
        gaze = get_gaze(face)

        if look_away_alert(gaze):
            status = "PROCTOR ALERT: LOOK AWAY"
            confidence_score -= 10
        else:
            emotion = detect_emotion(face)

            if emotion == "CONFUSED":
                if confusion_start_time is None:
                    confusion_start_time = now
                elif now - confusion_start_time >= CONFUSION_TIME:
                    status = "CONFUSED"
                    confidence_score -= 2 * dt
            else:
                confusion_start_time = None
                status = emotion
                if emotion == "FOCUSED":
                    confidence_score += 2 * dt

            timeline.append((now, status))

            ear = eye_aspect_ratio(face, LEFT_EYE)
            if ear < 0.18 and not blink_state:
                blink_counter += 1
                blink_state = True
            if ear > 0.22:
                blink_state = False

    confidence_score = max(0, min(100, confidence_score))

    cv2.putText(frame, status, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Confidence: {int(confidence_score)}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(frame, f"Blinks: {blink_counter}", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow("Super Vision – Full Engagement Engine", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
