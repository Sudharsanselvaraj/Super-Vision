import cv2
import mediapipe as mp
import time

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True
)

LEFT_EYE = [33, 133]
EYE_TOP = 159
EYE_BOTTOM = 145
IRIS = 468

X_THRESHOLD = 0.35
Y_THRESHOLD = 0.35

look_start_time = None
ALERT_TIME = 4

def get_gaze(lm):
    left_x = lm.landmark[LEFT_EYE[0]].x
    right_x = lm.landmark[LEFT_EYE[1]].x
    iris_x = lm.landmark[IRIS].x
    x_pos = (iris_x - left_x) / (right_x - left_x)

    top_y = lm.landmark[EYE_TOP].y
    bottom_y = lm.landmark[EYE_BOTTOM].y
    iris_y = lm.landmark[IRIS].y
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

def analyze_frame(frame):
    global look_start_time

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    response = {
        "status": "FOCUSED",
        "gaze": "CENTER",
        "proctor_alert": False,
        "timestamp": time.time()
    }

    if not result.multi_face_landmarks:
        response["status"] = "NO_FACE"
        response["proctor_alert"] = True
        look_start_time = None
        return response

    if len(result.multi_face_landmarks) > 1:
        response["status"] = "MULTIPLE_FACES"
        response["proctor_alert"] = True
        look_start_time = None
        return response

    face = result.multi_face_landmarks[0]
    gaze = get_gaze(face)
    response["gaze"] = gaze

    if gaze != "CENTER":
        if look_start_time is None:
            look_start_time = time.time()
        elif time.time() - look_start_time >= ALERT_TIME:
            response["status"] = "LOOK_AWAY"
            response["proctor_alert"] = True
    else:
        look_start_time = None

    return response
