import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face in result.multi_face_landmarks:
            for lm in face.landmark:
                h, w, _ = frame.shape
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Face Mesh", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
