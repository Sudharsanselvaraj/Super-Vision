def eyebrow_furrow(landmarks):
    left_brow = landmarks.landmark[65].y
    right_brow = landmarks.landmark[295].y
    eye_level = landmarks.landmark[159].y
    distance = abs((left_brow + right_brow) / 2 - eye_level)
    return distance < 0.015


def mouth_neutral(landmarks):
    left = landmarks.landmark[61].x
    right = landmarks.landmark[291].x
    top = landmarks.landmark[13].y
    bottom = landmarks.landmark[14].y
    ratio = (bottom - top) / (right - left)
    return ratio < 0.25


def head_tilt(landmarks):
    nose = landmarks.landmark[1].y
    chin = landmarks.landmark[152].y
    return abs(chin - nose) > 0.35
    # this is for making contri today
