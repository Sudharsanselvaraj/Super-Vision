from confusion_detector import eyebrow_furrow, mouth_neutral, head_tilt

def detect_emotion(landmarks):
    score = 0

    if eyebrow_furrow(landmarks):
        score += 1
    if mouth_neutral(landmarks):
        score += 1
    if head_tilt(landmarks):
        score += 1

    if score >= 2:
        return "CONFUSED"
    if not mouth_neutral(landmarks):
        return "HAPPY"
    return "FOCUSED"
