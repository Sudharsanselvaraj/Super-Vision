from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import base64
from vision_engine import analyze_frame

app = FastAPI()

@app.websocket("/ws/student")
async def student_ws(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_text()
        img_bytes = base64.b64decode(data)
        np_img = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        result = analyze_frame(frame)
        await ws.send_json(result)

