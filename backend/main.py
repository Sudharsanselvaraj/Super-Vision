from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/student")
async def student_ws(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_json()
        await ws.send_json(data)
