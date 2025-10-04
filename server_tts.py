"""server_tts.py - Run on EC2 with: uvicorn server_tts:app --host 0.0.0.0 --port 9000"""
import os, json, io, wave, asyncio
from typing import Set
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from RealtimeTTS import TextToAudioStream
from RealtimeTTS.engines.piper_engine import PiperEngine, PiperVoice
from pathlib import Path
from fastapi.staticfiles import StaticFiles

PIPER = "./piper-repo/piper"
MODEL = "./piper-repo/voice-onnx-json-files/en_US-hfc_female-medium.onnx"
CFG = "./piper-repo/voice-onnx-json-files/en_US-hfc_female-medium.onnx.json"

with open(CFG, "r") as f:
    SAMPLE_RATE = int(json.load(f).get("audio", {}).get("sample_rate", 22050))

app = FastAPI()
WEB_DIR = Path(__file__).parent / "web"
if WEB_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

voice = PiperVoice(model_file=MODEL, config_file=CFG)
engine = PiperEngine(piper_path=PIPER, voice=voice)
clients: Set[WebSocket] = set()

class StreamingSession:
    def __init__(self):
        self.audio_queue = deque()
        self.is_playing = False
        self.playback_task = None
        self.lock = asyncio.Lock()
        
    async def add_text_and_synthesize(self, text: str):
        if not text.strip():
            return
        loop = asyncio.get_event_loop()
        wav_bytes = await loop.run_in_executor(None, self._synthesize_to_wav, text)
        async with self.lock:
            self.audio_queue.append({'text': text, 'audio': wav_bytes})
            print(f"Queued: '{text[:50]}...' | Queue: {len(self.audio_queue)} | Clients: {len(clients)}")
    
    def _synthesize_to_wav(self, text: str) -> bytes:
        local_chunks = []
        def on_chunk(b: bytes):
            if b:
                local_chunks.append(b)
        stream = TextToAudioStream(engine, muted=True)
        stream.feed(text)
        stream.play(on_audio_chunk=on_chunk, minimum_first_fragment_length=8)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            for chunk in local_chunks:
                wf.writeframes(chunk)
        return buf.getvalue()
    
    async def start_playback(self):
        async with self.lock:
            if self.is_playing:
                return False
            if len(self.audio_queue) == 0:
                print("Queue empty")
                return False
            if len(clients) == 0:
                print("No clients connected!")
                return False
            print(f"Starting playback: {len(self.audio_queue)} chunks -> {len(clients)} clients")
            self.is_playing = True
        self.playback_task = asyncio.create_task(self._playback_loop())
        return True
    
    async def _playback_loop(self):
        empty_checks = 0
        chunks_sent = 0
        while True:
            async with self.lock:
                if not self.audio_queue:
                    empty_checks += 1
                    if empty_checks >= 30:
                        self.is_playing = False
                        for ws in list(clients):
                            try:
                                await ws.send_text("__end__")
                            except:
                                clients.discard(ws)
                        print(f"Playback complete: {chunks_sent} chunks")
                        break
                else:
                    empty_checks = 0
                    chunk = self.audio_queue.popleft()
            if empty_checks > 0:
                await asyncio.sleep(0.1)
                continue
            for ws in list(clients):
                try:
                    await ws.send_text(f"__text__:{chunk['text']}")
                    await ws.send_bytes(chunk['audio'])
                except:
                    clients.discard(ws)
            chunks_sent += 1
            await asyncio.sleep(0.05)
    
    def reset(self):
        if self.playback_task and not self.playback_task.done():
            self.playback_task.cancel()
        self.audio_queue.clear()
        self.is_playing = False
        print("Reset")

current_session = StreamingSession()

@app.post("/synthesize")
async def synthesize(text: str = Query(...)):
    await current_session.add_text_and_synthesize(text)
    return {"status": "queued", "queue_size": len(current_session.audio_queue)}

@app.post("/play")
async def play():
    if len(clients) == 0:
        return JSONResponse({"error": "No clients. Open http://YOUR_IP:9000/app"}, status_code=400)
    success = await current_session.start_playback()
    return {"status": "playing" if success else "error"}

@app.post("/reset")
async def reset():
    current_session.reset()
    return {"status": "reset"}

@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    print(f"Client connected! Total: {len(clients)}")
    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text("pong")
    except:
        clients.discard(ws)
        print(f"Client disconnected. Total: {len(clients)}")

@app.get("/health")
async def health():
    return {"status": "ok", "clients": len(clients), "queue_size": len(current_session.audio_queue)}

@app.get("/")
async def root():
    return HTMLResponse(f"""<html><body style='font-family:Arial;padding:40px'>
    <h1>TTS Server</h1>
    <p>Status: Running on port 9000</p>
    <p>Clients: {len(clients)}</p>
    <p>Queue: {len(current_session.audio_queue)}</p>
    <h3>To hear audio: <a href='/app'>Open /app</a> and click Connect</h3>
    </body></html>""")