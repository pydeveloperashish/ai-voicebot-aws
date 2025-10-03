# ubuntu@ip-172-31-15-135:~/voiceai-bot/RealtimeTTS-repo$ cat server_tts.py 
import os, json, io, wave, asyncio, threading
from typing import Set, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import JSONResponse
from RealtimeTTS import TextToAudioStream
from RealtimeTTS.engines.piper_engine import PiperEngine, PiperVoice
from pathlib import Path
from fastapi.staticfiles import StaticFiles

# --- Configure your Piper paths ---
PIPER = "./piper-repo/piper"  # chmod +x
MODEL = "./piper-repo/voice-onnx-json-files/en_US-hfc_female-medium.onnx"
CFG   = "./piper-repo/voice-onnx-json-files/en_US-hfc_female-medium.onnx.json"

# Derive sample rate from Piper voice JSON (don’t hardcode)
with open(CFG, "r", encoding="utf-8") as f:
    _cfg = json.load(f)
SAMPLE_RATE = int(_cfg.get("audio", {}).get("sample_rate", 22050))

app = FastAPI(title="RealtimeTTS + Piper (push server)")
WEB_DIR = Path(__file__).parent / "web"
app.mount("/app", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

# Piper engine + voice
voice = PiperVoice(model_file=MODEL, config_file=CFG)
engine = PiperEngine(piper_path=PIPER, voice=voice)

# Shared client set
clients: Set[WebSocket] = set()

# Lock and buffer for capturing chunks for the *current* request
audio_lock = threading.Lock()
audio_chunks: List[bytes] = []

def chunks_to_wav(chunks: List[bytes], sample_rate: int) -> bytes:
    """Wrap raw PCM 16-bit mono chunks into a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        for c in chunks:
            if c:
                wf.writeframes(c)
    return buf.getvalue()



def make_streamer():
    local_chunks = []

    def on_chunk(b: bytes):
        if b:
            local_chunks.append(b)

    # Headless: no audio device
    stream = TextToAudioStream(engine, muted=False)
    return stream, local_chunks, on_chunk

@app.post("/say")
async def say(text: str = Query(..., min_length=1, max_length=2000)):
    try:
        # --- NEW: tell clients what we're about to speak ---
        dead = set()
        for ws in clients:
            try:
                await ws.send_text("__text__:" + text)
            except:
                dead.add(ws)
        for ws in dead:
            clients.discard(ws)
        # ---------------------------------------------------

        stream, local_chunks, on_chunk = make_streamer()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, stream.feed, text)
        await loop.run_in_executor(
            None,
            lambda: stream.play(
                on_audio_chunk=on_chunk,
                minimum_first_fragment_length=8
            )
        )

        # wrap to WAV (unchanged)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
            for c in local_chunks: wf.writeframes(c)
        wav_bytes = buf.getvalue()
        print(f"Broadcasting WAV: {len(wav_bytes)} bytes to {len(clients)} client(s)")
        # broadcast audio + end (unchanged)
        dead = set(); sent = 0
        for ws in clients:
            try:
                await ws.send_bytes(wav_bytes)
                await ws.send_text("__end__")
                sent += 1
            except:
                dead.add(ws)
        for ws in dead:
            clients.discard(ws)

        return JSONResponse({"status":"ok","text":text,"clients":len(clients),
                             "sent_to":sent,"audio_size":len(wav_bytes)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    print(f"Client connected. Total: {len(clients)}")
    try:
        while True:
            # keepalive (client can send "ping")
            _ = await ws.receive_text()
    except WebSocketDisconnect:
        clients.discard(ws)
        print(f"Client disconnected. Total: {len(clients)}")
    except Exception:
        clients.discard(ws)

@app.get("/health")
async def health():
    return {"status": "ok", "clients": len(clients)}