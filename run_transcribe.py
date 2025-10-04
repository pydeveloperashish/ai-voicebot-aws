"""run_transcribe.py - Run on LOCAL machine with: python run_transcribe.py"""
import asyncio, os, sys, json, queue, time
from typing import Optional
import numpy as np
import sounddevice as sd
import boto3
from amazon_transcribe.client import TranscribeStreamingClient
import aiohttp

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 0.1
LANGUAGE_CODE = "en-US"
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
MIN_PARTIAL_CHARS = 30
STABLE_MS = 200
PAUSE_MS = 500

EC2_PUBLIC_IP = "13.202.148.182"
LLM_URL = f"http://{EC2_PUBLIC_IP}:8000/generate"
TTS_URL = f"http://{EC2_PUBLIC_IP}:9000"
LLM_TIMEOUT_SECS = 120
FALLBACK_REGION = None

audio_queue: queue.Queue[bytes] = queue.Queue()

def detect_region() -> Optional[str]:
    for env_var in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        v = os.environ.get(env_var)
        if v:
            return v
    try:
        sess = boto3.Session()
        if sess.region_name:
            return sess.region_name
    except:
        pass
    return FALLBACK_REGION

def audio_callback(indata: np.ndarray, frames: int, time_info, status):
    if status:
        print(f"Audio: {status}", file=sys.stderr)
    data16 = (indata * 32767.0).astype(np.int16) if indata.dtype in ("float32", "float64") else indata.astype(np.int16)
    audio_queue.put(data16.tobytes())

async def audio_sender_loop(input_stream):
    loop = asyncio.get_event_loop()
    try:
        while True:
            chunk = await loop.run_in_executor(None, audio_queue.get)
            if chunk is None:
                break
            await input_stream.send_audio_event(audio_chunk=chunk)
    finally:
        try:
            await input_stream.end_stream()
        except:
            pass

def _is_partial(result_obj) -> bool:
    return bool(getattr(result_obj, "is_partial", None) or getattr(result_obj, "isPartial", None))

class LiveGenerator:
    def __init__(self, llm_url: str, tts_url: str, timeout: int = 120):
        self.llm_url = llm_url
        self.tts_url = tts_url
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self.tts_session: Optional[aiohttp.ClientSession] = None

    async def ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        if self.tts_session is None or self.tts_session.closed:
            self.tts_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))

    async def send_user_question(self, question: str):
        """Send user's question to TTS for display in browser"""
        try:
            await self.ensure_session()
            await self.tts_session.post(
                f"{self.tts_url}/user_message",
                params={"text": question}
            )
        except Exception as e:
            print(f"Failed to send user question: {e}", file=sys.stderr)

    async def wait_for_queue_ready(self) -> bool:
        for attempt in range(40):
            try:
                await self.ensure_session()
                async with self.tts_session.get(f"{self.tts_url}/health") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        queue_size = data.get('queue_size', 0)
                        if queue_size >= 2 or (attempt >= 20 and queue_size >= 1):
                            return True
            except:
                pass
            await asyncio.sleep(0.1)
        return False

    async def trigger_playback(self):
        try:
            await self.ensure_session()
            has_content = await self.wait_for_queue_ready()
            if not has_content:
                return False
            async with self.tts_session.post(f"{self.tts_url}/play") as resp:
                if resp.status == 200:
                    print("\n[Playing]", file=sys.stderr)
                    return True
        except Exception as e:
            print(f"\n[Play error: {e}]", file=sys.stderr)
        return False

    async def reset_tts(self):
        try:
            await self.ensure_session()
            await self.tts_session.post(f"{self.tts_url}/reset")
        except:
            pass

    async def _stream_job(self, prompt: str):
        try:
            await self.ensure_session()
            await self.reset_tts()
            
            # Send user's question to browser
            await self.send_user_question(prompt)
            
            payload = {"input_texts": [f"Question: {prompt}\nAnswer:"], "streaming": True, "enable_tts": True, "temperature": 0.7, "top_p": 0.9}
            headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
            async with self.session.post(self.llm_url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    print(f"\n[LLM error {resp.status}]", file=sys.stderr)
                    return
                print("\n[Assistant] ", end="", flush=True)
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data and data != "[DONE]":
                            try:
                                token = json.loads(data).get("text")
                                if token:
                                    sys.stdout.write(token)
                                    sys.stdout.flush()
                            except:
                                pass
            sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception as e:
            print(f"\n[Error: {e}]", file=sys.stderr)

    async def start(self, prompt: str) -> bool:
        async with self._lock:
            if self.task and not self.task.done():
                return False
            self.task = asyncio.create_task(self._stream_job(prompt))
            return True

    async def wait_current(self):
        if self.task:
            try:
                await self.task
            except:
                pass

    async def close(self):
        await self.wait_current()
        if self.session and not self.session.closed:
            await self.session.close()
        if self.tts_session and not self.tts_session.closed:
            await self.tts_session.close()

async def process_output_stream(output_stream, generator: LiveGenerator):
    last_partial_printed = ""
    current_partial_text = ""
    last_change_ts = time.time()
    started_for_utterance = False
    playback_triggered = False

    async for event in output_stream:
        transcript = getattr(event, "transcript", None) or event
        results = getattr(transcript, "results", None)
        if not results:
            continue
        first = results[0]
        alternatives = getattr(first, "alternatives", None) or []
        if not alternatives:
            continue
        text = getattr(alternatives[0], "transcript", "") or ""

        if text != current_partial_text:
            current_partial_text = text
            last_change_ts = time.time()

        if _is_partial(first):
            pad = " " * max(0, len(last_partial_printed) - len(text))
            sys.stdout.write("\r[You] " + text + pad)
            sys.stdout.flush()
            last_partial_printed = text

            if not started_for_utterance:
                stable_ms = (time.time() - last_change_ts) * 1000.0
                if len(text) >= MIN_PARTIAL_CHARS and stable_ms >= STABLE_MS:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    if await generator.start(text):
                        started_for_utterance = True

            if started_for_utterance and not playback_triggered:
                silence_ms = (time.time() - last_change_ts) * 1000.0
                if silence_ms >= PAUSE_MS:
                    if await generator.trigger_playback():
                        playback_triggered = True
        else:
            pad = " " * max(0, len(last_partial_printed) - len(text))
            sys.stdout.write("\r[You] " + text + pad + "\n")
            sys.stdout.flush()
            last_partial_printed = ""

            if not started_for_utterance and len(text) > 0:
                await generator.start(text)
                started_for_utterance = True

            if not playback_triggered:
                await generator.trigger_playback()

            started_for_utterance = False
            playback_triggered = False
            current_partial_text = ""
            last_change_ts = time.time()

async def main(device: Optional[int] = None):
    region = detect_region()
    if not region:
        print("\nERROR: No AWS region. Run: aws configure\n", file=sys.stderr)
        return

    print("\nTesting EC2 connectivity...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{LLM_URL.replace('/generate', '/health')}", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                print(f"✓ LLM Server OK" if resp.status == 200 else f"⚠ LLM returned {resp.status}")
            async with session.get(f"{TTS_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                print(f"✓ TTS Server OK" if resp.status == 200 else f"⚠ TTS returned {resp.status}")
    except Exception as e:
        print(f"✗ Cannot reach servers: {e}")
        return

    print(f"\nOpen http://{EC2_PUBLIC_IP}:9000/app in browser!\n")

    client = TranscribeStreamingClient(region=region)
    stream = await client.start_stream_transcription(language_code=LANGUAGE_CODE, media_sample_rate_hz=SAMPLE_RATE, media_encoding="pcm")
    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels = CHANNELS
    mic = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32", blocksize=CHUNK_SAMPLES, callback=audio_callback, device=device)
    mic.start()
    generator = LiveGenerator(LLM_URL, TTS_URL, timeout=LLM_TIMEOUT_SECS)
    sender_task = asyncio.create_task(audio_sender_loop(stream.input_stream))
    receiver_task = asyncio.create_task(process_output_stream(stream.output_stream, generator))

    try:
        while True:
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        audio_queue.put(None)
        for t in (receiver_task, sender_task):
            if not t.done():
                t.cancel()
                try:
                    await t
                except:
                    pass
        try:
            mic.stop()
            mic.close()
        except:
            pass
        await generator.close()

if __name__ == "__main__":
    try:
        asyncio.run(main(device=None))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise