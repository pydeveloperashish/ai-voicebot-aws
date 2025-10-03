"""
run_transcribe.py

Live mic -> AWS Transcribe (streaming) -> your EC2 LLM endpoint (SSE)
- Debounced early start: len(partial) >= 24 and stable >= 300 ms
- Pause trigger: if no partial for 350 ms, fire generation too
- No cancellation: once a generation starts for an utterance, let it finish

Run:
    python run_transcribe.py
"""

import asyncio
import os
import sys
import json
import queue
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import boto3
from amazon_transcribe.client import TranscribeStreamingClient
import aiohttp

# ---------- CONFIG ----------
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 0.1  # seconds
LANGUAGE_CODE = "en-US"
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

# Latency knobs
MIN_PARTIAL_CHARS = 24       # early trigger threshold
STABLE_MS = 0            # partial must remain unchanged for this long
PAUSE_MS = 450               # if no partial arrives for this long, treat as boundary

# Your LLM endpoint (SSE style: "data: {json}" with {"text": "..."} tokens)
LLM_URL = "http://13.202.148.182:8000/generate"
LLM_TIMEOUT_SECS = 120

# Optional AWS region fallback (else uses env or ~/.aws/config)
FALLBACK_REGION = None
# ---------------------------

# Audio queue from mic -> Transcribe
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
    except Exception:
        pass
    return FALLBACK_REGION


def audio_callback(indata: np.ndarray, frames: int, time_info, status):
    if status:
        print(f"Audio input status: {status}", file=sys.stderr)
    if indata.dtype in ("float32", "float64"):
        data16 = (indata * 32767.0).astype(np.int16)
    else:
        data16 = indata.astype(np.int16)
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
        except Exception:
            pass


def _is_partial(result_obj) -> bool:
    return bool(getattr(result_obj, "is_partial", None) or getattr(result_obj, "isPartial", None))


class LiveGenerator:
    """
    Starts SSE POST to LLM; prints tokens as they arrive.
    - No cancel: if a job is running, new starts are ignored.
    - Accepts optional prev context (last user utterance).
    """

    def __init__(self, url: str, timeout: int = 120):
        self.url = url
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def ensure_session(self):
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)

    @staticmethod
    def build_prompt(curr: str, prev: Optional[str]) -> str:
        if prev and prev.strip():
            # Simple, model-agnostic stitch. Keeps "one previous" as context.
            return f"Previous: {prev.strip()}\nQuestion: {curr.strip()}"
        return curr.strip()

    async def _stream_job(self, prompt: str):
        try:
            await self.ensure_session()
            payload = {"input_texts": [prompt], "streaming": True}
            headers = {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
            async with self.session.post(self.url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    print(f"\n[LLM HTTP {resp.status}] {err}", file=sys.stderr)
                    return
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        data = line[len("data:"):].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            msg = json.loads(data)
                            token = msg.get("text")
                            if token is not None:
                                sys.stdout.write(token)
                                sys.stdout.flush()
                        except Exception:
                            pass
            sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception as e:
            print(f"\n[LLM stream error] {e}", file=sys.stderr)

    async def start(self, prompt: str) -> bool:
        """
        Start a generation only if none is running. Returns True if started.
        """
        async with self._lock:
            if self.task and not self.task.done():
                return False
            self.task = asyncio.create_task(self._stream_job(prompt))
            return True

    async def wait_current(self):
        if self.task:
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def close(self):
        await self.wait_current()
        if self.session and not self.session.closed:
            await self.session.close()



async def process_output_stream(output_stream, generator: LiveGenerator):
    """
    Debounced early start (>= 24 chars stable for 300 ms) OR pause trigger (>= 350 ms of no change).
    No cancellations. One start per utterance.
    """
    last_partial_printed = ""
    current_partial_text = ""
    last_change_ts = time.time()
    started_for_utterance = False

    def now_ms():
        return time.time() * 1000.0

    last_event_ms = now_ms()

    async for event in output_stream:
        last_event_ms = now_ms()

        transcript = getattr(event, "transcript", None) or event
        results = getattr(transcript, "results", None)
        if not results:
            continue
        first = results[0]
        alternatives = getattr(first, "alternatives", None) or []
        if not alternatives:
            continue
        text = getattr(alternatives[0], "transcript", "") or ""

        # Track stability window
        if text != current_partial_text:
            current_partial_text = text
            last_change_ts = time.time()

        if _is_partial(first):
            # Show partial in-place
            pad = " " * max(0, len(last_partial_printed) - len(text))
            sys.stdout.write("\r" + text + pad)
            sys.stdout.flush()
            last_partial_printed = text

            if not started_for_utterance:
                stable_ms = (time.time() - last_change_ts) * 1000.0

                # Debounced early trigger
                if len(text) >= MIN_PARTIAL_CHARS and stable_ms >= STABLE_MS:
                    sys.stdout.write("\n[early gen]\n")
                    sys.stdout.flush()
                    started = await generator.start(text)
                    if started:
                        started_for_utterance = True
                else:
                    # Pause trigger: boundary even if short
                    silence_ms = (time.time() - last_change_ts) * 1000.0
                    if len(text) > 0 and silence_ms >= PAUSE_MS:
                        sys.stdout.write("\n[pause gen]\n")
                        sys.stdout.flush()
                        started = await generator.start(text)
                        if started:
                            started_for_utterance = True

        else:
            # FINAL result for this utterance
            pad = " " * max(0, len(last_partial_printed) - len(text))
            sys.stdout.write("\r" + text + pad + "\n")
            sys.stdout.flush()
            last_partial_printed = ""

            if not started_for_utterance and len(text) > 0:
                await generator.start(text)

            # Reset per-utterance state; generator continues printing
            started_for_utterance = False
            current_partial_text = ""
            last_change_ts = time.time()


async def main(device: Optional[int] = None):
    region = detect_region()
    if not region:
        print(
            "\nERROR: No AWS region found.\n"
            "Set AWS_REGION / AWS_DEFAULT_REGION, or configure via `aws configure`,\n"
            "or set FALLBACK_REGION in this script.\n",
            file=sys.stderr,
        )
        return

    client = TranscribeStreamingClient(region=region)
    stream = await client.start_stream_transcription(
        language_code=LANGUAGE_CODE,
        media_sample_rate_hz=SAMPLE_RATE,
        media_encoding="pcm",
    )

    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels = CHANNELS

    print(f"Using AWS region: {region}")
    print("Mic is live. Speak. Ctrl+C to stop.\n")

    mic = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=CHUNK_SAMPLES,
        callback=audio_callback,
        device=device,
    )
    mic.start()

    generator = LiveGenerator(LLM_URL, timeout=LLM_TIMEOUT_SECS)

    sender_task = asyncio.create_task(audio_sender_loop(stream.input_stream))
    receiver_task = asyncio.create_task(process_output_stream(stream.output_stream, generator))

    try:
        while True:
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        audio_queue.put(None)
        for t in (receiver_task, sender_task):
            if not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        try:
            mic.stop()
            mic.close()
        except Exception:
            pass
        await generator.close()
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    try:
        asyncio.run(main(device=None))
    except Exception as e:
        print("Fatal error:", e, file=sys.stderr)
        raise
