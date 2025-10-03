import json
from typing import List, Optional

import torch
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# TensorRT-LLM utils
from utils import load_tokenizer
from tensorrt_llm.runtime import ModelRunner

# ---------------- Hardcoded Config ----------------
ENGINE_DIR = "/home/ubuntu/voiceai-bot/TensorRT-LLM/project-models/trt-engine"
TOKENIZER_DIR = "/home/ubuntu/voiceai-bot/TensorRT-LLM/project-models/raw-model"
MAX_INPUT_LENGTH = 1024
MAX_OUTPUT_LEN = 512

# TTS Server URL - connects to your server_tts.py
TTS_SERVER_URL = "http://localhost:8001"
# --------------------------------------------------

# FastAPI app
app = FastAPI(title="TensorRT-LLM + TTS Client")

# Load tokenizer once
tokenizer, pad_id, end_id = load_tokenizer(
    tokenizer_dir=TOKENIZER_DIR,
    model_name="rank0.engine",
    model_version=None
)

# Load model runner once
runner = ModelRunner.from_dir(
    engine_dir=ENGINE_DIR,
    max_output_len=MAX_OUTPUT_LEN,
    rank=0
)

# HTTP client for TTS server communication
http_client = httpx.AsyncClient(timeout=30.0)

# ---------------- Pydantic Request ----------------
class GenerationRequest(BaseModel):
    input_texts: List[str]
    temperature: Optional[float] = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 1.0
    num_beams: Optional[int] = 1
    streaming: Optional[bool] = False
    enable_tts: Optional[bool] = True

# ---------------- Helper Function ----------------
async def send_to_tts(text: str):
    """Send text to TTS server"""
    try:
        response = await http_client.post(
            f"{TTS_SERVER_URL}/say",
            params={"text": text}
        )
        return response.status_code == 200
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

# ---------------- API Endpoint ----------------
@app.post("/generate")
async def generate(request: GenerationRequest):
    # Encode inputs into tokens
    batch_input_ids = []
    for text in request.input_texts:
        input_ids = tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH
        )
        batch_input_ids.append(torch.tensor(input_ids, dtype=torch.int32))

    # ---------------- STREAMING MODE ----------------
    if request.streaming:
        async def token_stream():
            accumulated_text = ""
            try:
                with torch.no_grad():
                    outputs = runner.generate(
                        batch_input_ids=batch_input_ids,
                        max_new_tokens=MAX_OUTPUT_LEN,
                        temperature=request.temperature,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        num_beams=request.num_beams,
                        end_id=end_id,
                        pad_id=pad_id,
                        streaming=True,
                        return_dict=True
                    )

                    for chunk in outputs:
                        ids = chunk["output_ids"][0][0].tolist()
                        gen_ids = ids[len(batch_input_ids[0]):]
                        piece = tokenizer.decode(gen_ids, skip_special_tokens=True)

                        if piece.strip():
                            accumulated_text = piece
                            yield f"data: {json.dumps({'text': piece})}\n\n"

                # Send complete text to TTS
                if request.enable_tts and accumulated_text.strip():
                    await send_to_tts(accumulated_text.strip())

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(token_stream(), media_type="text/event-stream")

    # ---------------- NON-STREAMING MODE ----------------
    try:
        with torch.no_grad():
            outputs = runner.generate(
                batch_input_ids=batch_input_ids,
                max_new_tokens=MAX_OUTPUT_LEN,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                num_beams=request.num_beams,
                end_id=end_id,
                pad_id=pad_id,
                return_dict=True
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Decode final outputs
    output_texts = []
    output_ids = outputs["output_ids"]
    for i, ids_tensor in enumerate(output_ids):
        ids = ids_tensor[0][len(batch_input_ids[i]):].tolist()
        text_out = tokenizer.decode(ids, skip_special_tokens=True)
        output_texts.append(text_out)

        # Send to TTS server
        if request.enable_tts and text_out.strip():
            await send_to_tts(text_out.strip())

    return {"outputs": output_texts}


@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()


# ---------------- Run the server ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)