"""custom_LLM_inference.py - Run on EC2 with: uvicorn custom_LLM_inference:app --host 0.0.0.0 --port 8000"""
import json
from typing import List, Optional
import torch, httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from utils import load_tokenizer
from tensorrt_llm.runtime import ModelRunner

ENGINE_DIR = "/home/ubuntu/voiceai-bot/TensorRT-LLM/project-models/trt-engine"
TOKENIZER_DIR = "/home/ubuntu/voiceai-bot/TensorRT-LLM/project-models/raw-model"
MAX_INPUT_LENGTH = 1024
MAX_OUTPUT_LEN = 512
TTS_SERVER_URL = "http://127.0.0.1:9000"

app = FastAPI()
tokenizer = None
pad_id = None
end_id = None
runner = None
http_client = None

@app.on_event("startup")
async def startup_event():
    global tokenizer, pad_id, end_id, runner, http_client
    print("Loading model...")
    tokenizer, pad_id, end_id = load_tokenizer(tokenizer_dir=TOKENIZER_DIR, model_name="rank0.engine", model_version=None)
    runner = ModelRunner.from_dir(engine_dir=ENGINE_DIR, max_output_len=MAX_OUTPUT_LEN, rank=0)
    http_client = httpx.AsyncClient(timeout=30.0)
    print("LLM Server ready!")

@app.on_event("shutdown")
async def shutdown_event():
    if http_client:
        await http_client.aclose()

class GenerationRequest(BaseModel):
    input_texts: List[str]
    temperature: Optional[float] = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 1.0
    num_beams: Optional[int] = 1
    streaming: Optional[bool] = False
    enable_tts: Optional[bool] = True

async def send_to_tts_buffer(text: str):
    if not text or not text.strip():
        return False
    try:
        response = await http_client.post(f"{TTS_SERVER_URL}/synthesize", params={"text": text.strip()})
        return response.status_code == 200
    except Exception as e:
        print(f"TTS error: {e}")
        return False

@app.post("/generate")
async def generate(request: GenerationRequest):
    if tokenizer is None or runner is None:
        raise HTTPException(status_code=503, detail="Model loading")
    
    batch_input_ids = []
    for text in request.input_texts:
        input_ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=MAX_INPUT_LENGTH)
        batch_input_ids.append(torch.tensor(input_ids, dtype=torch.int32))

    if request.streaming:
        async def token_stream():
            try:
                with torch.no_grad():
                    gen = runner.generate(
                        batch_input_ids=batch_input_ids, max_new_tokens=MAX_OUTPUT_LEN,
                        temperature=request.temperature, top_k=request.top_k, top_p=request.top_p,
                        num_beams=request.num_beams, end_id=end_id, pad_id=pad_id,
                        streaming=True, return_dict=True
                    )
                    decoded_so_far = ""
                    sentence_buffer = ""
                    for step in gen:
                        ids_all = step["output_ids"][0][0].tolist()
                        gen_ids = ids_all[len(batch_input_ids[0]):]
                        current_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                        if len(current_text) > len(decoded_so_far):
                            delta = current_text[len(decoded_so_far):]
                            decoded_so_far = current_text
                            sentence_buffer += delta
                            yield f'data: {json.dumps({"text": delta})}\n\n'
                            if request.enable_tts:
                                if any(p in sentence_buffer for p in (".", "?", "!", "\n")) or len(sentence_buffer) >= 80:
                                    await send_to_tts_buffer(sentence_buffer.strip())
                                    sentence_buffer = ""
                    if request.enable_tts and sentence_buffer.strip():
                        await send_to_tts_buffer(sentence_buffer.strip())
            except Exception as e:
                yield f'data: {json.dumps({"error": str(e)})}\n\n'
            yield "data: [DONE]\n\n"
        return StreamingResponse(token_stream(), media_type="text/event-stream")
    
    # Non-streaming
    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids=batch_input_ids, max_new_tokens=MAX_OUTPUT_LEN,
            temperature=request.temperature, top_k=request.top_k, top_p=request.top_p,
            num_beams=request.num_beams, end_id=end_id, pad_id=pad_id, return_dict=True
        )
    output_texts = []
    for i, ids_tensor in enumerate(outputs["output_ids"]):
        ids = ids_tensor[0][len(batch_input_ids[i]):].tolist()
        text_out = tokenizer.decode(ids, skip_special_tokens=True)
        output_texts.append(text_out)
        if request.enable_tts and text_out.strip():
            await send_to_tts_buffer(text_out.strip())
    return {"outputs": output_texts}

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": runner is not None}