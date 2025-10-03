# (venv) ubuntu@ip-172-31-15-135:~/voiceai-bot/TensorRT-LLM$ cat examples/custom_llm_inference.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import torch

# TensorRT-LLM imports
from tensorrt_llm.runtime import ModelRunner
from utils import load_tokenizer, throttle_generator

# ---------------- Hardcoded Config ----------------
ENGINE_DIR = "/home/ubuntu/voiceai-bot/TensorRT-LLM/project-models/trt-engine"
TOKENIZER_DIR = "/home/ubuntu/voiceai-bot/TensorRT-LLM/project-models/raw-model"
MAX_INPUT_LENGTH = 1024
MAX_OUTPUT_LEN = 512
STREAMING_INTERVAL = 1
# --------------------------------------------------

app = FastAPI(title="TensorRT-LLM Inference API")

# Load tokenizer & model once (kept in memory)
tokenizer, pad_id, end_id = load_tokenizer(
    tokenizer_dir=TOKENIZER_DIR,
    model_name="rank0.engine",
    model_version=None,
    tokenizer_type=None,
)

runner = ModelRunner.from_dir(
    engine_dir=ENGINE_DIR,
    max_output_len=MAX_OUTPUT_LEN,
    rank=0,
)

# ---------- Request schema ----------
class GenerationRequest(BaseModel):
    input_texts: List[str] = Field(..., description="List of prompts")
    streaming: bool = Field(False, description="Return tokens as they are generated")
    temperature: Optional[float] = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 1.0
    num_beams: Optional[int] = 1


@app.get("/")
def root():
    return {"status": "ok", "endpoint": "/generate"}


def _encode_batch(texts: List[str]):
    batch = []
    for txt in texts:
        ids = tokenizer.encode(
            txt,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        )
        batch.append(torch.tensor(ids, dtype=torch.int32))
    return batch


@app.post("/generate")
def generate(request: GenerationRequest):
    if request.streaming and request.num_beams != 1:
        request.num_beams = 1  # enforce for streaming

    if not request.input_texts:
        raise HTTPException(status_code=400, detail="input_texts must be provided")

    batch_input_ids = _encode_batch(request.input_texts)

    # -------- Non-streaming mode --------
    if not request.streaming:
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
                    return_dict=True,
                    streaming=False,
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

        output_texts = []
        for i, ids_tensor in enumerate(outputs["output_ids"]):
            full_ids = ids_tensor[0].tolist()
            prompt_len = len(batch_input_ids[i])
            gen_ids = full_ids[prompt_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            output_texts.append(text)
        return {"outputs": output_texts}

    # -------- Streaming mode (SSE) --------
    try:
        gen = runner.generate(
            batch_input_ids=batch_input_ids,
            max_new_tokens=MAX_OUTPUT_LEN,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            num_beams=1,
            end_id=end_id,
            pad_id=pad_id,
            return_dict=True,
            streaming=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    def sse_event_stream():
        decoded_so_far = [""] * len(batch_input_ids)

        for step in throttle_generator(gen, STREAMING_INTERVAL):
            output_ids = step.get("output_ids", None)
            if output_ids is None:
                continue

            for i, ids_tensor in enumerate(output_ids):
                full_ids = ids_tensor[0].tolist()
                prompt_len = len(batch_input_ids[i])
                gen_ids = full_ids[prompt_len:]

                current_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

                # Compute only the delta (new text)
                delta = current_text[len(decoded_so_far[i]):]
                if delta:
                    decoded_so_far[i] = current_text
                    yield f"data: {{\"text\": \"{delta}\"}}\n\n"

        # End of stream
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_event_stream(), media_type="text/event-stream")


# ---------- Run server ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



# curl -s -N -X POST http://13.202.148.182:8000/generate \
#   -H "Content-Type: application/json" \
#   -d '{
#     "input_texts": ["Explain in detail What is Glaucoma?"],
#     "streaming": true
#   }' | while read line; do
#     if [[ $line == data:* && $line != "data: [DONE]" ]]; then
#       text=$(echo "$line" | sed -E 's/data: \{"text": "(.*)"\}/\1/')
#       if [[ -n "$text" ]]; then
#         echo -n "$text"
#       fi
#     fi
#   done && echo
