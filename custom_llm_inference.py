"""custom_LLM_inference.py - Run on EC2 with: uvicorn custom_LLM_inference:app --host 0.0.0.0 --port 8000"""
import json
from typing import List, Optional, Dict
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

# Default system prompt to control behavior
DEFAULT_SYSTEM_PROMPT = """You are a helpful voice assistant. Provide direct, concise answers to user questions. Do not ask follow-up questions unless specifically requested. Keep responses conversational and natural for voice output."""

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

class Message(BaseModel):
    role: str  # "system", "user", or "assistant"
    content: str

class GenerationRequest(BaseModel):
    messages: Optional[List[Message]] = None  # For chat-style requests
    input_texts: Optional[List[str]] = None  # For backward compatibility
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 1.0
    num_beams: Optional[int] = 1
    streaming: Optional[bool] = False
    enable_tts: Optional[bool] = True
    stop_sequences: Optional[List[str]] = ["?", "\n\n"]  # Stop if model asks questions

def format_chat_prompt(messages: List[Message], system_prompt: str = None) -> str:
    """Format messages into a single prompt string with proper structure."""
    prompt_parts = []
    
    # Add system prompt
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    prompt_parts.append(f"System: {sys_prompt}\n")
    
    # Add conversation history
    for msg in messages:
        if msg.role == "user":
            prompt_parts.append(f"User: {msg.content}\n")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}\n")
    
    # Add final assistant prompt
    prompt_parts.append("Assistant:")
    
    return "".join(prompt_parts)

async def send_to_tts_buffer(text: str):
    if not text or not text.strip():
        return False
    try:
        response = await http_client.post(f"{TTS_SERVER_URL}/synthesize", params={"text": text.strip()})
        return response.status_code == 200
    except Exception as e:
        print(f"TTS error: {e}")
        return False

def check_stop_sequence(text: str, stop_sequences: List[str]) -> bool:
    """Check if text contains any stop sequences."""
    if not stop_sequences:
        return False
    return any(seq in text for seq in stop_sequences)

def clear_model_state():
    """Clear any residual state from the model runner."""
    global runner
    if runner is not None:
        try:
            # Force clear KV cache and reset internal states
            runner.session.reset()
        except AttributeError:
            # If reset() doesn't exist, try to clear context
            pass

@app.post("/generate")
async def generate(request: GenerationRequest):
    if tokenizer is None or runner is None:
        raise HTTPException(status_code=503, detail="Model loading")
    
    # Clear any residual state from previous generations
    clear_model_state()
    
    # Prepare input texts
    if request.messages:
        # New chat-style format
        input_texts = [format_chat_prompt(request.messages, request.system_prompt)]
    elif request.input_texts:
        # Backward compatibility - but add system prompt
        if request.system_prompt:
            input_texts = [f"System: {request.system_prompt}\n\nUser: {text}\n\nAssistant:" for text in request.input_texts]
        else:
            input_texts = [f"System: {DEFAULT_SYSTEM_PROMPT}\n\nUser: {text}\n\nAssistant:" for text in request.input_texts]
    else:
        raise HTTPException(status_code=400, detail="Either 'messages' or 'input_texts' required")
    
    batch_input_ids = []
    for text in input_texts:
        input_ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=MAX_INPUT_LENGTH)
        batch_input_ids.append(torch.tensor(input_ids, dtype=torch.int32))
    
    # Ensure we're using fresh context for each generation
    input_lengths = [len(ids) for ids in batch_input_ids]

    if request.streaming:
        async def token_stream():
            try:
                with torch.no_grad():
                    gen = runner.generate(
                        batch_input_ids=batch_input_ids, 
                        max_new_tokens=MAX_OUTPUT_LEN,
                        temperature=request.temperature, 
                        top_k=request.top_k, 
                        top_p=request.top_p,
                        num_beams=request.num_beams, 
                        end_id=end_id, 
                        pad_id=pad_id,
                        streaming=True, 
                        return_dict=True,
                        input_lengths=input_lengths  # Explicitly pass input lengths
                    )
                    decoded_so_far = ""
                    sentence_buffer = ""
                    should_stop = False
                    
                    for step in gen:
                        if should_stop:
                            break
                            
                        ids_all = step["output_ids"][0][0].tolist()
                        gen_ids = ids_all[len(batch_input_ids[0]):]
                        current_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                        
                        if len(current_text) > len(decoded_so_far):
                            delta = current_text[len(decoded_so_far):]
                            decoded_so_far = current_text
                            sentence_buffer += delta
                            
                            # Check for stop sequences
                            if check_stop_sequence(current_text, request.stop_sequences):
                                # Send what we have without the question
                                clean_text = current_text
                                for seq in request.stop_sequences or []:
                                    if seq in clean_text:
                                        clean_text = clean_text.split(seq)[0]
                                        break
                                if clean_text.strip():
                                    yield f'data: {json.dumps({"text": clean_text[len(decoded_so_far) - len(delta):].strip()})}\n\n'
                                should_stop = True
                                break
                            
                            yield f'data: {json.dumps({"text": delta})}\n\n'
                            
                            if request.enable_tts:
                                if any(p in sentence_buffer for p in (".", "!", "\n")) or len(sentence_buffer) >= 80:
                                    # Don't send questions to TTS
                                    if not check_stop_sequence(sentence_buffer, request.stop_sequences):
                                        await send_to_tts_buffer(sentence_buffer.strip())
                                    sentence_buffer = ""
                    
                    if request.enable_tts and sentence_buffer.strip():
                        if not check_stop_sequence(sentence_buffer, request.stop_sequences):
                            await send_to_tts_buffer(sentence_buffer.strip())
                            
            except Exception as e:
                yield f'data: {json.dumps({"error": str(e)})}\n\n'
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(token_stream(), media_type="text/event-stream")
    
    # Non-streaming
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
            input_lengths=input_lengths  # Explicitly pass input lengths
        )
    
    output_texts = []
    for i, ids_tensor in enumerate(outputs["output_ids"]):
        ids = ids_tensor[0][len(batch_input_ids[i]):].tolist()
        text_out = tokenizer.decode(ids, skip_special_tokens=True)
        
        # Check for stop sequences and truncate
        if request.stop_sequences:
            for seq in request.stop_sequences:
                if seq in text_out:
                    text_out = text_out.split(seq)[0]
                    break
        
        output_texts.append(text_out)
        
        if request.enable_tts and text_out.strip():
            await send_to_tts_buffer(text_out.strip())
    
    return {"outputs": output_texts}

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": runner is not None}

@app.post("/reset")
async def reset_model_state():
    """Manually reset model state to clear any residual context."""
    clear_model_state()
    return {"status": "reset_complete"}