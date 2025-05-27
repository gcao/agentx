from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import base64
from typing import Optional, List
import json

app = FastAPI()

class ChatMessage(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 256
    tools: Optional[List[dict]] = None
    tool_choice: Optional[str] = None

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    # Process messages with memory context
    context = await memory_manager.get_context(request.messages)
    
    # Handle images if present
    images = []
    for msg in request.messages:
        if msg.images:
            for img_data in msg.images:
                # Decode base64 images
                images.append(decode_base64_image(img_data))
    
    # Generate response
    response = model.generate(
        context, 
        images[-1] if images else None,
        max_tokens=request.max_tokens
    )
    
    # Store in memory
    await memory_manager.store_interaction(request.messages, response)
    
    # Return OpenAI-compatible response
    return {
        "id": f"chatcmpl-{generate_id()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response
            },
            "finish_reason": "stop"
        }]
    }