from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import httpx
import asyncio

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    message = data.get("message")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:8000/api/chat",
                json={"message": message},
                timeout=30.0
            )
            response.raise_for_status()
            return HTMLResponse(
                f'<div class="user-message">{message}</div>'
                f'<div class="bot-message">{response.json()["response"]}</div>'
            )
        except httpx.HTTPStatusError as e:
            return JSONResponse(
                {"error": f"Backend returned {e.response.status_code}: {e.response.text}"},
                status_code=502
            )
        except httpx.RequestError as e:
            return JSONResponse(
                {"error": f"Connection error: {str(e)}"},
                status_code=503
            )
        except Exception as e:
            return JSONResponse(
                {"error": f"Unexpected error: {str(e)}"},
                status_code=500
            )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
