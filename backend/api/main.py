from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "API server is running"}

@app.get("/test")
async def test_endpoint():
    return {"test": "success", "details": "API server is working correctly"}
