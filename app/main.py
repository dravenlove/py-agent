from fastapi import FastAPI

app = FastAPI(title="AI Agent 30D", version="0.1.0")

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "day": 1}
