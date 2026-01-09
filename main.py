import os
import base64
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# ✅ CORS MUST BE HERE — BEFORE ROUTES
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RemixRequest(BaseModel):
    image_url: str
    prompt: str
    mask_type: str = "surface"  # background | surface | logo_text

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/remix")
def remix(req: RemixRequest):
    return {
        "status": "done",
        "output_image_url": req.image_url,
        "echo_prompt": req.prompt,
        "mask_type": req.mask_type
    }

    data = {
        "model": "gpt-image-1",
        "prompt": req.prompt,
        "size": "1024x1024",
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    r = requests.post(url, headers=headers, files=files, data=data, timeout=120)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {r.text}")

    payload = r.json()

    # 4) Return base64 so Shopify can display it immediately (no storage needed for MVP)
    # The response includes image data; we convert it to a data URL.
    b64 = payload["data"][0].get("b64_json")
    if not b64:
        raise HTTPException(status_code=500, detail="No b64_json returned")

    data_url = "data:image/png;base64," + b64
    return {"status": "done", "output_image_url": data_url}
