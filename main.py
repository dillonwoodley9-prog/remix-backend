import os
import base64
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

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
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY on server")

    if not req.image_url or not req.prompt:
        raise HTTPException(status_code=400, detail="image_url and prompt are required")

    if len(req.prompt) > 300:
        raise HTTPException(status_code=400, detail="Prompt too long (max 300 chars for MVP)")

    if req.mask_type not in ("background", "surface", "logo_text"):
        raise HTTPException(status_code=400, detail="Invalid mask_type")

    # 1) Download the product image bytes
    img_resp = requests.get(req.image_url, timeout=20)
    if img_resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Could not download image_url")

    image_bytes = img_resp.content

    # 2) Load preset mask bytes from your repo (Phase 1: store masks on backend)
    # Put your 3 mask files in a folder named "masks/"
    mask_path = f"masks/{req.mask_type}.png"
    if not os.path.exists(mask_path):
        raise HTTPException(status_code=500, detail=f"Mask file missing on server: {mask_path}")

    with open(mask_path, "rb") as f:
        mask_bytes = f.read()

    # 3) Call OpenAI Images Edits endpoint (multipart form-data)
    # Docs: POST https://api.openai.com/v1/images/edits
    # image is required, prompt is required, mask is optional but we use it
    # Supported models include GPT Image models (e.g., gpt-image-1) and dall-e-2
    # Ref: OpenAI Images API Reference
    url = "https://api.openai.com/v1/images/edits"

    files = {
        "image": ("image.png", image_bytes, "image/png"),
        "mask": ("mask.png", mask_bytes, "image/png"),
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
