"""Optional FastAPI wrapper (alternative to app.py's Streamlit UI).

Run with:  uvicorn api:app --reload
Then POST an image file to /caption
"""
import io
from fastapi import FastAPI, UploadFile, File
from PIL import Image

from src.predict import get_predictor

app = FastAPI(title="Image Caption Generator API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    predictor = get_predictor()
    caption_text = predictor.predict(image)
    return {"caption": caption_text}
