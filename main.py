import io
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates


# =========================
# Config
# =========================
MODEL_PATH = Path("models/face_authenticity_detector_full.pth")
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_FILE_SIZE_MB = 10

model = None
class_names = None
img_size = 224


# Hellow
# =========================
# Model helpers
# =========================
def build_model():
    m = models.efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 1)
    )
    return m


def load_model():
    global model, class_names, img_size

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_names = checkpoint.get("class_names", ["fake", "real"])
    img_size = checkpoint.get("img_size", 224)


def get_transforms(size: int):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict_image_bytes(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = get_transforms(img_size)(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        prob_real = torch.sigmoid(logits).item()
        prob_fake = 1.0 - prob_real

    # Conservative thresholds
    if prob_real >= 0.60:
        label = "Real"
        confidence = prob_real
        verdict = "This image is likely real."
    elif prob_fake >= 0.60:
        label = "AI-Generated"
        confidence = prob_fake
        verdict = "This image is likely AI-generated."
    else:
        label = "Uncertain"
        confidence = max(prob_real, prob_fake)
        verdict = "The model is not confident enough to decide."

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "prob_real": round(prob_real * 100, 2),
        "prob_fake": round(prob_fake * 100, 2),
        "verdict": verdict
    }


# =========================
# FastAPI app
# =========================
templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="AI-Based Media Authenticity Detection System",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "classes": class_names,
        "img_size": img_size
    }


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, PNG, and WEBP files are allowed."
        )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max allowed size is {MAX_FILE_SIZE_MB} MB."
        )

    try:
        result = predict_image_bytes(content)
        return JSONResponse({
            "filename": file.filename,
            **result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")