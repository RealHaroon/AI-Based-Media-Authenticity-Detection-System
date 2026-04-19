from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from model_loader import load_model
from predictor import predict

# ── Globals ──────────────────────────────────────────────
model = None
class_names = []
img_size = 224
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png"}

# ── Lifespan (loads model once on startup) ────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names, img_size
    print("Loading model...")
    model, class_names, img_size = load_model()
    print(f"Model loaded. Classes: {class_names}")
    yield
    print("Shutting down.")

# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="Face Authenticity Detector API",
    description="Detects whether an uploaded face image is real or AI-generated.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": class_names
    }

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Only JPG and PNG are accepted."
        )

    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = predict(image_bytes, model, class_names, img_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return {
        "filename": file.filename,
        **result
    }