import torch
from torchvision import models
import torch.nn as nn
from pathlib import Path

MODEL_PATH = Path("models/face_authenticity_detector_full.pth")

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    model_name = checkpoint.get("model_name", "efficientnet_b0")
    class_names = checkpoint["class_names"]
    img_size = checkpoint["img_size"]

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 1)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names, img_size