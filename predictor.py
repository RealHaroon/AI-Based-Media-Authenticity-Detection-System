import torch
from torchvision import transforms
from PIL import Image
import io

def get_transforms(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def predict(image_bytes: bytes, model, class_names: list, img_size: int) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tfms = get_transforms(img_size)
    x = tfms(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        prob_fake = torch.sigmoid(logits).item()

    # class_names from ImageFolder is alphabetical: ['fake', 'real']
    # prob_fake is probability of the FIRST class (fake = index 0)
    fake_idx = class_names.index("fake") if "fake" in class_names else 0
    real_idx = class_names.index("real") if "real" in class_names else 1

    if fake_idx == 0:
        prob_real = 1.0 - prob_fake
    else:
        prob_real = prob_fake
        prob_fake = 1.0 - prob_real

    if prob_fake >= 0.5:
        label = "AI-Generated"
        confidence = round(prob_fake * 100, 2)
        verdict = "⚠️ This image is likely AI-generated or synthetic."
    else:
        label = "Real"
        confidence = round(prob_real * 100, 2)
        verdict = "✅ This image appears to be real."

    return {
        "label": label,
        "confidence": confidence,
        "prob_fake": round(prob_fake * 100, 2),
        "prob_real": round(prob_real * 100, 2),
        "verdict": verdict
    }