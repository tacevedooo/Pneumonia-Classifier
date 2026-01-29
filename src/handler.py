import io
import base64
import json
from io import BytesIO

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import runpod #type: ignore

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load model
# =========================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('models/pneumonia_classifier.pth', map_location=device))
model = model.to(device)
model.eval()

# =========================
# Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# Classes
# =========================
class_names = ['NORMAL', 'PNEUMONIA']

# =========================
# Input validation
# =========================
def validate_input(job_input):
    if job_input is None:
        return None, "Please provide an image"

    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON input"

    image_data = job_input.get("image")

    if not image_data or not isinstance(image_data, str):
        return None, "Image data must be a base64-encoded string"

    return {"image": image_data}, None

# =========================
# Handler
# =========================
def handler(job):
    job_input = job.get("input")
    validate_data, error_message = validate_input(job_input)

    if error_message:
        return {"error": error_message}

    image_base64 = validate_data["image"]

    try:
        # Decode base64
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')

        # Apply transforms
        image = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)

        predicted_class = class_names[preds.item()]

        return {"prediction": predicted_class}

    except (base64.binascii.Error, ValueError):
        return {"error": "Invalid base64-encoded image data"}
    except IOError:
        return {"error": "Unable to open image. Please ensure the image data is valid."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

# =========================
# Start serverless
# =========================
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
