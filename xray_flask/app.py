import base64
from io import BytesIO

from flask import Flask, render_template, request
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

# =========================
# Config
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = Flask(__name__)

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
# Routes
# =========================
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    original_image = None

    if request.method == 'POST':
        file = request.files['image']
        # Mostrar imagen en el navegador
        original_image = "data:image/jpeg;base64," + base64.b64encode(file.read()).decode('utf-8')
        
        # Reiniciar puntero
        file.seek(0)
        image = Image.open(file).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Predicción
        with torch.no_grad():
            outputs = model(image_tensor)
            _, preds = torch.max(outputs, 1)
            prediction = class_names[preds.item()]

    return render_template('index.html', original_image=original_image, prediction=prediction)

# =========================
# Run Flask
# =========================
if __name__ == "__main__":
    app.run(debug=True)
