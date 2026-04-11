import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io, os

app = Flask(__name__)

# ── Models ──────────────────────────────────────────────────

class ANN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 784)
        return self.layer2(self.relu(self.layer1(x)))

class ANN_CAT_DOG(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12288, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(2048, 1024),  nn.BatchNorm1d(1024),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 256),   nn.BatchNorm1d(256),   nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2),
        )
    def forward(self, x):
        return self.model(x)

# ── Load weights ─────────────────────────────────────────────

BASE   = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mnist_model = ANN_MNIST()
mnist_model.load_state_dict(torch.load(os.path.join(BASE, "ANN_MNIST.pth"), map_location=device))
mnist_model.eval().to(device)

catdog_model = ANN_CAT_DOG()
catdog_model.load_state_dict(torch.load(os.path.join(BASE, "ann_cat_and_dog_model.pth"), map_location=device))
catdog_model.eval().to(device)

# ── Transforms ───────────────────────────────────────────────

mnist_tf = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),          # đảo màu: chữ đen/nền trắng → chữ sáng/nền đen (chuẩn MNIST)
    transforms.Normalize((0.5,), (0.5,)),
])
catdog_tf = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3),
])

MNIST_LABELS  = ["0","1","2","3","4","5","6","7","8","9"]
CATDOG_LABELS = ["Meo", "Cho"]


# ── Routes ───────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Khong co anh."}), 400
    file = request.files["image"]
    model_name = request.form.get("model", "mnist")
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except:
        return jsonify({"error": "Khong doc duoc anh."}), 400

    try:
        if model_name == "mnist":
            t = mnist_tf(image).unsqueeze(0).to(device)
            with torch.no_grad(): out = mnist_model(t)
            probs = torch.softmax(out, 1)[0]
            top = torch.topk(probs, 1)
            label = MNIST_LABELS[top.indices[0].item()]
        else:
            t = catdog_tf(image).unsqueeze(0).to(device)
            with torch.no_grad(): out = catdog_model(t)
            probs = torch.softmax(out, 1)[0]
            top = torch.topk(probs, 1)
            label = CATDOG_LABELS[top.indices[0].item()]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"label": label})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
