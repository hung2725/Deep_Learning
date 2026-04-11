import os
import io
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from torchvision import transforms
from PIL import Image

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- ĐỊNH NGHĨA CÁC KIẾN TRÚC MẠNG ---
class CatDog_CNN(nn.Module):
    def __init__(self):
        super(CatDog_CNN, self).__init__()
        self.conv1a, self.bn1a = nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32)
        self.conv1b, self.bn1b = nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2a, self.bn2a = nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64)
        self.conv2b, self.bn2b = nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3a, self.bn3a = nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128)
        self.conv3b, self.bn3b = nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4a, self.bn4a = nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256)
        self.conv4b, self.bn4b = nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = torch.relu(self.bn1a(self.conv1a(x)))
        x = self.pool1(torch.relu(self.bn1b(self.conv1b(x))))
        x = torch.relu(self.bn2a(self.conv2a(x)))
        x = self.pool2(torch.relu(self.bn2b(self.conv2b(x))))
        x = torch.relu(self.bn3a(self.conv3a(x)))
        x = self.pool3(torch.relu(self.bn3b(self.conv3b(x))))
        x = torch.relu(self.bn4a(self.conv4a(x)))
        x = self.pool4(torch.relu(self.bn4b(self.conv4b(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.conv1a, self.bn1a = nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32)
        self.conv1b, self.bn1b = nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2a, self.bn2a = nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64)
        self.conv2b, self.bn2b = nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3a, self.bn3a = nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128)
        self.conv3b, self.bn3b = nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.bn1a(self.conv1a(x)))
        x = self.pool1(torch.relu(self.bn1b(self.conv1b(x))))
        x = torch.relu(self.bn2a(self.conv2a(x)))
        x = self.pool2(torch.relu(self.bn2b(self.conv2b(x))))
        x = torch.relu(self.bn3a(self.conv3a(x)))
        x = self.pool3(torch.relu(self.bn3b(self.conv3b(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class PlantVillage_CNN(nn.Module):
    def __init__(self, num_classes=38):
        super(PlantVillage_CNN, self).__init__()
        self.conv1a, self.bn1a = nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32)
        self.conv1b, self.bn1b = nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2a, self.bn2a = nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64)
        self.conv2b, self.bn2b = nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3a, self.bn3a = nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128)
        self.conv3b, self.bn3b = nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4a, self.bn4a = nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256)
        self.conv4b, self.bn4b = nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1a(self.conv1a(x)))
        x = self.pool1(torch.relu(self.bn1b(self.conv1b(x))))
        x = torch.relu(self.bn2a(self.conv2a(x)))
        x = self.pool2(torch.relu(self.bn2b(self.conv2b(x))))
        x = torch.relu(self.bn3a(self.conv3a(x)))
        x = self.pool3(torch.relu(self.bn3b(self.conv3b(x))))
        x = torch.relu(self.bn4a(self.conv4a(x)))
        x = self.pool4(torch.relu(self.bn4b(self.conv4b(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# --- KHAI BÁO NHÃN PLANTVILLAGE (38 LỚP THEO THỨ TỰ ALPHABET CỦA THƯ MỤC) ---
PLANT_VILLAGE_LABELS = [
    'Táo - Bệnh ghẻ (vảy)', 
    'Táo - Bệnh thối đen', 
    'Táo - Bệnh gỉ sắt', 
    'Táo - Khỏe mạnh',
    'Việt quất - Khỏe mạnh', 
    'Anh đào - Bệnh phấn trắng', 
    'Anh đào - Khỏe mạnh',
    'Ngô (Bắp) - Bệnh đốm lá xám', 
    'Ngô (Bắp) - Bệnh gỉ sắt thông thường',
    'Ngô (Bắp) - Bệnh cháy lá sọc lớn', 
    'Ngô (Bắp) - Khỏe mạnh', 
    'Nho - Bệnh thối đen',
    'Nho - Bệnh sởi đen (Esca)', 
    'Nho - Bệnh cháy lá', 
    'Nho - Khỏe mạnh',
    'Cam - Bệnh vàng lá gân xanh (Greening)', 
    'Đào - Bệnh đốm vi khuẩn', 
    'Đào - Khỏe mạnh',
    'Ớt chuông - Bệnh đốm vi khuẩn', 
    'Ớt chuông - Khỏe mạnh', 
    'Khoai tây - Bệnh mốc sương sớm',
    'Khoai tây - Bệnh sương mai', 
    'Khoai tây - Khỏe mạnh', 
    'Mâm xôi - Khỏe mạnh', 
    'Đậu nành - Khỏe mạnh',
    'Bí - Bệnh phấn trắng', 
    'Dâu tây - Bệnh cháy lá', 
    'Dâu tây - Khỏe mạnh',
    'Cà chua - Bệnh đốm vi khuẩn', 
    'Cà chua - Bệnh mốc sương sớm', 
    'Cà chua - Bệnh sương mai', 
    'Cà chua - Bệnh nấm lá',
    'Cà chua - Bệnh đốm lá Septoria', 
    'Cà chua - Nhện đỏ (Nhện hai đốm)', 
    'Cà chua - Bệnh đốm vòng tròn',
    'Cà chua - Virus xoăn vàng lá', 
    'Cà chua - Virus khảm', 
    'Cà chua - Khỏe mạnh'
]
# --- CẤU HÌNH MÔ HÌNH ---
CONFIGS = {
    'catdog': {'model': CatDog_CNN, 'file': 'cat_and_dog_model.pth', 'size': 128, 'labels': ['Mèo', 'Chó']},
    'cifar10': {'model': CIFAR10_CNN, 'file': 'CIFAR10_model.pth', 'size': 32, 'labels': ['Máy bay', 'Ô tô', 'Chim', 'Mèo', 'Nai', 'Chó', 'Ếch', 'Ngựa', 'Tàu', 'Xe tải']},
    'plant': {'model': PlantVillage_CNN, 'file': 'PlantVillage_model.pth', 'size': 128, 'labels': PLANT_VILLAGE_LABELS}
}

loaded_models = {}

def load_all():
    for k, v in CONFIGS.items():
        if os.path.exists(v['file']):
            try:
                m = v['model']().to(device)
                m.load_state_dict(torch.load(v['file'], map_location=device, weights_only=True))
                m.eval()
                loaded_models[k] = m
                print(f"Loaded: {v['file']}")
            except Exception as e: print(f"Error {k}: {e}")

load_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    mid = request.form.get('model_id')
    file = request.files.get('file')
    if mid not in loaded_models or not file: return jsonify({'error': 'Invalid data'}), 400
    try:
        cfg = CONFIGS[mid]
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((cfg['size'], cfg['size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = loaded_models[mid](tensor)
            prob = torch.nn.functional.softmax(out, dim=1)
            conf, pred = torch.max(prob, 1)
        return jsonify({'prediction': cfg['labels'][pred.item()], 'confidence': f"{conf.item()*100:.2f}%"})
    except Exception as e: return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5007)