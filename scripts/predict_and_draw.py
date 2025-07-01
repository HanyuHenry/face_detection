import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import json
from models.face_model import FaceBoxRegressor

BASE_DIR = Path(__file__).resolve().parent.parent

# ✅ 预测使用原始大小图像
img_dir = BASE_DIR / "datasets/images_Prepared"
annotation_path = BASE_DIR / "datasets/annotations.json"
model_path = BASE_DIR / "face_box_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceBoxRegressor().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.ToTensor()

with open(annotation_path, "r") as f:
    data = json.load(f)

samples = random.sample(data, 5)
for item in samples:
    img_path = img_dir / item["filename"]
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)[0].cpu().numpy()

    draw = ImageDraw.Draw(image)
    x, y, w, h = pred
    draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

    for box in item["boxes"]:
        gx, gy, gw, gh = box["x"], box["y"], box["w"], box["h"]
        draw.rectangle([gx, gy, gx + gw, gy + gh], outline="green", width=2)

    plt.imshow(image)
    plt.axis("off")
    plt.title(item["filename"])
    plt.show()

    print("预测框:", pred)
    print("真实框:", item["boxes"])
