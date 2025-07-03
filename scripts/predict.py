import os
import sys
import torch
import json
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms


# ✅ 项目路径
# ✅ 加上这两行让 Python 能找到 models 目录
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from models.face_model import get_face_detector

# ✅ 路径设置
ANNOTATION_PATH = BASE_DIR / "datasets" / "annotations.json"
IMG_BASE = BASE_DIR / "datasets" / "raw" / "images"
MODEL_PATH = BASE_DIR / "checkpoint.pt"

# ✅ 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_face_detector(num_classes=2)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# ✅ 图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ✅ 显示图像带框（蓝色：真实框，红色：预测框）
def visualize(image, gt_boxes, pred_boxes, pred_scores):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # 画真实框（蓝色）
    for box in gt_boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

    # 画预测框（红色）
    for box, score in zip(pred_boxes, pred_scores):
        if score >= 0.5:
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{score:.2f}", color='red', fontsize=8)

    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    with open(ANNOTATION_PATH, "r") as f:
        data = json.load(f)

    random.shuffle(data)
    selected = data[:3]

    for item in selected:
        rel_path = item["filename"]
        gt_boxes = item["boxes"]
        img_path = IMG_BASE / rel_path

        if not img_path.exists():
            print(f"❌ 图像不存在: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        resized = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(resized)[0]
            pred_boxes = preds["boxes"].cpu()
            pred_scores = preds["scores"].cpu()

        # 🔍 显示图像 + 真实框 + 预测框
        visualize(image, gt_boxes, pred_boxes, pred_scores)
