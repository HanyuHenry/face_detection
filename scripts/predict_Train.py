import os
import sys
import json
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ✅ 添加根路径
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# ✅ 导入封装类
from scripts.inference import FaceDetector

# ✅ 路径设置
ANNOTATION_PATH = BASE_DIR / "datasets" / "annotations.json"
IMG_BASE = BASE_DIR / "datasets" / "raw" / "images"
MODEL_PATH = BASE_DIR / "best_model.pt"

# ✅ 初始化模型
detector = FaceDetector()
detector.load_weights(MODEL_PATH)

# ✅ 显示图像带框（蓝色：真实框，红色：预测框）
def visualize(pil_image, gt_boxes, pred_boxes, pred_scores, conf_thresh=0.1):
    fig, ax = plt.subplots(1)
    ax.imshow(pil_image)

    for box in gt_boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

    for box, score in zip(pred_boxes, pred_scores):
        if score >= conf_thresh:
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{score:.2f}", color='red', fontsize=8)

    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ✅ 主程序
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

        # ✅ 使用封装预测
        pred_boxes, pred_scores = detector.predict(image)

        print(f"\n📷 图像: {rel_path}")
        for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
            print(f"  ▶ Box {i+1}: {box.tolist()}, Score: {score.item():.4f}")

        visualize(image, gt_boxes, pred_boxes, pred_scores)
