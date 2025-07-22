import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import random
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from torchvision.ops import nms

from models.face_model import get_face_detector

# ✅ 路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
IMG_DIR = BASE_DIR / "datasets" / "raw" / "images"
MODEL_PATH = BASE_DIR / "best_model.pt"

# ✅ 初始化模型（直接在这里加载）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_face_detector(num_classes=2).to(device)

# ✅ 加载权重（兼容 'model' 键）
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict["model"] if "model" in state_dict else state_dict)
model.eval()
print("✅ 模型加载完成")

# ✅ transform 与训练保持一致
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# ✅ 找图
image_files = list(IMG_DIR.rglob("*.[jJ][pP][gG]")) + list(IMG_DIR.rglob("*.[pP][nN][gG]"))
if not image_files:
    print("⚠️ 没有找到图片")
    exit()

sample_images = random.sample(image_files, min(3, len(image_files)))

for img_path in sample_images:
    image = Image.open(img_path).convert("RGB")
    orig_w, orig_h = image.size
    resized = image.resize((640, 640))
    img_tensor = transform(resized).to(device)

    with torch.no_grad():
        preds = model([img_tensor])[0]
        boxes = preds["boxes"].cpu()
        scores = preds["scores"].cpu()

    # 映射回原图尺寸
    scale_x = orig_w / 640
    scale_y = orig_h / 640
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    print(f"\n📷 {img_path.name}")
    for i, (box, score) in enumerate(zip(boxes, scores)):
        print(f"  ▶ Box {i+1}: {box.tolist()}, Score: {score.item():.4f}")

    # ✅ 显示
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box, score in zip(boxes, scores):
        if score >= 0.5:
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f"{score:.2f}", color='red', fontsize=10)
    plt.axis('off')
    plt.title(img_path.name)
    plt.show()
