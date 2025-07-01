from pathlib import Path
import os
import json
from shutil import copy2
from PIL import Image

# === 路径设置 ===
BASE_DIR = Path(__file__).resolve().parent.parent
img_root = BASE_DIR / "datasets/raw/images"
label_file = BASE_DIR / "datasets/raw/wider_face_train_bbx_gt.txt"
output_img_dir = BASE_DIR / "datasets/images_singleface"  # 只保留单人脸图
output_json = BASE_DIR / "datasets/annotations_singleface.json"
max_images = 10000

# === 创建输出目录 ===
output_img_dir.mkdir(parents=True, exist_ok=True)

annotations = []
i = 0
parsed = 0

# === 读取标签文件 ===
with open(label_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

while i < len(lines) and parsed < max_images:
    rel_path = lines[i].strip()
    i += 1

    if i >= len(lines):
        break
    try:
        box_count = int(lines[i].strip())
    except ValueError:
        print(f"⚠️ 跳过非数字框计数行：{lines[i].strip()}")
        continue
    i += 1

    boxes = []
    for _ in range(box_count):
        if i >= len(lines):
            break
        parts = lines[i].strip().split()
        if len(parts) >= 4:
            try:
                x, y, w, h = map(int, parts[:4])
                if w > 0 and h > 0:
                    boxes.append({"x": x, "y": y, "w": w, "h": h})
            except ValueError:
                pass
        i += 1

    # ✅ 只保留 box 数量为 1 的
    if len(boxes) != 1:
        continue

    filename = Path(rel_path).name
    src = img_root / rel_path
    dst = output_img_dir / filename

    if not src.exists():
        print(f"⚠️ 原图不存在：{src}")
        continue

    try:
        Image.open(src).verify()
        copy2(src, dst)
        annotations.append({
            "filename": filename,
            "boxes": boxes
        })
        parsed += 1
    except Exception as e:
        print(f"⚠️ 跳过损坏图片 {filename}: {e}")

# === 保存标注 ===
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(annotations, f, indent=2)

print(f"✅ 已保存 {parsed} 张单人脸图像到 {output_img_dir}")
print(f"✅ 标注保存为：{output_json}")
