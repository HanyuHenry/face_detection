import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TXT_PATH = BASE_DIR / "datasets" / "raw" / "wider_face_train_bbx_gt.txt"
IMG_DIR = BASE_DIR / "datasets" / "raw" / "images"
OUTPUT_PATH = BASE_DIR / "datasets" / "annotations.json"

def prepare_annotations(txt_path, img_dir, output_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    data = []
    i = 0
    missing = 0
    skipped = 0

    while i < len(lines):
        filename = lines[i].strip()
        if not filename.endswith(".jpg"):
            i += 1
            continue

        i += 1
        if i >= len(lines):
            break

        try:
            num_faces = int(lines[i].strip())
        except ValueError:
            print(f"⚠️  跳过无效记录（无法解析人脸数量）: {filename}")
            skipped += 1
            continue

        i += 1
        boxes = []

        for _ in range(num_faces):
            if i >= len(lines): break
            parts = lines[i].strip().split()
            if len(parts) >= 4:
                try:
                    x, y, w, h = map(int, parts[:4])
                    if w > 10 and h > 10:
                        boxes.append({"x": x, "y": y, "w": w, "h": h})
                except:
                    pass
            i += 1

        img_path = IMG_DIR / filename
        if img_path.exists() and boxes:
            data.append({"filename": filename, "boxes": boxes})
        elif not img_path.exists():
            missing += 1
            print(f"❌ 图像不存在，跳过: {filename}")

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ 转换完成")
    print(f"📊 有效图像: {len(data)}")
    print(f"❌ 缺失图像: {missing}")
    print(f"⚠️ 格式跳过: {skipped}")
    print(f"📄 输出路径: {output_path}")

if __name__ == "__main__":
    prepare_annotations(TXT_PATH, IMG_DIR, OUTPUT_PATH)
