from pathlib import Path
import json
from shutil import copy2
from PIL import Image

# ✅ 路径设置
BASE_DIR = Path(__file__).resolve().parent.parent
raw_img_dir = BASE_DIR / "datasets/raw/images"
annotation_file = BASE_DIR / "datasets/annotations.json"
output_img_dir = BASE_DIR / "datasets/images_singleface"
output_json = BASE_DIR / "datasets/annotations_singleface.json"

# ✅ 创建输出目录
output_img_dir.mkdir(parents=True, exist_ok=True)

# ✅ 加载原始标注
with open(annotation_file, "r", encoding="utf-8") as f:
    all_data = json.load(f)

# ✅ 筛选：只保留 boxes 长度为 1 的样本
filtered = []
for item in all_data:
    if len(item["boxes"]) == 1:
        src_path = raw_img_dir / item["filename"]
        dst_path = output_img_dir / item["filename"]
        if src_path.exists():
            try:
                # 可选：尝试打开以确保图像没损坏
                Image.open(src_path).verify()
                copy2(src_path, dst_path)
                filtered.append(item)
            except Exception as e:
                print(f"跳过损坏图像：{item['filename']}，错误：{e}")

# ✅ 保存标注
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2)

print(f"✅ 共复制 {len(filtered)} 张单人脸图像")
print(f"✅ 标注文件已生成：{output_json}")
