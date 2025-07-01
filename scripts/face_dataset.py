import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torch

class FaceDataset(Dataset):
    def __init__(self, img_dir, annotation_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        # 加载标注数据
        with open(annotation_path, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)

        # 只保留至少有一个 box 的图片
        self.annotations = [ann for ann in self.annotations if ann["boxes"]]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.img_dir, self.annotations[idx]["filename"])
        image = Image.open(img_path).convert("RGB")

        # 图像变换
        if self.transform:
            image = self.transform(image)

        # 默认只取第一个人脸框（可按需扩展）
        box = self.annotations[idx]["boxes"][0]
        target = torch.tensor([box["x"], box["y"], box["w"], box["h"]], dtype=torch.float32)

        return image, target
