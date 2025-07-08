from PIL import Image
import os
import torch
from torch.utils.data import Dataset
import json
import torchvision.transforms as T

class FaceDetectionDataset(Dataset):
    def __init__(self, img_dir, annotation_path, transform=None, resize=(640, 640)):
        self.img_dir = img_dir
        self.transform = transform
        self.resize = resize

        with open(annotation_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item["filename"])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在: {img_path}")

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        new_w, new_h = self.resize

        image = image.resize((new_w, new_h))

        boxes = []
        for box in item["boxes"]:
            x1 = box["x"] * new_w / orig_w
            y1 = box["y"] * new_h / orig_h
            x2 = (box["x"] + box["w"]) * new_w / orig_w
            y2 = (box["y"] + box["h"]) * new_h / orig_h
            boxes.append([x1, y1, x2, y2])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, target
