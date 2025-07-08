import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # ✅ 添加项目根目录到路径

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.amp import autocast, GradScaler
from torchvision.ops import nms
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import random
import json

from models.face_model import get_face_detector
from scripts.face_dataset import FaceDetectionDataset

def collate_fn(batch):
    return tuple(zip(*batch))

# ✅ 可视化预测结果（包含 NMS + 真实框）
def visualize_predictions(model, all_image_paths, device, annotation_path=None):
    if not all_image_paths:
        print("⚠️ 无测试图像，跳过可视化")
        return

    image_paths = random.sample(all_image_paths, min(3, len(all_image_paths)))

    annotations = {}
    if annotation_path and Path(annotation_path).exists():
        with open(annotation_path, "r") as f:
            data = json.load(f)
            annotations = {item["filename"]: item["boxes"] for item in data}

    model.eval()
    transform = ToTensor()
    fig, axs = plt.subplots(1, len(image_paths), figsize=(5 * len(image_paths), 5))
    if len(image_paths) == 1:
        axs = [axs]

    for idx, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).convert("RGB")
            tensor_img = transform(img).to(device)

            with torch.no_grad():
                prediction = model([tensor_img])[0]

            boxes = prediction['boxes'].cpu()
            scores = prediction['scores'].cpu()

            keep = nms(boxes, scores, iou_threshold=0.3)
            boxes = boxes[keep]
            scores = scores[keep]

            axs[idx].imshow(img)

            CONF_THRESH = 0.5
            for box, score in zip(boxes, scores):
                if score >= CONF_THRESH:
                    x1, y1, x2, y2 = box
                    axs[idx].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                     edgecolor='r', facecolor='none', linewidth=2))

            try:
                relative_path = str(img_path.relative_to(Path(__file__).resolve().parent / "datasets" / "raw" / "images")).replace("\\", "/")
                for gt in annotations.get(relative_path, []):
                    x, y, w, h = gt['x'], gt['y'], gt['w'], gt['h']
                    axs[idx].add_patch(plt.Rectangle((x, y), w, h,
                                                     edgecolor='blue', facecolor='none', linewidth=2))
            except Exception as e:
                print(f"⚠️ 跳过真实框标注加载失败: {e}")

            axs[idx].set_title(f"Image {idx+1}")
            axs[idx].axis('off')

        except Exception as e:
            print(f"⚠️ 跳过图像 {img_path.name}，原因: {e}")

    plt.tight_layout()
    plt.show()

def main():
    data_limit = None # 设置为 None 以使用所有数据，或设置为整数以限制样本数
    print("🚀 开始训练人脸检测模型...")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 使用设备：{device}")

    BASE_DIR = Path(__file__).resolve().parent.parent
    CHECK_DIR = BASE_DIR / "Checks"
    CHECK_DIR.mkdir(exist_ok=True)
    CHECKPOINT_PATH = BASE_DIR / "checkpoint.pt"
    BEST_MODEL_PATH = BASE_DIR / "best_model.pt"
    IMG_DIR = BASE_DIR / "datasets" / "raw" / "images"
    ANN_PATH = BASE_DIR / "datasets" / "annotations.json"

    all_images = list(IMG_DIR.rglob("*.jpg"))

    transform = ToTensor()
    full_dataset = FaceDetectionDataset(str(IMG_DIR), str(ANN_PATH), transform=transform, resize=(640, 640))
    dataset = full_dataset if data_limit is None else torch.utils.data.Subset(
        full_dataset, random.sample(range(len(full_dataset)), data_limit)
    )
    dataloader = DataLoader(dataset, 
                            batch_size=6, 
                            shuffle=True, 
                            num_workers=4,
                            pin_memory=True, 
                            collate_fn=collate_fn)

    model = get_face_detector(num_classes=2).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.0001)
    scaler = GradScaler(device='cuda')
    start_epoch = 0
    best_loss = float('inf')

    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f"\n🔄 继续训练：从第 {start_epoch} 轮恢复")

        # ✅ 如果存在 best model，加载其 loss
        if BEST_MODEL_PATH.exists():
            best_model_state = torch.load(BEST_MODEL_PATH, map_location=device)
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for images, targets in dataloader:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = model(images, targets)
                    val_loss += sum(loss for loss in loss_dict.values()).item()
                best_loss = val_loss
                print(f"✅ 已恢复历史最佳模型 Loss: {best_loss:.4f}")
    else:
        print("\n🆕 从头开始训练")

    print(f"📊 当前样本数: {len(dataset)}，每轮 batch 数: {len(dataloader)}\n")

    epochs = 100
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, targets in progress_bar:
            try:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                with autocast(device_type='cuda'):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += losses.item()
                progress_bar.set_postfix(loss=losses.item())

            except Exception as e:
                print(f"⚠️ 跳过异常 batch: {e}")
                continue

        print(f"📘 Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), CHECK_DIR / f"backup_epoch_{epoch+1}.pt")    

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1
        }, CHECKPOINT_PATH)

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"⭐ 保存最优模型（loss: {best_loss:.4f}）")

    print("\n✅ 训练完成并已保存模型")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
