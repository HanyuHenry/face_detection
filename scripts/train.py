
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import random
import json
from models.face_model import get_face_detector
from scripts.face_dataset import FaceDetectionDataset

def collate_fn(batch):
    return tuple(zip(*batch))

# ✅ 保存完整 checkpoint（用于续训）
def save_checkpoint(model, optimizer, epoch, best_loss, path):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': best_loss
    }, path)

# ✅ 保存最简 best model（用于推理）
def save_best_model(model, best_loss, path):
    torch.save({
        'model': model.state_dict(),
        'loss': best_loss
    }, path)

def main():
    data_limit = None
    print("\n🚀 开始训练人脸检测模型...")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备：{device}")

    BASE_DIR = Path(__file__).resolve().parent.parent
    CHECK_DIR = BASE_DIR / "Checks"
    CHECK_DIR.mkdir(exist_ok=True)
    CHECKPOINT_PATH = BASE_DIR / "checkpoint.pt"
    BEST_MODEL_PATH = BASE_DIR / "best_model.pt"
    IMG_DIR = BASE_DIR / "datasets" / "raw" / "images"
    ANN_PATH = BASE_DIR / "datasets" / "annotations.json"

    transform = ToTensor()
    full_dataset = FaceDetectionDataset(str(IMG_DIR), str(ANN_PATH), transform=transform, resize=(640, 640))
    dataset = full_dataset if data_limit is None else torch.utils.data.Subset(
        full_dataset, random.sample(range(len(full_dataset)), data_limit))

    dataloader = DataLoader(dataset,
                            batch_size=6,
                            shuffle=True,
                            num_workers=0,
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
        start_epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('loss', float('inf'))
        print(f"\n🔄 继续训练：从第 {start_epoch} 轮恢复，best_loss: {best_loss:.4f}")
    elif BEST_MODEL_PATH.exists():
        best_state = torch.load(BEST_MODEL_PATH, map_location=device)
        best_loss = best_state.get('loss', float('inf'))
        print(f"📦 发现 best_model.pt，历史 best_loss: {best_loss:.4f}")

    print(f"📊 当前样本数: {len(dataset)}，每轮 batch 数: {len(dataloader)}\n")

    epochs = 100
    for epoch in range(start_epoch, epochs):
        print(f"\n🚩 开始第 {epoch+1} 轮训练")
        model.train()
        total_loss = 0.0
        total_samples = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_i, (images, targets) in enumerate(progress_bar):
            try:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += losses.item() * len(images)
                total_samples += len(images)
                progress_bar.set_postfix(loss=losses.item())
            except Exception as e:
                print(f"⚠️ 跳过异常 batch {batch_i+1}: {e}")
                continue

        avg_loss = total_loss / total_samples
        print(f"📘 Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), CHECK_DIR / f"backup_epoch_{epoch+1}.pt")
        save_checkpoint(model, optimizer, epoch + 1, best_loss, CHECKPOINT_PATH)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_best_model(model, best_loss, BEST_MODEL_PATH)
            print(f"⭐ 保存最优模型（Avg Loss: {best_loss:.4f}）")

    print("\n✅ 训练完成并已保存模型")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
import torch