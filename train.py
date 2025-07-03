import os
import torch
import torchvision
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
from models.face_model import get_face_detector
from scripts.face_dataset import FaceDetectionDataset
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()


from tqdm import tqdm

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # ✅ 根目录
    BASE_DIR = Path(__file__).resolve().parent
    CHECKPOINT_PATH = BASE_DIR / "checkpoint.pt"

    # ✅ 数据路径
    img_dir = BASE_DIR / "datasets" / "raw" / "images"
    ann_path = BASE_DIR / "datasets" / "annotations.json"

    # ✅ 数据增强
    '''transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])'''

    transform = transforms.ToTensor()


    MAX_SAMPLES = None  # 想测试多少张图像就写多少，None 表示全部使用

    dataset = FaceDetectionDataset(str(img_dir), str(ann_path), transform=transform)

    if MAX_SAMPLES is not None:
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(min(MAX_SAMPLES, len(dataset)))))
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # ✅ 初始化模型和优化器
    model = get_face_detector(num_classes=2).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.0001)
    start_epoch = 0

    # ✅ 如果有 checkpoint 就恢复
    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f"🔄 恢复模型，从第 {start_epoch} 轮继续训练")
    else:
        print("🚀 从头开始训练")

    # ✅ 开始训练
    epochs = 5
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, targets in progress_bar:
            try:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                with autocast():  # ✅ 开启混合精度
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += losses.item()
                progress_bar.set_postfix(loss=losses.item())

            except Exception as e:
                print(f"⚠️ 跳过异常批次: {e}")
                continue

        print(f"📘 Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

        # ✅ 保存 checkpoint
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1
        }, CHECKPOINT_PATH)

    print("✅ 模型训练完成并保存")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
