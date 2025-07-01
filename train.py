import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision.transforms as transforms

from scripts.face_dataset import FaceDataset
from models.face_model import FaceBoxRegressor

def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # ✅ 根目录
    BASE_DIR = Path(__file__).resolve().parent

    # ✅ 路径配置

    model_path = BASE_DIR / "face_box_model.pt"
    epoch_file = BASE_DIR / "last_epoch.txt"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    '''dataset = FaceDataset(
        img_dir="datasets/images_Prepared",  # 原始图像路径（未压缩）
        annotation_path="datasets/annotations.json",
        transform=transform
    )'''

    dataset = FaceDataset(
        img_dir="datasets/images_singleface",
        annotation_path="datasets/annotations_singleface.json",
        transform=transform
    )


    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    # ✅ 模型初始化
    model = FaceBoxRegressor().to(device)
    start_epoch = 0

    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
        if epoch_file.exists():
            with open(epoch_file, "r") as f:
                start_epoch = int(f.read().strip())
        print(f"🔄 已加载模型权重，继续从第 {start_epoch + 1} 轮训练...")
    else:
        print("🚀 从头开始训练")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ✅ 训练循环
    epochs = 20
    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        model.train()
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

        torch.save(model.state_dict(), model_path)
        with open(epoch_file, "w") as f:
            f.write(str(epoch + 1))

    print("✅ 模型训练完成并保存。")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
