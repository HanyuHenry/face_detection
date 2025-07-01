# models/face_model.py
import torch.nn as nn
import torchvision.models as models

class FaceBoxRegressor(nn.Module):
    def __init__(self):
        super(FaceBoxRegressor, self).__init__()
        # 加载预训练的 ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 替换最后的全连接层，输出为 4 个值（x, y, w, h）
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 4)

    def forward(self, x):
        return self.backbone(x)
