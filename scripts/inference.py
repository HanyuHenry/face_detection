# scripts/inference.py
import torch
from torchvision import transforms
from PIL import Image
from models.face_model import get_face_detector

class FaceDetector:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_face_detector(num_classes=2).to(self.device)
        self.model.eval()
        print(f"🚀 模型初始化完成，设备：{self.device}")

    def load_weights(self, model_path):
        """加载模型权重"""
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"✅ 加载权重: {model_path}")

    def predict(self, image: Image.Image, resize_to=(640, 640)):
        """
        推理图像，image 为原始 PIL 图片
        会 resize 到 (640, 640) 再输入模型，输出映射回原图尺寸
        """
        orig_w, orig_h = image.size
        resized_img = image.resize(resize_to)
        scale_x = orig_w / resize_to[0]
        scale_y = orig_h / resize_to[1]

        img_tensor = transforms.ToTensor()(resized_img).to(self.device)  # ✅ 正确：[3, H, W]

        with torch.no_grad():
            prediction = self.model([img_tensor])[0]
            boxes = prediction['boxes'].cpu()
            scores = prediction['scores'].cpu()

        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        return boxes, scores
