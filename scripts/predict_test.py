from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw
import io
import torch

from scripts.inference import FaceDetector

app = FastAPI()
detector = FaceDetector()
detector.load_weights("best_model.pt")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    boxes, scores = detector.predict(image)

    # ✅ 画框
    draw = ImageDraw.Draw(image)
    for box, score in zip(boxes, scores):
        if score.item() >= 0.5:
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # ✅ 转换为字节流返回
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")
