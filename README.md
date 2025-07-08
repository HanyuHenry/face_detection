
Face Detection - PyTorch Project
================================

This is a face detection project built with PyTorch. It supports multi-face detection, training, inference, and visualization. It is intended for personal research, learning, or deployment use.

Project Structure
-----------------
face_detection/
  ├── datasets/
  │     ├── raw/images/         (Input images)
  │     └── annotations.json    (Ground-truth annotations)
  ├── models/
  │     └── face_model.py       (Model architecture)
  ├── scripts/
  │     └── face_dataset.py     (Custom dataset loader)
  ├── train.py                  (Main training script)
  ├── predict.py                (Inference and visualization)
  ├── .gitignore                (Git exclusion rules)
  └── README.txt                (This file)

Dependencies
------------
- Python 3.8+
- PyTorch
- torchvision
- tqdm
- matplotlib
- Pillow

Install:
    pip install torch torchvision matplotlib tqdm pillow

Data Format
-----------
Images: datasets/raw/images/

Annotations: datasets/annotations.json
Format (JSON):
[
  {
    "filename": "img1.jpg",
    "boxes": [
      {"x": 30, "y": 40, "w": 60, "h": 80}
    ]
  }
]

Training
--------
To start training:
    python train.py

Checkpoints:
- checkpoint.pt          : Saved after each epoch
- best_model.pt          : Saved when loss improves
- Checks/backup_epoch_XX.pt : Saved every 5 epochs

Inference / Visualization
--------------------------
To visualize predictions:
    python predict.py

Shows:
- Red boxes  : predicted faces (after NMS)
- Blue boxes : ground truth boxes (if available)

GitHub Push Notes
-----------------
Large files like best_model.pt are excluded via .gitignore.
If you accidentally added them:

    git rm --cached best_model.pt
    git commit -m "Remove large model file"
    git push origin main --force

Planned Features
----------------
- Annotation UI tool
- Video stream detection
- Lightweight model support (e.g., YOLOv8)

Contact
-------
For feedback or contributions, open an issue at:
https://github.com/HanyuHenry/face_detection/issues
