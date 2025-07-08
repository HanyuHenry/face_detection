Face Detection - PyTorch Project
=================================

This is a face detection project built with PyTorch. It supports multi-face detection, training, inference, and visualization. It is intended for personal research, learning, or deployment use.

Project Structure
-----------------
face_detection/
│
├── datasets/                # Image data and annotations
│   ├── raw/images/          # Input images (.jpg)
│   └── annotations.json     # Ground-truth annotations
│
├── models/
│   └── face_model.py        # Model architecture definition
│
├── scripts/
│   └── face_dataset.py      # Custom dataset loader
│
├── train.py                 # Main training script
├── predict.py               # Inference and visualization
├── .gitignore               # Git exclusion rules
└── README.txt               # This file

Dependencies
------------
- Python 3.8+
- PyTorch
- torchvision
- tqdm
- matplotlib
- Pillow

To install dependencies:
    pip install torch torchvision matplotlib tqdm pillow

Data Format
-----------
Images go into: datasets/raw/images/

The annotation file should be in JSON format (datasets/annotations.json) like:
[
  {
    "filename": "img1.jpg",
    "boxes": [
      {"x": 30, "y": 40, "w": 60, "h": 80}
    ]
  },
  ...
]

Training
--------
To start training:
    python train.py

Checkpoints:
- checkpoint.pt          : Auto-saved after each epoch
- best_model.pt          : Saved when loss improves
- Checks/backup_epoch_XX.pt : Saved every 5 epochs

Inference / Visualization
-------------------------
To visualize detection results on 3 random images:
    python predict.py

It will display:
- Red boxes  : predicted bounding boxes (after NMS)
- Blue boxes : ground truth boxes (if annotations exist)

Git Notice
----------
Large files like 'best_model.pt' are excluded by .gitignore and should not be pushed to GitHub.
If you accidentally committed a large file, remove it using:
    git rm --cached best_model.pt
    git commit -m "Remove large model file"
    git push origin main --force

Planned Features
----------------
- Annotation UI support
- Real-time video stream detection
- Lightweight models like YOLOv8

Contact
-------
For feedback, bugs, or contributions, please open an issue on GitHub.

