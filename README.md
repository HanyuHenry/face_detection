
# Face Detection with Single-Face Regression

This project implements a simple face detection system using bounding box regression.
It is designed to work with images containing **only one face**, using the WIDER FACE dataset.

## Project Structure

```
face_detection/
│
├── datasets/
│   ├── raw/                         # Raw WIDER FACE data
│   │   ├── images/                 # Original images
│   │   └── wider_face_train_bbx_gt.txt  # Original annotations
│   ├── images_singleface/          # ✅ Filtered images with only one face
│   └── annotations_singleface.json # ✅ Corresponding annotations (generated automatically)
│
├── models/
│   └── face_model.py               # Model definition
│
├── scripts/
│   ├── prepare_singleface.py       # Script to extract single-face data
│   └── predict_and_draw.py         # Script to visualize predictions
│
├── train.py                        # Training script
├── face_box_model.pt               # Trained model weights
└── last_epoch.txt                  # Tracks last training epoch
```

## Installation

```bash
pip install torch torchvision pillow matplotlib
```

## Data Preparation

1. Download and unzip the **WIDER FACE Training Images** and the `wider_face_train_bbx_gt.txt` file to `datasets/raw/`
2. Run the following script to filter and prepare single-face images:

```bash
python scripts/prepare_singleface.py
```

## Training

```bash
python train.py
```

- Trains on the `datasets/images_singleface` folder.
- Saves model weights to `face_box_model.pt`.

## Visualization

```bash
python scripts/predict_and_draw.py
```

- Randomly selects 5 images.
- Shows red predicted box and green ground truth boxes.
- Coordinates are printed in the terminal.

## Data Augmentation

The following augmentations are applied during training:

- `transforms.Resize((640, 640))`
- `transforms.RandomHorizontalFlip()`
- `transforms.ColorJitter()`
- `transforms.RandomRotation(10)`

## Notes

- The model assumes only **one face per image**.
- Training on multi-face images will reduce performance.
- For multi-face detection, consider switching to detection models like YOLO or Faster R-CNN.
