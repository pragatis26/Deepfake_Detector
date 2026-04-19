"""
dataset_prep/prepare_dataset.py
================================
Prepares FaceForensics++ (or any deepfake dataset) for training:
  1. Detects and crops faces from each image/frame
  2. Saves cropped faces into  data/train/real, data/train/fake,
                               data/val/real,   data/val/fake
  3. Prints a summary

Usage:
    python dataset_prep/prepare_dataset.py \
        --real_dir  /path/to/original_videos_or_frames \
        --fake_dir  /path/to/manipulated_videos_or_frames \
        --out_dir   data/ \
        --val_split 0.2 \
        --max_per_video 30

FaceForensics++ download instructions:
    https://github.com/ondyari/FaceForensics
    Request access → download original + Deepfakes (c23 compression)
"""

import os
import cv2
import random
import argparse
from pathlib import Path

try:
    from facenet_pytorch import MTCNN
    import torch
    mtcnn = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')
    USE_MTCNN = True
except ImportError:
    USE_MTCNN = False
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_and_crop_face(img_bgr, margin=20):
    """Detect largest face and return cropped PIL image (BGR numpy)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    if USE_MTCNN:
        from PIL import Image
        pil = Image.fromarray(img_rgb)
        boxes, _ = mtcnn.detect(pil)
        if boxes is None:
            return None
        x1, y1, x2, y2 = [int(v) for v in boxes[0]]
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        x1, y1, x2, y2 = x, y, x + fw, y + fh

    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
    x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
    return img_bgr[y1:y2, x1:x2]


def process_source(src_dir, out_real_train, out_real_val, out_fake_train, out_fake_val,
                   label, val_split, max_per_video):
    """Process all images/videos in src_dir and save face crops."""
    src_path = Path(src_dir)
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}

    all_files = [f for f in src_path.rglob('*') if f.suffix.lower() in image_exts | video_exts]
    random.shuffle(all_files)

    saved_train, saved_val = 0, 0
    train_dir = out_real_train if label == 'real' else out_fake_train
    val_dir   = out_real_val   if label == 'real' else out_fake_val

    for file_path in all_files:
        frames = []

        if file_path.suffix.lower() in image_exts:
            img = cv2.imread(str(file_path))
            if img is not None:
                frames = [img]
        else:
            cap = cv2.VideoCapture(str(file_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step  = max(1, total // max_per_video)
            for i in range(0, total, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                if len(frames) >= max_per_video:
                    break
            cap.release()

        for idx, frame in enumerate(frames):
            face = detect_and_crop_face(frame)
            if face is None:
                continue

            is_val = random.random() < val_split
            dest   = val_dir if is_val else train_dir
            fname  = f"{file_path.stem}_{idx:04d}.jpg"
            out_p  = os.path.join(dest, fname)
            cv2.imwrite(out_p, cv2.resize(face, (224, 224)))

            if is_val:
                saved_val += 1
            else:
                saved_train += 1

    return saved_train, saved_val


def main(args):
    for split in ['train', 'val']:
        for cls in ['real', 'fake']:
            os.makedirs(os.path.join(args.out_dir, split, cls), exist_ok=True)

    print(f"Processing REAL images from: {args.real_dir}")
    rt, rv = process_source(
        args.real_dir,
        os.path.join(args.out_dir, 'train', 'real'),
        os.path.join(args.out_dir, 'val',   'real'),
        os.path.join(args.out_dir, 'train', 'fake'),  # unused for real
        os.path.join(args.out_dir, 'val',   'fake'),  # unused for real
        label='real', val_split=args.val_split, max_per_video=args.max_per_video
    )
    print(f"  → Train: {rt} | Val: {rv}")

    print(f"\nProcessing FAKE images from: {args.fake_dir}")
    ft, fv = process_source(
        args.fake_dir,
        os.path.join(args.out_dir, 'train', 'real'),  # unused for fake
        os.path.join(args.out_dir, 'val',   'real'),  # unused for fake
        os.path.join(args.out_dir, 'train', 'fake'),
        os.path.join(args.out_dir, 'val',   'fake'),
        label='fake', val_split=args.val_split, max_per_video=args.max_per_video
    )
    print(f"  → Train: {ft} | Val: {fv}")

    print(f"\n{'='*50}")
    print(f"Dataset ready in: {args.out_dir}")
    print(f"  Train  Real: {rt}  Fake: {ft}")
    print(f"  Val    Real: {rv}  Fake: {fv}")
    print(f"  Total: {rt+rv+ft+fv} face crops")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dir',      required=True)
    parser.add_argument('--fake_dir',      required=True)
    parser.add_argument('--out_dir',       default='data/')
    parser.add_argument('--val_split',     type=float, default=0.2)
    parser.add_argument('--max_per_video', type=int,   default=30)
    main(parser.parse_args())
