# 🔍 AI-Based DeepFake Detection for Digital Media Authentication

> Final Year Project — Deep Learning | Computer Vision | Explainable AI

A complete web application that detects AI-generated deepfake images and videos using a fine-tuned **EfficientNet-B4** model, with **Grad-CAM** heatmap visualizations showing *exactly which facial regions* the model used to make its decision.

---

## 📸 Features

- **Image Detection** — Upload any JPG/PNG/WEBP image; get a Real/Fake verdict with confidence score
- **Video Detection** — Analyzes up to 20 evenly-spaced frames; shows per-frame fake probability chart
- **Grad-CAM Heatmaps** — Visual explanation of model decisions (red = suspicious regions)
- **Face Detection** — MTCNN automatically locates and crops the face before classification
- **Clean Web UI** — Dark-mode Flask frontend, drag-and-drop upload, responsive design

---

## 🏗 Project Structure

```
deepfake_detector/
├── app.py                          # Flask web server & API routes
├── train.py                        # Model training script
├── evaluate.py                     # Metrics: accuracy, AUC, confusion matrix, ROC
├── requirements.txt
│
├── models/
│   ├── detector.py                 # DeepFakeDetector class (EfficientNet-B4)
│   └── efficientnet_deepfake.pth   # ← saved here after training
│
├── utils/
│   ├── gradcam.py                  # Grad-CAM heatmap generation
│   └── video_utils.py              # Frame extraction from videos
│
├── dataset_prep/
│   └── prepare_dataset.py          # Face crop pipeline for FF++ / DFDC
│
├── static/
│   ├── css/style.css
│   └── js/app.js
│
└── templates/
    ├── index.html
    └── about.html
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/deepfake-detector.git
cd deepfake-detector

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the app (demo mode)

```bash
python app.py
```

Open **http://localhost:5000** — the app works immediately in demo mode using ImageNet pretrained weights. Results will not be accurate until you train on a deepfake dataset (Step 4).

---

## 🧠 Model Training

### Step 3: Get the dataset

Download **FaceForensics++** (most popular academic deepfake dataset):
1. Request access at: https://github.com/ondyari/FaceForensics
2. Download `original_sequences/` (real) and `manipulated_sequences/Deepfakes/` (fake)
3. Use c23 (light compression) for best results

Or use **Celeb-DF** (easier to download):
```
https://github.com/yuezunli/celeb-deepfakeforensics
```

### Step 4: Prepare dataset (face cropping)

```bash
python dataset_prep/prepare_dataset.py \
    --real_dir /path/to/original_videos \
    --fake_dir /path/to/deepfake_videos \
    --out_dir  data/ \
    --val_split 0.2 \
    --max_per_video 30
```

This creates:
```
data/
  train/real/   train/fake/
  val/real/     val/fake/
```

### Step 5: Train

```bash
python train.py --data_dir data/ --epochs 10 --batch_size 32
```

Training details:
- **Epochs 1–3**: Only classifier head trained (frozen backbone)
- **Epoch 4+**: Full fine-tuning at 10× lower learning rate
- Best model (highest Val AUC) saved to `models/efficientnet_deepfake.pth`

Expected results on FaceForensics++:
| Metric | Value |
|--------|-------|
| Accuracy | ~93–96% |
| ROC-AUC | ~0.97–0.99 |

---

## 📊 Evaluation

```bash
python evaluate.py --data_dir data/val --output_dir results/
```

Generates in `results/`:
- `metrics.json` — accuracy, AUC, confusion matrix
- `roc_curve.png` — ROC curve plot
- `confusion_matrix.png` — confusion matrix heatmap

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch 2.x |
| Model backbone | EfficientNet-B4 (ImageNet pretrained) |
| Face detection | MTCNN (facenet-pytorch) |
| Explainability | Grad-CAM (custom implementation) |
| Image processing | OpenCV, Pillow |
| Web framework | Flask 3.x |
| Frontend | Vanilla JS, CSS3 |

---

## 📚 References

1. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML 2019*
2. Rössler et al. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. *ICCV 2019*
3. Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV 2017*
4. Li et al. (2020). Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics. *CVPR 2020*

---

## ⚠️ Disclaimer

This project is developed for **academic and research purposes only**. Uploaded media is not permanently stored. Detection results should not be used as sole evidence in legal or forensic proceedings.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
