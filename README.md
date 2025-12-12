# 📘 **README.md – Brain Tumor Multi-Step Diagnosis Pipeline (Detection + VLM Reporting)**


## 🧠 Project Overview

This repository implements a **complete multi-step AI pipeline** for automated brain tumor analysis from MRI scans.
The system combines:

1. **A custom-trained tumor detection & classification model** (YOLO / Faster-RCNN / CNN)
2. **A feature extraction module** (tumor size, location, cropped ROI, etc.)
3. **A Vision-Language Model (VLM)** fine-tuned to generate **radiology-style reports**

All models are trained **from scratch**, including data preprocessing, augmentation, training loops, evaluation, and fine-tuning.

---

## 🔥 Features

* **Tumor classification** (Glioma, Meningioma, Pituitary, No Tumor)
* **Bounding box detection** (if YOLO / Faster-RCNN option is used)
* **Tumor region cropping & medically relevant feature extraction**
* **VLM-based radiology report generation** using structured inputs + image encodings
* **End-to-end training scripts** for all modules
* **Evaluation pipeline** with metrics (mAP, F1, BLEU/ROUGE for text quality)
* **Modular architecture** → easy to swap models
* **FastAPI inference server** (optional)
* **Clean dataset schema and JSON annotation format**

---

## 🏗 Project Architecture

```
brain-tumor-ai/
│
├── data/
│   ├── raw/                # Original datasets (BRATS, Kaggle, Figshare, etc.)
│   ├── processed/          # Preprocessed MRI images
│   ├── annotations/        # JSON annotations for detection + VLM
│
├── models/
│   ├── detector/           # YOLO/FasterRCNN implementation & training
│   ├── vlm/                # VLM fine-tuning scripts (Unsloth / vLLM)
│   ├── feature_extractor/  # Tumor location, size, ROI crop
│
├── scripts/
│   ├── preprocess.py
│   ├── train_detector.py
│   ├── train_vlm.py
│   ├── evaluate.py
│   ├── inference_pipeline.py
│
├── configs/
│   ├── detector.yaml
│   ├── vlm_config.json
│   ├── dataset_schema.json
│
├── notebooks/
│   ├── EDA.ipynb           # Dataset exploration
│   ├── Detector_Training.ipynb
│   ├── VLM_FineTune.ipynb
│
├── docs/
│   ├── architecture_diagram.png
│   ├── dataset_guidelines.md
│   ├── vlm_prompting.md
│
├── app/
│   ├── api.py              # FastAPI server
│   ├── ui/                 # Optional frontend
│
└── README.md
```

---

## 🧩 Pipeline Description

### **Step 1 — Tumor Detection & Classification**

You will train a model from scratch using PyTorch.
You can choose between:

* CNN classifier
* Faster-RCNN
* YOLOv8/Yolov12

**Output Example:**

```json
{
  "tumor_type": "Glioma",
  "confidence": 0.93,
  "bbox": [x1, y1, x2, y2]
}
```

---

### **Step 2 — Medical Feature Extraction**

Using the bounding box you compute:

* Tumor location (left/right hemisphere)
* Estimated size (mm² or cm²)
* Crop of tumor region
* Shape + intensity stats

**Output Example:**

```json
{
  "location": "Left frontal lobe",
  "size_mm": 24.7,
  "crop_path": "data/crops/image123.png"
}
```

---

### **Step 3 — VLM Radiology Reporting**

You fine-tune a Vision-Language Model using Unsloth or vLLM.

**Inputs to the VLM:**

* Original MRI
* Tumor crop
* Detected tumor class
* Extracted features

**Output:**
A radiology-style, structured report.

---

## 📦 Dataset Requirements

This project supports multiple sources:

* **BRATS 2020/2021**
* **Kaggle Brain Tumor Dataset**
* **Figshare MRI datasets**

You must unify all datasets into the following JSON format:

### **dataset_schema.json**

```json
{
  "image": "path/to/mri.png",
  "label": "Glioma",
  "bbox": [100, 40, 350, 300],
  "extra_features": {
    "location": "Left temporal lobe",
    "size_mm": 27.1
  },
  "report": "Ground truth radiology report here."
}
```

---

## 🏋️ Training From Scratch

### **1️⃣ Train the Tumor Detector**

```bash
python scripts/train_detector.py \
    --config configs/detector.yaml \
    --epochs 100 \
    --batch-size 16
```

**Quickstart for the new training script**

1. Install deps: `pip install torch torchvision pyyaml pillow`
2. Prepare `data/annotations/train.json` and `data/annotations/val.json` (list of records with `image`, `label`, and `bbox`).
3. Point `configs/detector.yaml` to your image root and label set. Set `task` to `classification` (ResNet18 head) or `detection` (FasterRCNN).
4. Run: `python scripts/train_detector.py --config configs/detector.yaml`

---

### **2️⃣ Generate Features & Cropped Tumor Regions**

```bash
python scripts/preprocess.py
```

---

### **3️⃣ Fine-Tune the VLM**

```bash
python scripts/train_vlm.py \
    --config configs/vlm_config.json \
    --epochs 5
```

---

## 🧪 Evaluation

### **Detection Metrics**

* mAP (0.5 / 0.5:0.95)
* Precision / Recall
* Confusion Matrix

### **Text Report Metrics**

* BLEU
* ROUGE-L
* Medical factuality score (custom)

Run evaluation:

```bash
python scripts/evaluate.py
```

---

## 🚀 Inference Pipeline

For deployment, combine all steps:

```bash
python scripts/inference_pipeline.py \
    --image test/sample.png \
    --output report.json
```

Output example:

```json
{
  "tumor_type": "Pituitary",
  "features": { "size_mm": 18.4, "location": "Right side" },
  "report": "The scan demonstrates a pituitary macroadenoma..."
}
```

---

## 🌐 Optional: FastAPI Server

Start the API:

```bash
uvicorn app.api:app --reload
```

Send a request:

```bash
POST /analyze
```

---

## 📚 Documentation

| Topic         | File                            |
| ------------- | ------------------------------- |
| Dataset rules | `docs/dataset_guidelines.md`    |
| VLM prompting | `docs/vlm_prompting.md`         |
| Architecture  | `docs/architecture_diagram.png` |

---

## 🤝 Contributing

Pull requests are welcome!
Please open an issue to discuss improvements or bugs.

---

## 📄 License

MIT License — This project is fully open for research and educational use.

---

## ⭐ Acknowledgements

This project integrates:

* PyTorch
* Unsloth / vLLM
* YOLO / Faster-RCNN
* Medical imaging datasets (BRATS, Figshare)

