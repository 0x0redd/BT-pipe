# Brain Tumor Multi-Step Diagnosis Pipeline (BraTS Task01)

**Goal:** end-to-end pipeline that takes a **4D multi-modal MRI** (FLAIR/T1/T1gd/T2) → performs **tumor detection + multi-class 3D segmentation** → generates a **structured Findings JSON** (volume/location/composition) → produces a **natural-language report** using **Ollama (Llama 3.1)**.

> Research/education project only — not for clinical diagnosis.

---

## What you already have (from `my-final-project.ipynb`)

Your notebook is a good starting point for **data access + 3D model experimentation**, but it currently:

* uses **only one modality** (channel 0) during resizing
* performs **binary segmentation** (single sigmoid output)
* resizes labels with **anti_aliasing=True** (this can corrupt discrete classes)
* doesn’t include BraTS metrics (WT/TC/ET) or a reporting step

This README describes how to upgrade it to the full **multi-step diagnosis pipeline**.

---

## Dataset (Medical Segmentation Decathlon / BraTS Task01)

* **Input per case:** one **4D NIfTI** `BRATS_XXX.nii.gz` with shape **(240, 240, 155, 4)**

  * channel 0: FLAIR
  * channel 1: T1w
  * channel 2: T1gd
  * channel 3: T2w
* **Label per case:** one **3D NIfTI** with segmentation classes:

  * 0 background
  * 1 edema
  * 2 non-enhancing tumor
  * 3 enhancing tumor
    *(Some Task01 releases use label 4 for enhancing; if so, remap 4 → 3 in preprocessing.)*

---

## Final Pipeline (high level)

### Phase 1 — Detection + 3D Segmentation

1. **Load & preprocess** 4D MRI:

   * Convert to channel-first: `(4, H, W, D)`
   * Normalize intensities **per modality**
   * Patch-based sampling (GPU friendly)
2. **3D multi-class segmentation**:

   * baseline: 3D U-Net / DynUNet / SegResNet
   * output: voxel mask with classes 0–3
3. **Detection output** (derived from segmentation):

   * tumor present/absent
   * bounding box of tumor region
   * optional: per-region bboxes (edema vs core)

### Phase 2 — Findings JSON + VLM/LLM Reporting

4. **Measurements extraction** (deterministic):

   * volumes per class (cm³) + WT/TC/ET volumes
   * centroid + laterality (left/right by midline heuristic)
   * bounding box & largest diameter (mm) using voxel spacing
   * QC flags (empty mask, tiny tumor, etc.)
5. **Report generation via Ollama (Llama 3.1)**:

   * feed Findings JSON into a strict prompt:

     * “use only provided measurements”
     * “if missing → Not assessed”
   * output: Technique / Findings / Impression

---

## Recommended Repo Structure

```
brain-tumor-pipeline/
├─ notebooks/
│  ├─ my-final-project.ipynb
│  ├─ brats_segmentation_train.ipynb
│  └─ brats_phase2_reporting.ipynb
├─ src/
│  ├─ data/
│  │  ├─ brats_dataset.py
│  │  └─ transforms.py
│  ├─ models/
│  │  └─ seg_unet3d.py
│  ├─ inference/
│  │  ├─ sliding_window.py
│  │  └─ postprocess.py
│  ├─ reporting/
│  │  ├─ findings.py
│  │  ├─ prompt_templates.py
│  │  └─ ollama_client.py
│  └─ metrics/
│     ├─ segmentation_metrics.py
│     └─ brats_regions.py
├─ scripts/
│  ├─ train_seg.py
│  ├─ infer_case.py
│  └─ generate_report.py
├─ runs/
│  ├─ checkpoints/
│  ├─ predictions/
│  └─ reports/
└─ README.md
```

---

## Environment Setup

### 1) Python + CUDA (recommended)

* Python **3.10/3.11**
* PyTorch **CUDA build** (driver can be CUDA 12.x; PyTorch cu121 is fine)

Example:

```bash
python -m venv dl
dl\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install monai nibabel numpy matplotlib requests
```

### 2) Ollama + Llama 3.1

Make sure Ollama is running:

```bash
ollama list
```

---

## How to reach the goal (Implementation Roadmap)

### Milestone A — Correct preprocessing (multi-modal, segmentation-safe)

* ✅ Keep all **4 modalities**
* ✅ Don’t resize labels with anti-aliasing

  * if resizing is necessary: **nearest neighbor** for labels
* ✅ Use orientation normalization + intensity normalization per modality
* ✅ Patch sampling (pos/neg) to fit GPU memory

**Deliverable:** dataloader that returns:

* image tensor: `(B, 4, H, W, D)`
* label tensor: `(B, 1, H, W, D)` integer classes

---

### Milestone B — Baseline 3D multi-class segmentation model

Start with a 3D U-Net baseline:

* loss: **Dice + CrossEntropy** (handles imbalance)
* training: mixed precision + patch sampling
* inference: **sliding window**

**Deliverable:** checkpoint + validation metrics.

---

### Milestone C — BraTS evaluation (the right way)

Compute:

* per-class Dice/IoU (classes 1/2/3)
* BraTS regions:

  * **WT** = (1|2|3)
  * **TC** = (2|3)
  * **ET** = (3)
* optional: sensitivity/specificity per region

**Deliverable:** metrics table + plots per epoch.

---

### Milestone D — Detection output (derived from mask)

From predicted segmentation:

* tumor present/absent
* bounding box `(minx,miny,minz,maxx,maxy,maxz)`
* largest diameter estimate

**Deliverable:** `prediction.json` per case including detection fields.

---

### Milestone E — Findings JSON + Reporting with Ollama (Llama 3.1)

1. Build Findings JSON:

* volumes (cm³) by class + WT/TC/ET
* centroid + laterality
* diameter + bbox
* QC flags

2. Generate report:

* strict prompt that forbids hallucination
* output to `runs/reports/<case_id>.md`

**Deliverable:** `findings.json` + `report.md` per case.

---

## Prompting Strategy (Ollama / Llama 3.1)

Use a **fact-locked** prompt:

* Provide Findings JSON as the single source of truth
* Rules:

  * “Use ONLY these values”
  * “If missing → Not assessed”
  * “No patient demographics or symptoms”

This makes the report **auditable** and reduces hallucination.

---

## Expected Baseline Outcomes (sanity targets)

With a reasonable 3D U-Net baseline (proper preprocessing + loss), you should see:

* Dice for WT/TC/ET improving steadily (ET is usually hardest)
* Meaningful volume estimates and stable laterality/centroid outputs
* Reports that consistently include computed numbers (not invented)

*(Exact numbers depend on split/augmentation and compute limits.)*

---

## Reliability, Interpretability, and Compliance

* **No PHI** should be stored or sent to the LLM
* Store only:

  * segmentation masks
  * derived numeric findings (volumes, bbox, centroid)
  * optional overlay images without identifiers
* Add QC flags to every case:

  * empty prediction
  * tiny tumor
  * suspicious volumes

