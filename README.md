# Evaluating the challenges in object segmentation in the video: CUTIE vs. XMem

<p align="center">
  Comparative evaluation framework for Video Object Segmentation using <b>CUTIE</b> and <b>XMem</b>
</p>

---

## Overview

This project supports a research study on:

**Applying Deep Learning to the Video Object Segmentation Task**

The main objective is to evaluate and compare two state-of-the-art video object segmentation models:

- **CUTIE**
- **XMem**

The project includes:
- inference result management
- multiple evaluation metrics
- qualitative visualization with overlay videos

---

## Project Structure

```bash
.
├── assets/
│   ├── Efficiency/
│   ├── Object Persistence/
│   ├── Results/
│   ├── Shape Quality/
│   └── Temporal Stability/
│
├── data/
│   ├── Annotations/
│   └── JPEGImages/
│
├── evaluation/
│   ├── efficiency.py
│   ├── object_persistence.py
│   ├── shape_quality.py
│   ├── temporal_stability.py
│   └── video_demo.py
│
├── experiments/
│   ├── results_cutie/
│   └── results_xmem/
│
├── model/
│   ├── Cutie/
│   └── XMem/
│
└── scripts/
    ├── prepare_dataset.sh
    ├── run_cutie.sh
    └── run_xmem.sh
```

---

## Directory Description

### `assets/`
Stores output resources used for visualization, reporting, and presentation.

Subfolders:
- `Efficiency/` — efficiency evaluation figures and results
- `Object Persistence/` — object persistence evaluation results
- `Results/` — general outputs and final visualizations
- `Shape Quality/` — mask shape quality evaluation results
- `Temporal Stability/` — temporal stability evaluation results

---

### `data/`
Contains the dataset used for evaluation.

- `JPEGImages/` — RGB frames for each sequence
- `Annotations/` — ground-truth segmentation masks

Example:
```bash
data/
├── Annotations/
│   ├── 0ql59q5s/
│   ├── 1wjebgyd/
│   └── 3v2kgn6k/
└── JPEGImages/
    ├── 0ql59q5s/
    ├── 1wjebgyd/
    └── 3v2kgn6k/
```

---

### `evaluation/`
Contains scripts for metric computation and qualitative comparison.

- `efficiency.py` — evaluates runtime performance of CUTIE and XMem
- `object_persistence.py` — evaluates how well the object is preserved across frames
- `shape_quality.py` — evaluates the predicted mask quality
- `temporal_stability.py` — evaluates segmentation consistency over time
- `video_demo.py` — generates overlay comparison videos between CUTIE and XMem

---

### `experiments/`
Contains inference outputs from each model.

- `results_cutie/` — predicted masks from CUTIE
- `results_xmem/` — predicted masks from XMem

Example:
```bash
experiments/
├── results_cutie/
│   ├── 0ql59q5s/
│   ├── 1wjebgyd/
│   └── 3v2kgn6k/
└── results_xmem/
    ├── 0ql59q5s/
    ├── 1wjebgyd/
    └── 3v2kgn6k/
```

---

### `model/`
Contains the source code or local model folders for the two VOS methods.

- `Cutie/`
- `XMem/`

This directory is used for:
- inference execution
- checkpoint organization
- reference to the original model implementations

---

### `scripts/`
Contains shell scripts for running the full workflow.

- `prepare_dataset.sh` — dataset preparation
- `run_cutie.sh` — runs inference using CUTIE
- `run_xmem.sh` — runs inference using XMem

---

## Workflow

### 1. Prepare the dataset
Place the dataset into the following structure:

```bash
data/JPEGImages/<sequence_name>/
data/Annotations/<sequence_name>/
```

Then run:

```bash
bash scripts/prepare_dataset.sh
```

---

### 2. Run CUTIE inference

```bash
bash scripts/run_cutie.sh
```

Outputs will be saved to:

```bash
experiments/results_cutie/
```

---

### 3. Run XMem inference

```bash
bash scripts/run_xmem.sh
```

Outputs will be saved to:

```bash
experiments/results_xmem/
```

---

### 4. Run evaluations

```bash
python evaluation/efficiency.py
python evaluation/object_persistence.py
python evaluation/shape_quality.py
python evaluation/temporal_stability.py
```

---

### 5. Generate qualitative comparison video

```bash
python evaluation/video_demo.py
```

Generated videos and visual outputs are typically saved in:

```bash
assets/Results/
```

---

## Evaluation Categories

### 1. Efficiency
Measures runtime-related performance:
- per-frame processing time
- average latency
- average FPS
- latency stability comparison between CUTIE and XMem

### 2. Object Persistence
Measures how well the segmented object is maintained throughout the video:
- object continuity
- object preservation over time
- robustness against losing the target

### 3. Shape Quality
Measures the quality of predicted segmentation masks:
- completeness of the object region
- boundary quality
- similarity to the ground truth mask

### 4. Temporal Stability
Measures consistency between consecutive frames:
- reduced flickering
- reduced mask jitter
- stable segmentation over time

---

## Example Sequences

Several representative sequences used in evaluation:

- `0ql59q5s`
- `1wjebgyd`
- `3v2kgn6k`

---

## Future Work

Possible extensions for this project:
- Investigating the replacement of the Transformer architecture in CUTIE with Mamba (State Space Model - SSM) to evaluate its potential for improving performance and reducing computational cost
- Optimizing the model and processing pipeline to achieve real-time inference, thereby enhancing its practicality for real-world applications




