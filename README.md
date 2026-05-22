# Calibration-Free Vehicle Speed Estimation Using Computer Vision

A vision-based vehicle speed estimation pipeline that operates on monocular video footage from a fixed roadside camera without camera calibration.

Built as a final year project at Staffordshire University (BSc Artificial Intelligence & Robotics, 2026).

---

## Overview

Traditional vision-based speed estimation relies on explicit camera calibration (known camera height, angle, and focal length) to convert pixel displacement into real-world speed. This system eliminates that requirement entirely.

Instead, bounding box and trajectory features extracted from tracked vehicles are fed into a trained XGBoost regressor, which learns the implicit relationship between visual tracking behaviour and real-world speed directly from labelled data.

**Pipeline:**

```
Monocular Video → YOLOv8n Detection → DeepSORT Tracking → Feature Extraction → XGBoost Regression → Speed (km/h)
```

---

## Results

Evaluated on a held-out set of 3 unseen vehicle models (91 clips) from the VS13 dataset:

| Metric | Value |
|--------|-------|
| MAE | 3.66 km/h |
| RMSE | 4.88 km/h |
| R² | 0.949 |
| Within 10 km/h | 93.4% |
| Within 5 km/h | 75.8% |

**Per-vehicle breakdown:**

| Vehicle | MAE | RMSE | R² |
|---------|-----|------|----|
| Mazda3 (small) | 3.14 km/h | 5.18 km/h | 0.948 |
| NissanQashqai (mid) | 3.25 km/h | 3.94 km/h | 0.960 |
| MercedesAMG550 (large) | 4.62 km/h | 5.33 km/h | 0.940 |

Results are broadly competitive with calibration-free published work on the same dataset. Mareddy et al. (2025) achieve RMSE 3.96 km/h on VS13 using an LSTM with attention mechanism — a substantially more complex deep learning architecture. This system achieves 4.88 km/h RMSE using a compact, interpretable feature set on standard CPU hardware.

---

## Requirements

- Python 3.9+
- See `requirements.txt` for full dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

You will also need the YOLOv8n pretrained weights. Place `yolov8n.pt` in the same directory as `demo.py`. Weights can be downloaded from [Ultralytics](https://github.com/ultralytics/assets/releases).

---

## Usage

```bash
python demo.py <video_path> [annotation_path]
```

`annotation_path` is optional. If provided, ground truth speed is read and absolute error is reported.

**With ground truth:**
```bash
python demo.py "clip_001.mp4" "clip_001.txt"
```

**Without ground truth:**
```bash
python demo.py "myvideo.mp4"
```

**Example output:**
```
Loading models...
  YOLOv8n      : loaded
  XGBoost v3   : loaded

Ground truth speed : 68.0 km/h

Processing video   : clip_001.mp4
Running detection and tracking — single pass, exit anchored...

Track length       : 87 frames  (2.90s)
Exit frame         : 87
Extraction window  : frames 57–87  (30 points)

========================================
  PREDICTED SPEED  :  64.53 km/h
  GROUND TRUTH     :  68.0 km/h
  ABSOLUTE ERROR   :  3.47 km/h
========================================
```

---

## Dataset

The system was trained and evaluated on the [VS13 dataset](https://doi.org/10.1109/TELFOR56187.2022.9983773) (Djukanovic et al., 2022), comprising 400 video clips across 13 vehicle models with ground-truth speed annotations.

The dataset is not included in this repository. See the citation above for access details.

---

## How It Works

### Observation Window
Rather than relying on annotated pass-by timestamps (which would not be available at deployment), the system anchors the observation window to the **last confirmed frame of the vehicle's track** — the exit frame. This makes the pipeline genuinely deployable without any ground-truth timing information.

A 1.0-second window ending at the exit frame is used. Cross-validation across four window durations (1.0s–2.5s) confirmed this as the most stable configuration.

### Feature Set
23 features are extracted from the bounding box trajectory within the window, organised into three categories:

- **Width dynamics** — apparent width growth rate, expansion rate, normalised growth
- **Height dynamics** — height change rate, aspect ratio evolution
- **Centroid motion** — pixel velocity, displacement statistics

The most important feature is `width_growth_norm` — bounding box width growth rate normalised by mean vehicle width. This resolves a size-confounding issue in the raw growth ratio and accounts for approximately 70% of model decisions. Its dominance is physically defensible: for a vehicle approaching a near-POV fixed camera straight-on, apparent width growth rate is the strongest available visual proxy for approach speed.

### Model
XGBoost was selected over Random Forest (baseline) and neural alternatives. With 309 training samples, tree-based ensemble methods offer comparable performance to deep learning on structured tabular data at this scale, with the added benefit of interpretable feature importance.

Hyperparameters were tuned using RandomizedSearchCV (60 iterations, 5-fold CV), producing a notably regularised configuration (shallower trees, high minimum child weight) appropriate for the dataset scale.

---

## Limitations

- **Single camera geometry** — the model implicitly encodes the geometric properties of the VS13 camera. No claim of generalisation to other camera setups is made. Retraining on labelled data from a new camera would be required.
- **Systematic underestimation bias** — scales with vehicle size (−0.69 km/h for Mazda3, −3.47 km/h for MercedesAMG550), a known trade-off of the width_growth_norm normalisation.
- **High-speed degradation** — MAE at 90–110 km/h (4.91 km/h) is higher than at 30–70 km/h (2.99–3.47 km/h), due to signal compression in bounding box expansion at high approach speeds.

---

## Project Structure

```
├── demo.py                          # Single-video inference script
├── xgb_1s_speed_model_final_v3.pkl  # Trained XGBoost model
├── requirements.txt
└── README.md
```

Feature extraction and training scripts are available on request.

---

## References

- Djukanović et al. (2022) — VS13 dataset
- Mareddy et al. (2025) — LSTM baseline on VS13
- Wojke et al. (2017) — DeepSORT
- Chen & Guestrin (2016) — XGBoost
- Ali & Zhang (2024) — YOLOv8
