# Comic Panel Extraction

Official implementation of the paper:  
**"Comic Panel Extraction using U-Net++ Deep Learning Model and Line Segment Analysis"**

This is an upgraded version of the original repository, [huyproman/comic-panel-segmentation](https://github.com/huyproman/comic-panel-segmentation).

---

## Overview

Automatic comic panel extraction adapts page-based comics to digital displays. Comics feature diverse artistic styles, non-rectangular panels, black backgrounds, and speech balloons or characters that overlap panel boundaries.

This repository implements the proposed hybrid pipeline:

1. **Neural segmentation.** A **U-Net++** trained from scratch predicts a per-pixel panel probability map at 448×448, with no rigid geometric assumptions about panel shape.
2. **Line Segment Analysis (LSA).** Recovers missing panel borders. It detects straight boundary segments with the Probabilistic Hough Transform, merges near-duplicate segments, clusters their intersections into corner points with Mean-Shift, and carves a separating line between two corners when the path between them is supported by detected lines (checked with Bresenham sampling).
3. **Geometric Filtering (GF).** Removes small residual components (≤ 1000 px).
4. **Instance evaluation.** Splits predicted and ground-truth masks into panels (connected components), matches them one-to-one with the Hungarian algorithm, and reports segmentation and panel-detection metrics.

```
image ──► U-Net++ ──► probability map ──(≥ 0.5)──► raw mask ──► LSA ──► GF ──► final mask
                                                     448×448                  resized back to the image size
```

---

## Installation

### Requirements
- Python 3.10 or higher
- TensorFlow ≥ 2.15 and Keras 3 (installed automatically)
- OpenCV 4.8 or later (OpenCV 4 and 5 both work)
- A CUDA-compatible GPU is recommended for training.

### Install from source
```bash
git clone https://github.com/longle0702/comic-panel-segmentation.git
cd comic-panel-segmentation
pip install -e .                          # CPU
pip install -e ".[gpu]"                   # with CUDA-enabled TensorFlow
pip install -e ".[dev]"                   # adds pytest and ruff
```

The package installs a `comics-panel-extraction` command. `python -m comics_panel_extraction` runs the same CLI.

---

## Quickstart

### 1. Inference
Run the full pipeline (U-Net++ → LSA → GF) on one page:
```bash
comics-panel-extraction infer page.jpg --checkpoint path/to/model.keras --output final_mask.png
```

| Option | Meaning |
|---|---|
| `-c, --checkpoint` | Keras model file (required). Must map `(448, 448, 3)` to `(448, 448, 1)`. |
| `-o, --output` | Output mask. Written as `.npy` if the name ends in `.npy`, otherwise as an image. |
| `--raw-output PATH` | Also save the raw U-Net++ mask, before LSA and GF. |
| `--raw-only` | Skip LSA and GF; `--output` is then the raw mask. |
| `--threshold T` | Probability threshold for the raw mask (default `0.5`). |

The page is read as RGB, matching training, and resized to 448×448 for the model. A `.npy` input must already be RGB. LSA and GF run on the 448×448 mask, and the result is resized back to the original page size (nearest neighbour). Output masks are 0/255.

No trained checkpoint is included; see [Reproducibility](#reproducibility).

### 2. Post-processing only
Apply LSA + GF to an existing binary mask:
```bash
comics-panel-extraction postprocess raw_mask.png --output refined_mask.png
```

### 3. Evaluation
Evaluate a folder of predicted masks against a folder of ground-truth masks:
```bash
comics-panel-extraction evaluate --predictions preds/ --ground-truth gt/ --output-json metrics.json
```

- Predictions (`*.png`, `*.jpg`) are paired with the ground-truth file of **the same filename**; unpaired files are skipped.
- Masks are read as greyscale; any non-zero pixel is foreground.
- The result is written to `metrics.json` (see [Evaluation protocol](#evaluation-protocol) for the keys).

### 4. Training
```bash
comics-panel-extraction train --dataset data/train --val-dataset data/val \
    --config configs/default.yaml --epochs 150 --batch-size 8 --output-dir runs/train
```

Expected dataset layout (image and mask share the same filename):
```
data/train/
├── images/   page_001.jpg, page_002.png, ...
└── masks/    page_001.jpg, page_002.png, ...   (panel = white, background = black)
```

- Images are converted to RGB, resized to 448×448 (bilinear) and scaled to [0, 1]. Masks are resized with nearest neighbour and binarised at > 127.
- `--dataset`, `--val-dataset`, `--epochs`, `--batch-size` and `--output-dir` override the `training:` section of the config. The default config points to `data/train` and `data/val`; validation is skipped if the folder does not exist.
- After every epoch, the model with the lowest validation loss (training loss if there is no validation set) is saved as `best_model.keras`. The model after the last epoch is saved as `latest_model.keras`.

---

## Python API

```python
import cv2
from comics_panel_extraction.inference import load_inference_model, predict_mask
from comics_panel_extraction.postprocessing import run_line_segment_postprocessing
from comics_panel_extraction.evaluation import PanelInstanceEvaluator, evaluate_instance_dataset

model = load_inference_model("model.keras")
image = cv2.cvtColor(cv2.imread("page.jpg"), cv2.COLOR_BGR2RGB)  # the model was trained on RGB
final_mask, raw_mask, diagnostics = predict_mask(image, model)   # both 0/255, at the page size

# Ablation: run LSA and GF separately on a 0/255 (or 0/1) mask.
lsa_only, _ = run_line_segment_postprocessing(raw_mask, apply_lsa=True,  apply_gf=False)
gf_only,  _ = run_line_segment_postprocessing(raw_mask, apply_lsa=False, apply_gf=True)
lsa_gf,   _ = run_line_segment_postprocessing(raw_mask)               # both (default)

evaluator = PanelInstanceEvaluator(dice_threshold=0.90, connectivity=2)
gt_mask = cv2.imread("gt.png", cv2.IMREAD_GRAYSCALE)
page = evaluator.evaluate_page(final_mask, gt_mask)    # per-page metrics
dataset = evaluate_instance_dataset([page])            # mean over pages
```

`diagnostics` reports the number of detected Hough lines, merged horizontal and vertical lines, corner centroids, and carved connections.

---

## Method Specifications

### Model architecture (`model/unetpp.py`)
- **Architecture**: U-Net++ with dense nested skip pathways, without deep supervision (single output).
- **Encoder channels**: 32, 64, 128, 256, 512.
- **Blocks**: two × (3×3 Conv2D, He-normal initialisation → BatchNorm → ReLU); 2×2 max pooling; 2×2 transposed convolutions for upsampling.
- **Output**: 1×1 Conv2D with sigmoid.
- **Input / output**: 448×448×3 → 448×448×1 probability map.
- **Parameters**: 9,056,769.

### Training (`training/`)
- **Loss**: binary cross-entropy + soft Dice loss (Dice smoothing 1), summed with equal weight.
- **Optimizer**: AdamW, learning rate 1e-4, weight decay 1e-4.
- **Schedule**: linear warm-up over 5 epochs, then cosine decay to 0 by the last epoch.
- **Defaults**: 150 epochs, batch size 8, seed 42 (`keras.utils.set_random_seed`).

### Line Segment Analysis (`postprocessing/line_analysis.py`)
LSA ports the reference notebook pipeline step by step. Values that the notebook hard-coded for 384×384 masks are expressed relative to the actual mask size, so a 384×384 mask gives the notebook's output exactly.

1. **Smoothing**: morphological opening with a 20×20 rectangle.
2. **Edges**: Canny (50, 100), then dilate 5×5 (×2), erode 5×5, erode 3×3 to bridge small gaps.
3. **Lines**: Probabilistic Hough Transform (ρ = 2, θ = π/180, threshold = 25, minLineLength = 10, maxLineGap = 30). Each segment is classed as vertical if its angle to the x-axis is ≥ 45°, otherwise horizontal.
4. **Grouping**: segments of the same orientation are grouped when their angle difference is < 3° and their distances to three fixed reference points (placed relative to the image size) all differ by < 8.5 px. Groups sharing a segment are merged. From each group, the segments at the most common angle are kept, or the segment at the median angle if no angle repeats.
5. **Corners**: intersections of vertical × horizontal lines are clustered with Mean-Shift (bandwidth 13). Intersections with the image border are clustered separately (bandwidth 10). Border intersections are kept if 0 ≤ x ≤ W·400/384 and 0 ≤ y ≤ H·400/384.
6. **Carving**: the merged lines are drawn (thickness 2) into a support image. A black line (thickness 3) is carved into the smoothed mask between two corners when they are > 40 px apart and ≥ 75 % of the Bresenham path between them lies on the support image.

The notebook's angle comparison rounds the cosine to −1, 0 or 1 before `acos`. The port keeps this behaviour, so the 3° test in step 4 effectively accepts segments up to about 60° apart.

### Geometric Filtering
Keeps 8-connected components larger than 1000 px and removes the rest. GF was not part of the original notebook.

These LSA and GF parameters are fixed in `PostProcessParameters`. The `preprocessing`, `hough`, `line_grouping`, `intersections`, `line_verification` and `component_filtering` sections of `configs/default.yaml` list the same values but are **not read** by the pipeline. Only the `training` section is used.

### Evaluation protocol (`evaluation/evaluator.py`)
Per page:
1. Foreground is `mask > 0`. Panels are connected components (8-connectivity) of the prediction and of the ground truth.
2. Predicted and ground-truth panels are matched one-to-one with the Hungarian algorithm, maximising IoU.
3. A matched pair counts as a true positive (TP) if its Dice ≥ 0.90.

| Key in `metrics.json` | Paper | Per-page definition |
|---|---|---|
| `mean_iou` (= `canonical_instance_miou`) | mIoU | Mean IoU of each GT panel with its matched prediction (0 if unmatched) |
| `mean_dice` | mDice | Same with Dice |
| `precision` | PPr | TP / number of predicted panels |
| `recall` | PRc | TP / number of GT panels |
| `f1` | PF1 | 2·PPr·PRc / (PPr + PRc) on the page (0 if both are 0) |
| `panel_accuracy` | | TP / number of GT panels |
| `page_accuracy` | | 1 if every GT panel is found, else 0 |
| `strict_page_accuracy` | | 1 if every GT panel is found and there are no false positives, else 0 |

Dataset values are the **mean of the per-page values**, including F1. Reported F1 is therefore not, in general, the harmonic mean of the reported precision and recall. `metrics.json` also includes the totals `total_tp`, `total_fp`, `total_fn`, `total_gt`, `total_pred`, summed over pages.

---

## Repository Layout

```
configs/default.yaml              Default configuration (only `training` is used, see above)
src/comics_panel_extraction/
├── cli.py                        `comics-panel-extraction` command (infer, postprocess, evaluate, train)
├── model/unetpp.py               U-Net++ architecture
├── data/loader.py                Dataset loading and tf.data pipeline
├── training/                     Loss, warm-up + cosine schedule, training loop
├── inference/                    Checkpoint loading and prediction
├── postprocessing/line_analysis.py  LSA and GF
├── evaluation/                   Instance matching and metrics
├── io/image_io.py                Mask and tensor read/write helpers with format checks
└── config/models.py              Configuration dataclasses and YAML loader
tests/unit/                       Unit tests (pytest)
```

Run the tests with:
```bash
pytest
```

---

## Known Issues

- **Importing the package loads TensorFlow.** `import comics_panel_extraction` imports the model and TensorFlow, even if you only need post-processing or evaluation.

---

## Reproducibility

This repository reconstructs the model architecture, training pipeline, line segment analysis and evaluation protocol described in the paper. The trained models reported in the manuscript are not bundled because of storage and distribution constraints. Models can be trained from scratch with the provided training pipeline and configuration.

---

## Citation

```bibtex
@inproceedings{nguyen2025comic,
  title={Comic Panel Extraction using U-Net++ Deep Learning Model and Line Segment Analysis},
  author={Nguyen, Quang Huy and Le, Linh Long and Nghiem, Thi Phuong and Pham, Thai Son and Nguyen, Tuan Khai and Nguyen, Hai Dang and Tran, Giang Son},
  booktitle={Proceedings of the International Conference on Research in Intelligent and Computing in Engineering (RIVF)},
  year={2025}
}
```
