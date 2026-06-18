# ML Image Classification

Image classification project using **MobileNetV2** (transfer learning) for two tasks:

1. **CIFAR-10** — 5-class subset classification (airplane, automobile, bird, cat, deer)
2. **Guitar vs Sitar** — Binary classification on a custom scraped dataset

## Project Structure

```
├── part1_cifar10/                  # CIFAR-10 classification
│   ├── dataset_prep.py             # Download, filter to 5 classes, split
│   ├── train.py                    # Two-phase training (warm-up + fine-tune)
│   ├── evaluate.py                 # Test set metrics + confusion matrix
│   ├── inference.py                # Single image → per-class probabilities
│   ├── confusion_matrix.png        # Generated confusion matrix
│   └── training_plot.png           # Train/val loss & accuracy curves
│
├── part2_guitar_sitar/             # Guitar vs Sitar classification
│   ├── dataset_prep.py             # Organize dataset into train/val/test splits
│   ├── train.py                    # Two-phase MobileNetV2 fine-tuning
│   ├── evaluate.py                 # Test set metrics + confusion matrix
│   ├── inference.py                # Single image → OpenCV overlay with all class probs
│   └── confusion_matrix.png        # Generated confusion matrix
│
├── Final_Report.md                 # Detailed report with results and analysis
├── requirements.txt                # Python dependencies
└── README.md
```

## Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
pip install -r requirements.txt
```

## Part 1: CIFAR-10 (5 Classes)

### 1. Prepare Dataset
Downloads CIFAR-10, filters to 5 classes, and creates a stratified train/val/test split:
```bash
cd part1_cifar10
python dataset_prep.py
```

### 2. Train
Trains MobileNetV2 for 15 epochs (5 warm-up + 10 fine-tuning). Logs to TensorBoard:
```bash
python train.py
```

To view TensorBoard logs:
```bash
tensorboard --logdir=logs/
```

### 3. Evaluate
Calculates accuracy, precision, recall (macro + per-class) on the test set and generates a confusion matrix:
```bash
python evaluate.py
```

### 4. Inference
Run inference on a single image:
```bash
python inference.py <path_to_image>
```

### Results
| Metric | Score |
|---|---|
| Accuracy | 80.50% |
| Precision (macro) | 80.95% |
| Recall (macro) | 80.50% |

## Part 2: Guitar vs Sitar

### 1. Prepare Dataset
Organizes the raw dataset into `train/val/test` splits:
```bash
cd part2_guitar_sitar
python dataset_prep.py
```

### 2. Train
Fine-tunes MobileNetV2 for 10 epochs (3 warm-up + 7 fine-tuning):
```bash
python train.py
```

### 3. Evaluate
```bash
python evaluate.py
```

### 4. Inference
Runs inference and saves an output image with OpenCV overlay showing per-class probabilities:
```bash
python inference.py <path_to_image>
```

### Results
| Metric | Score |
|---|---|
| Accuracy | 97.01% |
| Precision (macro) | 97.10% |
| Recall (macro) | 97.00% |

## Approach

Both parts use the same transfer learning strategy with **MobileNetV2** (pretrained on ImageNet):

1. **Phase 1 (Warm-up):** Freeze the backbone, train only the classifier head with a higher learning rate (`1e-3`)
2. **Phase 2 (Fine-tuning):** Unfreeze all layers and train end-to-end with a lower learning rate (`1e-4`)

**Data Augmentation:**
- Part 1: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`
- Part 2: `RandomHorizontalFlip`, `RandomRotation(10)`

**Normalization:**
- Part 1: CIFAR-10 statistics (training at native 32×32 resolution)
- Part 2: ImageNet statistics (resized to 224×224)

## Dependencies

| Package | Version |
|---|---|
| PyTorch | ≥ 2.0.0 |
| TorchVision | ≥ 0.15.0 |
| TensorBoard | ≥ 2.14.0 |
| scikit-learn | ≥ 1.3.0 |
| OpenCV | ≥ 4.8.0 |
| Matplotlib | ≥ 3.7.0 |
| Seaborn | ≥ 0.12.0 |

## License

This project was created as part of an academic assignment.
