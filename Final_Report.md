# Image Classification Assignment Report

## Part 1: CIFAR-10 Sub-Dataset (5 Classes)

### Dataset Preparation
We downloaded the official CIFAR-10 dataset and filtered it to keep only 5 classes: **airplane, automobile, bird, cat, and deer**.
The resulting dataset was then split using a stratified 70% Train, 15% Validation, and 15% Test configuration.

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| airplane | 4,250 | 750 | 1,000 | 6,000 |
| automobile | 4,250 | 750 | 1,000 | 6,000 |
| bird | 4,250 | 750 | 1,000 | 6,000 |
| cat | 4,250 | 750 | 1,000 | 6,000 |
| deer | 4,250 | 750 | 1,000 | 6,000 |
| **Total** | **21,250** | **3,750** | **5,000** | **30,000** |

### Training Results (MobileNetV2)
The pretrained MobileNetV2 model was fine-tuned for 15 epochs. 
- **Phase 1 (Warm-up)**: Epochs 1-5 with the backbone frozen.
- **Phase 2 (Fine-tuning)**: Epochs 6-15 with the entire network unfrozen.

**Final Evaluation on Test Set (5,000 images):**
- **Accuracy:** 80.50%
- **Precision (macro):** 80.95%
- **Recall (macro):** 80.50%

**Per-Class Metrics:**

| Class | Precision | Recall |
|---|---|---|
| airplane | 85.64% | 87.50% |
| automobile | 91.14% | 91.70% |
| bird | 72.77% | 69.80% |
| cat | 69.11% | 66.70% |
| deer | 86.09% | 86.80% |

> **Note on Averaging:** Precision and recall are reported using **macro-averaging**, which computes the metric independently for each class and then takes the unweighted mean. This treats all classes equally regardless of sample count, which is appropriate here since the dataset is balanced (1,000 test samples per class).

![CIFAR-10 Training Plot](part1_cifar10/training_plot.png)
![CIFAR-10 Confusion Matrix](part1_cifar10/confusion_matrix.png)

---

## Part 2: Guitar vs Sitar Classification

### Dataset Preparation
The custom dataset was processed and split into Train, Validation, and Test sets based on the provided class folders.

| Class | Train | Val | Test |
|---|---|---|---|
| Guitar | 765 | 135 | 100 |
| Sitar | 765 | 135 | 101 |

### Training Results (MobileNetV2)
The model was fine-tuned for 10 epochs.

**Final Evaluation on Test Set (201 images):**
- **Accuracy:** 97.01%
- **Precision (macro):** 97.10%
- **Recall (macro):** 97.00%

**Per-Class Metrics:**

| Class | Precision | Recall |
|---|---|---|
| Guitar | 98.00% | 96.00% |
| Sitar | 96.19% | 98.02% |

> **Note on Averaging:** Same macro-averaging approach as Part 1. With nearly balanced test splits (100 vs 101 images), macro and weighted averages produce nearly identical results.

![Guitar vs Sitar Confusion Matrix](part2_guitar_sitar/confusion_matrix.png)

### Inference (OpenCV Overlay)
The inference script outputs a probability for **every class** with a Yes/No label overlaid on the image using OpenCV, matching the assignment example format:

```
Guitar: Yes (99.9%)
Sitar:  No  (0.1%)
```

The predicted class is displayed in green, and the remaining class in red, making the result immediately visually clear.

![Inference Output](pred_rProcess_Guitar_901.jpg)
