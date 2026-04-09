# PubMed RCT Classification - Optimized Training Pipeline

## Overview
SciBERT fine-tuning với warmup scheduler, class weights, mixed precision, gradient accumulation, và staged unfreezing.

## Model & Configuration

| Parameter | Value |
|-----------|-------|
| Model | `allenai/scibert_scivocab_uncased` |
| Max Length | 256 |
| Batch Size | 96 |
| Epochs | 50 |
| Freeze Epochs | 1 |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Warmup Ratio | 0.1 |
| Early Stopping Patience | 4 |

## Classes (5 labels)
- BACKGROUND (0)
- OBJECTIVE (1)
- METHODS (2)
- RESULTS (3)
- CONCLUSIONS (4)

## Optimization Techniques
1. **Mixed Precision (AMP)**: sử dụng bfloat16/float16 khi GPU hỗ trợ
2. **torch.compile**: model optimization với `max-autotune` mode
3. **Gradient Accumulation**: effective batch size = batch_size × accumulation_steps
4. **Class Weights**: `[2.24, 2.38, 0.61, 0.58, 1.30]` cho imbalanced data
5. **Warmup Scheduler**: linear schedule với 10% warmup
6. **Staged Unfreezing**: freeze backbone 1 epoch, sau đó unfreeze toàn bộ

## Data Statistics
- Train: 39,967 samples
- Val: 5,015 samples
- Test: 5,018 samples

## Training Progress

| Epoch | Train Loss | Train Acc | Train F1 | Val Loss | Val Acc | Val F1 |
|-------|------------|-----------|----------|----------|---------|--------|
| 1 (frozen) | 1.6195 | 0.2721 | 0.1637 | 1.5916 | 0.2700 | 0.1928 |
| 2 | 1.5723 | 0.3056 | 0.2706 | 1.5315 | 0.4467 | 0.3897 |
| 3 | 1.5053 | 0.4485 | 0.3980 | 1.4483 | 0.6195 | 0.5540 |
| 4 | 1.4188 | 0.5971 | 0.5299 | 1.3418 | 0.7288 | 0.6603 |
| 5 | 1.3184 | 0.6935 | 0.6208 | 1.2254 | 0.7795 | 0.7108 |
| ... | ... | ... | ... | ... | ... | ... |
| 25 | 0.6932 | 0.8270 | 0.7639 | 0.6268 | 0.8467 | **0.7926** |
| 29 | Early stopping triggered | | | | | |

## Final Test Results

| Metric | Score |
|--------|-------|
| Accuracy | 0.8434 |
| Macro F1-Score | 0.7814 |
| Macro Precision | 0.7835 |
| Macro Recall | 0.7823 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| BACKGROUND | 0.58 | 0.70 | 0.63 | 443 |
| OBJECTIVE | 0.69 | 0.64 | 0.66 | 395 |
| METHODS | 0.90 | 0.91 | 0.91 | 1662 |
| RESULTS | 0.90 | 0.89 | 0.90 | 1725 |
| CONCLUSIONS | 0.84 | 0.77 | 0.80 | 793 |

## Artifacts Generated
- `best_model_optimized.pth` - Best model checkpoint
- `training_curves_optimized.png` - Training/validation curves
- `confusion_matrix_optimized.png` - Confusion matrix visualization
- `per_class_metrics_optimized.png` - Per-class metrics bar chart

## Key Observations
1. Freeze backbone trong epoch đầu giúp classifier học trước khi fine-tune toàn bộ model
2. METHODS và RESULTS classes có hiệu suất cao nhất (F1 > 0.90)
3. BACKGROUND và OBJECTIVE classes khó phân loại hơn (F1 ~0.63-0.66)
4. Early stopping kích hoạt ở epoch 29 với patience=4