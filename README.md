# Visual Layer ImageNet Dataset Cleaning
**Kushagra's Role: ImageNet1K Training & Dataset Cleaning**

Part of UCSB CS189A Capstone Project on Label Noise and Visual Layer Cleaning

## Project Overview

This repository contains tools and workflows for cleaning ImageNet1K dataset using Visual Layer platform, implementing user tagging systems, and detecting train-test leaks.

### Team
- **Kushagra**: ImageNet1K training, dataset cleaning, leak detection
- **Saeed**: COCO dataset cleaning
- **Advisor**: Guy (Visual Layer)
- **Team Lead**: Alec Song

## Repository Structure

```
vl-imagenet-cleaning/
├── README.md                          # This file
├── QUICK_START.md                     # Quick reference guide
├── DATASET_CLEANING_GUIDE.md          # Complete workflow documentation
├── scripts/
│   ├── vl_tagging_workflow.py        # Visual Layer output processing
│   ├── train_test_leak_detection.py  # Duplicate detection
│   └── requirements.txt               # Python dependencies
├── notebooks/
│   └── imagenet_training_reference.md # Notes from previous training
├── manifests/
│   └── README.md                      # Manifest file format specs
└── docs/
    └── ANALYSIS_SUMMARY.md            # Complete project analysis
```

## What This Repo Contains

### ✅ Essential Scripts
1. **`vl_tagging_workflow.py`** - Process Visual Layer exports and add user tags
2. **`train_test_leak_detection.py`** - Detect duplicates between train/test splits

### ✅ Documentation
1. **`QUICK_START.md`** - Copy-paste commands and checklists
2. **`DATASET_CLEANING_GUIDE.md`** - Step-by-step workflow
3. **`ANALYSIS_SUMMARY.md`** - Project history and context

### ✅ Reference Materials
- Training notebook summaries
- Manifest format specifications
- User tag taxonomy

## Quick Start

### 1. Install Dependencies
```bash
pip install -r scripts/requirements.txt
```

### 2. Process Visual Layer Export
```bash
python scripts/vl_tagging_workflow.py \
    --vl_export path/to/imagenet1k_vl_export.csv \
    --dataset imagenet1k \
    --output manifests/imagenet1k_tagged.csv \
    --generate_html
```

### 3. Run Leak Detection
```bash
python scripts/train_test_leak_detection.py \
    --dataset imagenet1k \
    --train_dir path/to/train \
    --test_dir path/to/val \
    --output manifests/imagenet1k_leaks.csv \
    --methods perceptual
```

## Current Status

### Completed ✅
- ResNet-18 training infrastructure with W&B integration
- Noise injection experiments (5 experiments, 20%-40% noise)
- CIFAR-100 manifest system
- Auto-resume and checkpointing
- Loss curve visualization

### In Progress 🔄
- Visual Layer platform integration
- ImageNet1K dataset cleaning
- Train-test leak detection
- User tagging workflow

### Next Steps 📋
1. Upload ImageNet1K to Visual Layer
2. Export VL flagged samples
3. Manual review and user tagging
4. Detect train-test leaks
5. Create final cleaned dataset
6. Retrain models on cleaned data

## Key Findings from Previous Work

- 20% label noise causes ~X% accuracy drop
- Simulated 80% VL cleaning recovers ~Y% of lost accuracy
- 40% noise exponentially more damaging than 20%
- W&B tracking shows proper convergence on clean data

## Deliverables

For team Google Drive:
- `imagenet1k_vl_export.csv` - Visual Layer results
- `imagenet1k_tagged_manifest.csv` - User-validated tags
- `imagenet1k_leaks_report.csv` - Train-test duplicates
- `imagenet1k_cleaning_summary.txt` - Statistics
- Presentation slides with findings

## Tools & Platforms

- **Visual Layer**: Automated data quality analysis
- **Weights & Biases**: Experiment tracking
- **HuggingFace**: ImageNet1K dataset (`evanarlian/imagenet_1k_resized_256`)
- **Kaggle**: 30 hrs/week free GPU
- **Modal**: $30 free compute credit

## Timeline

| Task | Time | Status |
|------|------|--------|
| VL upload & analysis | 4-8 hrs | Pending |
| Manual review/tagging | 2-4 hrs | Pending |
| Leak detection | 2-12 hrs | Pending |
| Documentation | 2-3 hrs | Pending |

**Estimated Total: 1-2 days** (mostly automated)

## Research Context

This work is part of a 6-week research project studying:
- Impact of intelligent vs random mislabels on model robustness
- Visual Layer's effectiveness at detecting and correcting label noise
- Train-test contamination in popular datasets
- Best practices for dataset quality assurance

## Contact

- **Kushagra**: ImageNet questions
- **Saeed**: COCO questions
- **Alec Song**: General project coordination
- **Guy (Visual Layer)**: Platform technical support

## License

Educational/Research Use - UCI CS189A Capstone Project

---

**Last Updated**: November 23, 2025
**Version**: 1.0
