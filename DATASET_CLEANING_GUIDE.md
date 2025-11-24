# Dataset Cleaning Guide: ImageNet & COCO
## For Kushagra and Saeed - Visual Layer Integration

### Overview
This guide covers the complete workflow for cleaning ImageNet1K and COCO datasets using Visual Layer outputs, user tagging, and train-test leak detection.

---

## Prerequisites

### 1. Install Required Packages
```bash
pip install imagehash pillow pandas numpy torch torchvision tqdm
```

### 2. Access Visual Layer
- Use Guy's working login link (check team email)
- Log in with your @uci.edu email
- Verify access to the VL platform

### 3. Dataset Locations
- **ImageNet1K**: `evanarlian/imagenet_1k_resized_256` on HuggingFace
- **COCO**: TBD (check with team on Drive)

---

## Workflow

### Phase 1: Visual Layer Analysis

#### Step 1.1: Upload Dataset to Visual Layer
```
1. Log into Visual Layer platform
2. Create new project: "ImageNet1K_Cleaning"
3. Upload/connect to ImageNet1K dataset
   - If using HF dataset, may need to download locally first
   - Or point VL to your data directory
4. Run VL's automated analysis:
   - Mislabel detection
   - Outlier detection
   - Duplicate detection
   - Quality assessment
5. Wait for analysis to complete (may take hours for ImageNet)
```

#### Step 1.2: Export Visual Layer Results
```
1. In VL dashboard, go to "Issues" or "Flagged Samples"
2. Export all flagged items as CSV
3. Save as: imagenet1k_vl_export.csv
4. Download to: ~/capstone_project_Visual-Layer/capstone_project_visual_layer/vl_exports/
```

**Expected VL export columns** (adapt script if different):
- `image_id` or `id`: Unique identifier
- `image_path` or `path`: Path to image file
- `label` or `class`: Original class label
- `issue_type` or `flag`: Type of issue detected
- `confidence`: VL's confidence score
- `suggested_label`: VL's suggested correction (if any)
- `reason`: Explanation of why flagged

---

### Phase 2: User Tagging

#### Step 2.1: Create Initial Tagging Manifest
```bash
cd ~/capstone_project_Visual-Layer/capstone_project_visual_layer

python vl_tagging_workflow.py \
    --vl_export vl_exports/imagenet1k_vl_export.csv \
    --dataset imagenet1k \
    --output manifests/imagenet1k_tagged_manifest.csv \
    --generate_html
```

This creates:
- `manifests/imagenet1k_tagged_manifest.csv` - Structured manifest for tagging
- `manifests/review_interface.html` - Web interface for manual review

#### Step 2.2: Review and Tag Images

**Option A: Use HTML Interface**
```bash
open manifests/review_interface.html
# Review images in browser, make notes
```

**Option B: Batch Tagging in Python**
```python
from vl_tagging_workflow import VLTaggingWorkflow

# Load the workflow
workflow = VLTaggingWorkflow('vl_exports/imagenet1k_vl_export.csv', 'imagenet1k')
workflow.load_vl_export()
workflow.create_tagging_manifest()

# Example: Tag all mislabels that VL has high confidence on
high_conf_mislabels = workflow.vl_data[
    (workflow.vl_data['vl_flag_type'] == 'mislabel') &
    (workflow.vl_data['vl_confidence'] > 0.9)
]['image_id'].tolist()

workflow.add_user_tags_batch(
    image_ids=high_conf_mislabels,
    tags=['mislabel_confirmed'],
    action='relabel',
    reviewer_name='Kushagra'  # or 'Saeed'
)

# Example: Tag outliers that should be kept
valid_outliers = [123, 456, 789]  # Image IDs after manual review
workflow.add_user_tags_batch(
    image_ids=valid_outliers,
    tags=['outlier_valid', 'keep'],
    action='keep',
    reviewer_name='Saeed'
)

# Export updated manifest
workflow.export_manifest('manifests/imagenet1k_tagged_manifest_v2.csv')
```

#### User Tag Taxonomy
Use these standardized tags:
- `mislabel_confirmed`: Definitely wrong label
- `mislabel_uncertain`: Possibly wrong, needs discussion
- `outlier_valid`: Edge case but correct
- `outlier_invalid`: Outlier AND wrong label
- `duplicate_exact`: Exact duplicate
- `duplicate_near`: Near duplicate (>95% similar)
- `low_quality_blur`: Blurry/low resolution
- `low_quality_corrupt`: Corrupted file
- `ambiguous_class`: Could belong to multiple classes
- `train_test_leak`: Appears in both splits
- `keep`: Keep in dataset
- `remove`: Remove from dataset
- `relabel`: Change label to different class

---

### Phase 3: Train-Test Leak Detection

#### Step 3.1: Run Leak Detection on ImageNet1K

**Quick Method (Perceptual Hash Only - Fast)**
```bash
python train_test_leak_detection.py \
    --dataset imagenet1k \
    --train_dir ~/capstone_project_Visual-Layer/capstone_project_visual_layer/data/imagenet_official/train \
    --test_dir ~/capstone_project_Visual-Layer/capstone_project_visual_layer/data/imagenet_official/val \
    --output manifests/imagenet1k_leaks_report.csv \
    --methods perceptual \
    --phash_threshold 5
```

**Comprehensive Method (All Methods - Slow but Thorough)**
```bash
python train_test_leak_detection.py \
    --dataset imagenet1k \
    --train_dir ~/capstone_project_Visual-Layer/capstone_project_visual_layer/data/imagenet_official/train \
    --test_dir ~/capstone_project_Visual-Layer/capstone_project_visual_layer/data/imagenet_official/val \
    --output manifests/imagenet1k_leaks_report.csv \
    --methods exact perceptual semantic \
    --phash_threshold 5 \
    --semantic_threshold 0.95
```

**For HuggingFace ImageNet1K:**
```python
# First, download and organize dataset locally
from datasets import load_dataset
import shutil
from pathlib import Path

# Load dataset
train_ds = load_dataset("evanarlian/imagenet_1k_resized_256", split="train")
val_ds = load_dataset("evanarlian/imagenet_1k_resized_256", split="val")

# Save images to disk organized by split
train_dir = Path("data/imagenet1k_for_leak_check/train")
val_dir = Path("data/imagenet1k_for_leak_check/val")

print("Saving train images...")
for i, sample in enumerate(train_ds):
    label = sample['label']
    img_dir = train_dir / str(label)
    img_dir.mkdir(parents=True, exist_ok=True)
    sample['image'].save(img_dir / f"{i}.jpg")
    if i % 10000 == 0:
        print(f"Saved {i}/{len(train_ds)} images")

print("Saving val images...")
for i, sample in enumerate(val_ds):
    label = sample['label']
    img_dir = val_dir / str(label)
    img_dir.mkdir(parents=True, exist_ok=True)
    sample['image'].save(img_dir / f"{i}.jpg")

# Then run leak detection
# python train_test_leak_detection.py --dataset imagenet1k --train_dir data/imagenet1k_for_leak_check/train --test_dir data/imagenet1k_for_leak_check/val --output manifests/imagenet1k_leaks_report.csv --methods perceptual
```

#### Step 3.2: Review Leak Detection Results
```bash
# View summary
cat manifests/imagenet1k_leaks_report_summary.txt

# Open CSV for detailed review
# Column format:
# train_image, test_image, method, similarity, leak_type, [hamming_distance]
```

#### Step 3.3: Merge Leak Detection with Tagging Manifest
```python
import pandas as pd

# Load tagging manifest
manifest = pd.read_csv('manifests/imagenet1k_tagged_manifest.csv')

# Load leak report
leaks = pd.read_csv('manifests/imagenet1k_leaks_report.csv')

# Add leak flags to manifest
test_images_with_leaks = set(leaks['test_image'].tolist())

def flag_leaks(row):
    if row['image_path'] in test_images_with_leaks:
        # Add train_test_leak tag
        existing_tags = row['user_tags'] if pd.notna(row['user_tags']) else ''
        new_tags = existing_tags + ',train_test_leak' if existing_tags else 'train_test_leak'
        return new_tags
    return row['user_tags']

manifest['user_tags'] = manifest.apply(flag_leaks, axis=1)

# Mark leak images for removal
manifest.loc[manifest['image_path'].isin(test_images_with_leaks), 'user_action'] = 'remove'

# Save updated manifest
manifest.to_csv('manifests/imagenet1k_final_cleaning_manifest.csv', index=False)
```

---

### Phase 4: Apply Cleaning and Prepare Clean Dataset

#### Step 4.1: Create Clean Dataset Based on Manifest
```python
import pandas as pd
from pathlib import Path
import shutil

# Load final manifest
manifest = pd.read_csv('manifests/imagenet1k_final_cleaning_manifest.csv')

# Filter for images to remove
to_remove = manifest[manifest['user_action'] == 'remove']
to_relabel = manifest[manifest['user_action'] == 'relabel']

print(f"Images to remove: {len(to_remove)}")
print(f"Images to relabel: {len(to_relabel)}")

# Create cleaned dataset directory
cleaned_dir = Path('data/imagenet1k_cleaned')
cleaned_dir.mkdir(parents=True, exist_ok=True)

# Option 1: Create removal list for training script
removal_list = to_remove['image_id'].tolist()
with open('manifests/imagenet1k_removal_list.txt', 'w') as f:
    for img_id in removal_list:
        f.write(f"{img_id}\n")

# Option 2: Create relabeling mapping
relabel_map = {}
for _, row in to_relabel.iterrows():
    if pd.notna(row['new_label']):
        relabel_map[row['image_id']] = row['new_label']

import json
with open('manifests/imagenet1k_relabel_map.json', 'w') as f:
    json.dump(relabel_map, f, indent=2)

print("\nCleaning artifacts created:")
print("- imagenet1k_removal_list.txt")
print("- imagenet1k_relabel_map.json")
```

#### Step 4.2: Update Training Script to Use Cleaned Data
Add to your training notebook/script:

```python
import json

# Load cleaning artifacts
with open('manifests/imagenet1k_removal_list.txt', 'r') as f:
    removal_ids = set(int(line.strip()) for line in f)

with open('manifests/imagenet1k_relabel_map.json', 'r') as f:
    relabel_map = json.load(f)

# Modify dataset class
class CleanedImageNetDataset(Dataset):
    def __init__(self, hf_dataset, removal_ids, relabel_map, transform=None):
        self.dataset = hf_dataset
        self.removal_ids = removal_ids
        self.relabel_map = relabel_map
        self.transform = transform

        # Filter valid indices
        self.valid_indices = [
            i for i in range(len(hf_dataset))
            if i not in removal_ids
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        sample = self.dataset[real_idx]
        image = sample['image'].convert('RGB')

        # Use relabeled label if available, otherwise original
        label = self.relabel_map.get(str(real_idx), sample['label'])

        if self.transform:
            image = self.transform(image)

        return image, label

# Use in training
cleaned_dataset = CleanedImageNetDataset(
    full_train_dataset,
    removal_ids,
    relabel_map,
    transform=train_transform
)
```

---

### Phase 5: Repeat for COCO

Follow same workflow for COCO:
1. Upload COCO to Visual Layer
2. Export VL results
3. Run tagging workflow: `python vl_tagging_workflow.py --vl_export vl_exports/coco_vl_export.csv --dataset coco --output manifests/coco_tagged_manifest.csv`
4. Run leak detection: `python train_test_leak_detection.py --dataset coco --train_dir path/to/coco/train --test_dir path/to/coco/val --output manifests/coco_leaks_report.csv`
5. Create cleaned dataset

---

## Team Coordination

### Division of Work
**Kushagra:**
- Visual Layer upload and export for ImageNet1K
- Leak detection on ImageNet1K
- Integration with training scripts

**Saeed:**
- Visual Layer upload and export for COCO
- Leak detection on COCO
- User tagging and manifest creation

**Together:**
- Review flagged images
- Decide on tagging standards
- Validate cleaning results

### Sync Points
1. After VL exports ready - compare findings
2. After initial tagging - cross-review each other's tags
3. After leak detection - discuss unexpected results
4. Before training - verify cleaned datasets

---

## Deliverables for Guy/Team

### What to Upload to Google Drive

Create folder: `Dataset_Cleaning_Results/`

**1. Manifests/**
- `imagenet1k_vl_export.csv` - Raw VL export
- `imagenet1k_tagged_manifest.csv` - User-tagged manifest
- `imagenet1k_leaks_report.csv` - Train-test leak report
- `imagenet1k_final_cleaning_manifest.csv` - Final combined manifest
- Same files for COCO

**2. Summary_Reports/**
- `imagenet1k_cleaning_summary.txt` - Statistics and findings
- `coco_cleaning_summary.txt`
- Comparison doc between datasets

**3. Cleaned_Datasets/**
- `imagenet1k_removal_list.txt` - Images to exclude
- `imagenet1k_relabel_map.json` - Relabeling instructions
- Same for COCO

**4. Presentation/**
- Slides showing:
  - Number of issues found by VL
  - Breakdown by issue type
  - Train-test leak statistics
  - Impact on dataset size
  - Example flagged images
  - Cleaning methodology

---

## Troubleshooting

### VL Export Format Different
If VL exports different columns, modify `vl_tagging_workflow.py`:
```python
# In create_tagging_manifest(), update column mapping:
manifest = pd.DataFrame({
    'image_id': self.vl_data.get('your_id_column'),
    'image_path': self.vl_data.get('your_path_column'),
    # ... etc
})
```

### Leak Detection Too Slow
For large datasets, run in stages:
```bash
# First run exact duplicates (fast)
python train_test_leak_detection.py --methods exact ...

# Then perceptual (medium)
python train_test_leak_detection.py --methods perceptual ...

# Finally semantic if needed (slow)
python train_test_leak_detection.py --methods semantic ...
```

### Out of Memory During Leak Detection
Reduce batch size:
```bash
# Edit train_test_leak_detection.py line ~300
batch_size = 16  # Instead of 32
```

---

## Timeline Estimate

**For ImageNet1K (~1.2M images):**
- VL analysis: 4-8 hours (automated)
- Export and manifest creation: 30 min
- Manual review/tagging: 2-4 hours
- Leak detection (perceptual): 2-3 hours
- Leak detection (semantic): 8-12 hours
- Combining and finalizing: 1 hour

**Total: ~1-2 days** (mostly automated, can run overnight)

**For COCO:**
- Similar timeline but depends on COCO size

---

## Questions for Guy/Team

Before starting, clarify:
1. ✅ Which COCO split/version to use?
2. ✅ Should we clean train only, or train+val?
3. ✅ What's the threshold for "too many removals"? (e.g., if VL flags 10% of data)
4. ✅ Do we need to document every single flagged image, or just statistics?
5. ✅ Should cleaned datasets be uploaded somewhere, or just manifests?

---

## Next Steps After Cleaning

1. **Train models on cleaned datasets**
   - Compare clean vs original performance
   - Document accuracy improvements

2. **Create comparison experiments**
   - Exp 1: Original data
   - Exp 2: VL-cleaned data
   - Exp 3: VL-cleaned + manual review

3. **Generate results for paper**
   - Loss curves on cleaned data
   - Accuracy comparisons
   - Impact analysis

Good luck! 🚀
