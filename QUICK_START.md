# Quick Start Guide - Dataset Cleaning Task
**For: Kushagra & Saeed**

## 🎯 Your Mission
Clean ImageNet1K and COCO datasets using Visual Layer + manual tagging + leak detection

## 📋 Checklist (Copy to your notes!)

### Today
- [ ] Read `DATASET_CLEANING_GUIDE.md` (15 min)
- [ ] Read `ANALYSIS_SUMMARY.md` (10 min)
- [ ] Access Visual Layer - verify login works
- [ ] Coordinate with Saeed - who does ImageNet, who does COCO?
- [ ] Install dependencies: `pip install imagehash pillow pandas numpy torch torchvision tqdm`

### Phase 1: Visual Layer (4-8 hours automated)
- [ ] Log into Visual Layer with Guy's link
- [ ] Upload ImageNet1K to VL platform
- [ ] Run VL automated analysis (mislabels, outliers, duplicates)
- [ ] Wait for analysis (can run overnight)
- [ ] Export VL results as CSV to `vl_exports/imagenet1k_vl_export.csv`

### Phase 2: Tagging (2-4 hours active)
- [ ] Run: `python vl_tagging_workflow.py --vl_export vl_exports/imagenet1k_vl_export.csv --dataset imagenet1k --output manifests/imagenet1k_tagged.csv --generate_html`
- [ ] Open `review_interface.html` in browser
- [ ] Review flagged images and add user tags
- [ ] Cross-check tags with Saeed
- [ ] Export final tagged manifest

### Phase 3: Leak Detection (2-12 hours automated)
- [ ] Quick version: `python train_test_leak_detection.py --dataset imagenet1k --train_dir data/train --test_dir data/val --output manifests/imagenet1k_leaks.csv --methods perceptual`
- [ ] Review leak report
- [ ] Merge leaks into tagging manifest

### Phase 4: Deliverables (1-2 hours)
- [ ] Create final cleaning manifest
- [ ] Generate summary statistics
- [ ] Create presentation slides
- [ ] Upload everything to Google Drive folder: `Dataset_Cleaning_Results/`

## 🔑 Key Files You Created Today

| File | Purpose | When to Use |
|------|---------|-------------|
| `vl_tagging_workflow.py` | Process VL exports, add user tags | After VL export ready |
| `train_test_leak_detection.py` | Find duplicates between train/test | Before final cleaning |
| `DATASET_CLEANING_GUIDE.md` | Complete step-by-step guide | Reference throughout |
| `ANALYSIS_SUMMARY.md` | What you've done + what's next | Big picture understanding |

## ⚡ Copy-Paste Commands

```bash
# Navigate to project
cd ~/capstone_project_Visual-Layer/capstone_project_visual_layer

# Create directories
mkdir -p vl_exports manifests

# Run tagging (after VL export)
python vl_tagging_workflow.py \
    --vl_export vl_exports/imagenet1k_vl_export.csv \
    --dataset imagenet1k \
    --output manifests/imagenet1k_tagged.csv \
    --generate_html

# Run leak detection (quick)
python train_test_leak_detection.py \
    --dataset imagenet1k \
    --train_dir data/imagenet_official/train \
    --test_dir data/imagenet_official/val \
    --output manifests/imagenet1k_leaks.csv \
    --methods perceptual \
    --phash_threshold 5

# Run leak detection (thorough, overnight)
python train_test_leak_detection.py \
    --dataset imagenet1k \
    --train_dir data/imagenet_official/train \
    --test_dir data/imagenet_official/val \
    --output manifests/imagenet1k_leaks.csv \
    --methods exact perceptual semantic \
    --phash_threshold 5 \
    --semantic_threshold 0.95
```

## 📊 What to Upload to Google Drive

```
Dataset_Cleaning_Results/
├── imagenet1k_vl_export.csv          (from Visual Layer)
├── imagenet1k_tagged_manifest.csv    (after your tagging)
├── imagenet1k_leaks_report.csv       (from leak detection)
├── imagenet1k_cleaning_summary.txt   (statistics)
├── imagenet1k_removal_list.txt       (for training)
├── imagenet1k_relabel_map.json       (for training)
└── cleaning_results_slides.pdf       (for presentation)
```

## 🎨 User Tags (Standardized)

**Use these exact tags:**
- `mislabel_confirmed` - Definitely wrong
- `mislabel_uncertain` - Maybe wrong
- `outlier_valid` - Unusual but correct
- `outlier_invalid` - Unusual AND wrong
- `duplicate_exact` - Pixel-perfect duplicate
- `duplicate_near` - 95%+ similar
- `low_quality_blur` - Blurry/low res
- `low_quality_corrupt` - Corrupted file
- `ambiguous_class` - Could be multiple classes
- `train_test_leak` - In both train and test
- `keep` - Flag but keep
- `remove` - Delete from dataset
- `relabel` - Change to different class

## ⏱️ Time Budget

| Task | Kushagra | Saeed | Can Run Overnight? |
|------|----------|-------|-------------------|
| VL upload & analysis | 4-8 hrs | 4-8 hrs | ✅ Yes |
| Export & setup | 30 min | 30 min | ❌ No |
| Manual tagging | 2-4 hrs | 2-4 hrs | ❌ No |
| Leak detection (perceptual) | 2-3 hrs | 2-3 hrs | ✅ Yes |
| Leak detection (semantic) | 8-12 hrs | 8-12 hrs | ✅ Yes |
| Combining & docs | 1-2 hrs | 1-2 hrs | ❌ No |

**Total per person: ~12-20 hours over 2-3 days**

## 🤝 Team Coordination

**Kushagra:**
- ImageNet1K: VL export, leak detection, training integration

**Saeed:**
- COCO: VL export, leak detection, tagging coordination

**Together:**
- Cross-validate tags
- Discuss edge cases
- Create presentation

## 🆘 If Something Breaks

### VL export has different columns
- Edit `vl_tagging_workflow.py` line 74-85
- Update column mapping to match your VL export

### Leak detection out of memory
- Edit `train_test_leak_detection.py` line ~300
- Change `batch_size = 32` to `batch_size = 16`

### Can't find image files
- Check paths in leak detection command
- Make sure using absolute paths, not relative

### Scripts hang/freeze
- Use Ctrl+C to cancel
- Check you're not running on your entire ImageNet (1.2M images) without testing on subset first

## ✅ Success Looks Like

**End of Week:**
- VL analysis complete for both datasets
- Tagged manifests with your validation
- Leak reports generated
- Cleaning statistics documented
- Ready to present findings to Guy

**Quality Metrics:**
- Found X mislabels (VL confidence > 0.9)
- Detected Y train-test leaks (Z% of test set)
- Manually reviewed and tagged A high-priority issues
- Final cleaned dataset: B images removed, C relabeled

## 💬 Communication

**Team Chat Updates (Daily):**
- "VL analysis started for ImageNet1K - ETA 6 hours"
- "Leak detection running overnight - check results tomorrow"
- "Tagged 500 images, found 50 confirmed mislabels"

**Questions for Guy:**
- "VL flagged 15% of ImageNet as potential issues - should we be more conservative?"
- "Found 200 exact duplicates in train-test - remove all from test?"

## 🎓 Learning Outcomes

After this task, you'll understand:
- How real-world datasets have quality issues
- Automated vs manual validation trade-offs
- Train-test contamination detection
- Dataset versioning and documentation
- Collaboration on data cleaning at scale

## 🚀 Ready? GO!

1. Open `DATASET_CLEANING_GUIDE.md` in one window
2. Open Visual Layer in browser
3. Open terminal in project directory
4. Start with Phase 1: VL upload
5. Check off items in checklist above
6. Update team in group chat

**You got this! 💪**

---

**Pro tip:** Do a test run on 1000 images first to validate your workflow before running on full datasets!
