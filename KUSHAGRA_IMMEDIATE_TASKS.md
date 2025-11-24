# Kushagra's Immediate Tasks - Visual Layer Data Cleaning
**Based on Meeting Notes + Current VL State**

## 🎯 Your Mission (This Week)

Per Guy's instructions: **"Prepping/cleaning ImageNet by tagging VL's found mislabels/outliers/etc with user tags and doing train-test leak detection"**

---

## ✅ STEP 1: Export Duplicates from Visual Layer (DO THIS FIRST)

### What You See Now:
- ✅ "Found 14,851 duplicate images" in VL
- ✅ ImageNet 1K Enriched dataset loaded

### Actions:

#### Option A: Use VL Chat (Easiest)
```
In the VL Chat box (right side), type:

"Export all duplicate clusters as JSON including image paths, labels, and whether each image is in train or validation set"
```

#### Option B: Use Export Function
```
1. Click "Add Filter" button
2. Filter for "Duplicates"
3. Look for Export button (usually top right or in menu)
4. Choose JSON format
5. Save as: imagenet1k_vl_duplicates.json
```

### What the JSON Should Contain:
```json
{
  "clusters": [
    {
      "cluster_id": "cluster_001",
      "images": [
        {
          "id": "img_12345",
          "path": "train/n01440764/img_001.JPEG",
          "label": 0,
          "filename": "img_001.JPEG"
        },
        {
          "id": "img_67890",
          "path": "val/n01440764/img_999.JPEG",
          "label": 0,
          "filename": "img_999.JPEG"
        }
      ]
    }
  ]
}
```

**CRITICAL**: Guy's note says *"Contains val, duplicate = 1, means there is a train test leak because not present in training set"*

This means: If a duplicate cluster has images from BOTH train and val → **TRAIN-TEST LEAK**

---

## ✅ STEP 2: Analyze Duplicates for Train-Test Leaks

### Once you have the JSON:

```bash
cd ~/vl-imagenet-cleaning

# Run the new analyzer script
python scripts/analyze_vl_duplicates.py \
    --vl_json path/to/imagenet1k_vl_duplicates.json \
    --output manifests/imagenet1k_train_test_leaks.csv \
    --removal_list manifests/imagenet1k_val_removal_list.txt \
    --multi_label_report manifests/imagenet1k_multi_label_clusters.csv
```

### What This Does:
1. ✅ Checks each duplicate cluster for train AND val images
2. ✅ Identifies train-test leaks
3. ✅ Finds multi-label clusters (potential mislabels)
4. ✅ Creates removal list for validation images
5. ✅ Generates actionable reports

---

## ✅ STEP 3: Export Mislabels from Visual Layer

### VL Chat Query:
```
"Show all images flagged as potential mislabels with confidence scores above 0.7"
```

or

```
"Export all quality issues including mislabels and outliers as CSV"
```

### Export As:
- `imagenet1k_vl_mislabels.csv`

Expected columns:
- `image_id`
- `image_path`
- `current_label`
- `suggested_label`
- `confidence`
- `issue_type` (mislabel/outlier/quality)

---

## ✅ STEP 4: Tag with User Tags (Per Drive Schema)

### Meeting Note: "Tag VL's found mislabels/outliers with our created user tags (see drive)"

**ACTION NEEDED**: Check your Google Drive for the tagging schema

Likely structure:
```
User Tags Schema (from Drive):
- mislabel_confirmed
- mislabel_uncertain
- outlier_valid
- outlier_invalid
- ambiguous_primary_label
- multi_label
- background_noise
- crop_needed
- remove
- keep
- relabel_to_X
```

### Use the Tagging Script:
```bash
python scripts/vl_tagging_workflow.py \
    --vl_export imagenet1k_vl_mislabels.csv \
    --dataset imagenet1k \
    --output manifests/imagenet1k_tagged_manifest.csv \
    --generate_html
```

Then:
1. Open `manifests/review_interface.html`
2. Review flagged images
3. Apply user tags based on Drive schema
4. Export tagged manifest

---

## ✅ STEP 5: Address Specific Issues (Per Meeting Notes)

### A. Remove Clear Mislabels
```
Meeting note: "Remove clear mislabels and improper duplicates"
```

**Clear mislabels** = VL confidence > 0.9 AND manual verification confirms

Action:
- Tag with `mislabel_confirmed` + `remove`
- Add to removal list

### B. Handle Multi-Label Clusters
```
Meeting note: "Several clusters with exact duplicates, but they all have different labels"
```

Decision tree:
1. If ALL images in cluster are duplicates with DIFFERENT labels:
   - Tag as `multi_label` + `ambiguous_class`
   - Action: Pick most common label OR remove entire cluster

2. If cluster has mix:
   - Review manually
   - Keep best labeled one, remove rest

### C. Primary Label Issue
```
Meeting note: "Ex. image of dog with other objects, want the model to recognize most prominent object but does not output primary label"
```

For COCO (Rushil/Bhavya will handle):
- Crop objects from image
- Score classification

For ImageNet:
- Tag with `ambiguous_primary_label`
- Decide: keep or remove based on prominence

---

## ✅ STEP 6: Create Final Cleaning Artifacts

### After tagging, generate:

```python
# Run this after tagging is complete
import pandas as pd
import json

# Load tagged manifest
manifest = pd.read_csv('manifests/imagenet1k_tagged_manifest.csv')

# Create removal list (all tagged with 'remove')
to_remove = manifest[manifest['user_action'] == 'remove']['image_id'].tolist()
with open('manifests/imagenet1k_final_removal_list.txt', 'w') as f:
    for img_id in to_remove:
        f.write(f"{img_id}\n")

# Create relabel map
to_relabel = manifest[manifest['user_action'] == 'relabel']
relabel_map = {}
for _, row in to_relabel.iterrows():
    relabel_map[str(row['image_id'])] = int(row['new_label'])

with open('manifests/imagenet1k_relabel_map.json', 'w') as f:
    json.dump(relabel_map, f, indent=2)

print(f"✓ Removal list: {len(to_remove)} images")
print(f"✓ Relabel map: {len(relabel_map)} images")
```

---

## 📊 DELIVERABLES (For Next Meeting)

### 1. Train-Test Leak Report
- `imagenet1k_train_test_leaks.csv`
- Shows all duplicate clusters with train AND val images
- Recommended actions

### 2. Removal Lists
- `imagenet1k_val_removal_list.txt` - Val images to remove (from leaks)
- `imagenet1k_final_removal_list.txt` - All images to remove (leaks + mislabels)

### 3. Tagged Manifest
- `imagenet1k_tagged_manifest.csv`
- All VL findings + your user tags
- Actions decided (keep/remove/relabel)

### 4. Multi-Label Report
- `imagenet1k_multi_label_clusters.csv`
- Clusters with conflicting labels
- For manual review

### 5. Summary Statistics
```
Total ImageNet samples: 1,281,167
Duplicates found: 14,851
Train-test leaks: X
Mislabels confirmed: Y
Images to remove: Z
Images to relabel: W
Clean dataset size: 1,281,167 - Z
```

---

## 🤝 Coordination with Saeed

### Division of Work:
**You (Kushagra)**: ImageNet
**Saeed**: COCO

### Shared:
- Tagging schema (from Drive)
- Review methodology
- Cross-validate decisions

### Sync Points:
1. After VL exports ready - compare findings
2. After initial tagging - review each other's tags
3. Before finalizing - align on ambiguous cases

---

## ⏱️ Timeline

**TODAY (Sunday):**
- [x] Export duplicates from VL → 30 min
- [ ] Run leak analysis → 1 hour
- [ ] Export mislabels from VL → 30 min

**MONDAY:**
- [ ] Tag mislabels/outliers → 3-4 hours
- [ ] Review multi-label clusters → 1 hour
- [ ] Generate removal lists → 30 min

**TUESDAY:**
- [ ] Cross-review with Saeed → 1 hour
- [ ] Finalize manifests → 1 hour
- [ ] Create summary stats → 30 min
- [ ] Upload to Google Drive → 30 min

**WEDNESDAY (Meeting Day):**
- [ ] Present findings to Guy/team
- [ ] Discuss ambiguous cases
- [ ] Get approval on removal strategy

---

## 🚨 IMPORTANT NOTES (From Meeting)

### Guy's Key Points:

1. **Train-Test Leaks**:
   > "Contains val, duplicate = 1, means there is a train test leak because not present in training set"

   **Action**: Remove ALL val images that appear in train

2. **Multi-Label Clusters**:
   > "Can easily see if a cluster has multiple labels if the cluster has more than one label"

   **Action**: Review manually, decide per cluster

3. **Validation Images**:
   > "Validation images have 'val' in filename"

   **How to detect**: Check if 'val' in path

4. **Priority**:
   > "HIGH PRIORITY: get the preprocessing pipeline down"

   **Focus**: Clean data first, pipeline second

### What NOT to Do:
- ❌ Don't remove training images (only val if leak)
- ❌ Don't auto-relabel without manual review
- ❌ Don't commit actual data files to GitHub
- ❌ Don't skip cross-validation with Saeed

---

## 📝 Questions to Clarify

### Ask Guy/Team:

1. **Tagging Schema**: Confirm Drive location for official user tags
2. **Multi-Label Threshold**: How many different labels = "remove cluster"?
3. **Primary Label**: What % prominence = keep vs remove?
4. **Validation Leaks**: Remove from val only, or entire cluster?

### Ask Saeed:

1. **COCO Progress**: How far along? Need help?
2. **Tag Alignment**: Using same schema?
3. **Review Schedule**: When to cross-check tags?

---

## 🎯 SUCCESS CRITERIA

By next meeting, you should have:

- ✅ Quantified train-test leaks in ImageNet
- ✅ Tagged all VL findings with user tags
- ✅ Created removal and relabel lists
- ✅ Documented decision rationale
- ✅ Uploaded to Google Drive
- ✅ Ready to present findings

---

## 💻 Quick Reference Commands

```bash
# Clone your repo (if not already)
git clone https://github.com/KushagraKanaujia/vl-imagenet-cleaning.git
cd vl-imagenet-cleaning

# Analyze VL duplicates
python scripts/analyze_vl_duplicates.py \
    --vl_json vl_exports/imagenet1k_duplicates.json \
    --output manifests/imagenet1k_leaks.csv

# Tag VL findings
python scripts/vl_tagging_workflow.py \
    --vl_export vl_exports/imagenet1k_mislabels.csv \
    --dataset imagenet1k \
    --output manifests/imagenet1k_tagged.csv \
    --generate_html

# Commit progress
git add manifests/*.csv docs/*.md
git commit -m "ImageNet cleaning progress: analyzed leaks and tagged mislabels"
git push origin main
```

---

## 📞 If You Get Stuck

### Technical Issues:
- Check `DATASET_CLEANING_GUIDE.md` for detailed workflows
- Review `scripts/*.py` code comments
- Ask in team chat

### Conceptual Questions:
- Refer to meeting notes above
- Check Drive for tagging schema
- Discuss with Saeed for alignment

### VL Platform Issues:
- Try VL Chat for export queries
- Check VL documentation
- Message Guy for platform help

---

**YOU'VE GOT THIS! START WITH EXPORTING THE DUPLICATES JSON! 🚀**
