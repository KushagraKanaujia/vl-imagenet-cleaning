# Manifest File Specifications

This directory contains manifest files for tracking dataset issues and cleaning operations.

## File Format Standards

### Visual Layer Export Format
Expected columns from VL platform:
```
image_id, image_path, label, issue_type, confidence, suggested_label, reason, split
```

### Tagged Manifest Format
After user tagging workflow:
```
image_id, image_path, dataset, split, original_label, vl_flag_type, vl_confidence,
vl_suggested_label, vl_reason, user_tags, user_action, new_label, reviewer,
review_date, notes
```

### Leak Detection Report Format
From train-test leak detection:
```
train_image, test_image, method, similarity, leak_type, hamming_distance
```

## User Tag Taxonomy

Standard tags to use:
- `mislabel_confirmed` - Definitely wrong label
- `mislabel_uncertain` - Possibly wrong, needs discussion
- `outlier_valid` - Edge case but correct
- `outlier_invalid` - Outlier AND wrong label
- `duplicate_exact` - Exact duplicate
- `duplicate_near` - Near duplicate (>95% similar)
- `low_quality_blur` - Blurry/low resolution
- `low_quality_corrupt` - Corrupted file
- `ambiguous_class` - Could belong to multiple classes
- `train_test_leak` - Appears in both train and test
- `keep` - Flagged by VL but should keep
- `remove` - Should remove from dataset
- `relabel` - Should relabel to different class

## User Actions

Three possible actions:
1. `keep` - Keep image in dataset with current label
2. `remove` - Remove image from dataset
3. `relabel` - Change image label (must provide new_label)

## Example Files

### imagenet1k_vl_export.csv
```csv
image_id,image_path,label,issue_type,confidence,suggested_label,reason,split
12345,train/n01440764/n01440764_10026.JPEG,0,mislabel,0.95,156,"High confidence mismatch",train
67890,train/n01443537/n01443537_10007.JPEG,1,outlier,0.87,,"Unusual features",train
```

### imagenet1k_tagged_manifest.csv
```csv
image_id,image_path,dataset,split,original_label,vl_flag_type,vl_confidence,vl_suggested_label,vl_reason,user_tags,user_action,new_label,reviewer,review_date,notes
12345,train/n01440764/n01440764_10026.JPEG,imagenet1k,train,0,mislabel,0.95,156,"High confidence mismatch",mislabel_confirmed,relabel,156,Kushagra,2025-11-23T10:30:00,"Verified - clearly a different class"
67890,train/n01443537/n01443537_10007.JPEG,imagenet1k,train,1,outlier,0.87,,"Unusual features",outlier_valid,keep,,Kushagra,2025-11-23T10:32:00,"Edge case but correct label"
```

### imagenet1k_leaks_report.csv
```csv
train_image,test_image,method,similarity,leak_type,hamming_distance
train/n01440764/img_001.JPEG,val/n01440764/img_999.JPEG,phash,0.98,near_duplicate,2
train/n01443537/img_042.JPEG,val/n01443537/img_888.JPEG,md5_exact,1.0,exact_duplicate,
```

## Naming Conventions

- `{dataset}_vl_export.csv` - Raw Visual Layer export
- `{dataset}_tagged_manifest.csv` - After user tagging
- `{dataset}_tagged_manifest_v{N}.csv` - Versioned iterations
- `{dataset}_leaks_report.csv` - Leak detection results
- `{dataset}_final_cleaning_manifest.csv` - Combined final version
- `{dataset}_removal_list.txt` - Simple list of image IDs to remove
- `{dataset}_relabel_map.json` - Dictionary of {image_id: new_label}

## Storage

- **Local**: Store in this manifests/ directory during work
- **Google Drive**: Upload final versions to `Dataset_Cleaning_Results/Manifests/`
- **Version Control**: Track manifest schemas and examples in git, NOT actual data

## Best Practices

1. **Version Everything**: Use v1, v2, v3 suffixes as you iterate
2. **Document Changes**: Add notes column for important decisions
3. **Cross-Validate**: Have team members review each other's tags
4. **Backup Frequently**: Save to Google Drive after each work session
5. **Track Statistics**: Keep running counts of issues by type
