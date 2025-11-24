# ImageNet Training Reference Notes
**From Previous Experiments**

## Completed Training Runs

### Configuration Used
- **Model**: ResNet-18
- **Dataset**: ImageNet1K (evanarlian/imagenet_1k_resized_256)
- **Epochs**: 30-50 depending on experiment
- **Batch Size**: 256 (adjusted based on GPU memory)
- **Optimizer**: SGD (lr=0.1, momentum=0.9, weight_decay=1e-4)
- **Scheduler**: Cosine annealing + 5 epoch warmup
- **Mixed Precision**: Yes (AMP enabled)

### Experiments Completed

#### Experiment 1: Clean Baseline
- **Data**: 0% noise (original labels)
- **Purpose**: Baseline performance
- **Status**: ✅ Completed
- **Notebook**: `Final_FULL_Dataset_Training.ipynb`

#### Experiment 2: 20% Random Noise
- **Data**: 20% labels randomly flipped
- **Purpose**: Measure noise impact
- **Status**: ✅ Completed
- **Finding**: ~X% accuracy drop from baseline

#### Experiment 3: 20% Noise + 80% VL Cleaning
- **Data**: 20% noise, 80% detected and cleaned (simulated)
- **Purpose**: Measure VL cleaning effectiveness
- **Status**: ✅ Completed
- **Finding**: Recovered ~Y% of lost accuracy

#### Experiment 4: 40% Random Noise
- **Data**: 40% labels randomly flipped (extreme)
- **Purpose**: High noise scenario
- **Status**: ✅ Partially completed
- **Finding**: Exponentially worse than 20%

#### Experiment 5: 40% Noise + 60% VL Cleaning
- **Data**: 40% noise, 60% cleaned (simulated)
- **Purpose**: Cleaning under high noise
- **Status**: ✅ Completed

## Key Notebooks Locations

### On Local Machine
```
/Users/kush/Downloads/
├── Final_FULL_Dataset_Training.ipynb
├── Analyze_And_Train_ImageNet.ipynb
├── Plot_Experiments_WandB.ipynb
├── ImageNet100_ResNet18_Complete.ipynb
├── ResNet_ImageNet1K_Training.ipynb
└── Final_50Epoch_Training_WandB.ipynb
```

### Training Scripts
```
/Users/kush/capstone_project_Visual-Layer/capstone_project_visual_layer/imagenet100/
├── train_resnet18.py
├── visualize_loss_curves.py
└── ResNet18_ImageNet100_Training.ipynb
```

## Infrastructure Features

### ✅ Implemented
- Auto-resume from checkpoints (Drive storage)
- Weights & Biases integration
  - Loss/accuracy tracking
  - Weight/gradient statistics every epoch
  - Hyperparameter logging
- Progress format: `[epoch][step/100]`
- Top-1 and Top-5 accuracy
- Learning rate scheduling with warmup
- Mixed precision training
- Comprehensive plotting

### Key Code Patterns

#### Dataset with Corrupted Labels
```python
class ImageNetSubset(Dataset):
    def __init__(self, hf_dataset, indices, labels, transform):
        self.dataset = hf_dataset
        self.indices = indices
        self.labels = labels  # Can be corrupted
        self.transform = transform

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = self.dataset[int(real_idx)]
        image = sample['image'].convert('RGB')
        label = self.labels[idx]  # Use corrupted label
        return {'image': self.transform(image), 'label': label}
```

#### Noise Injection
```python
def inject_label_noise(dataset_indices, dataset, noise_rate=0.20):
    n = len(dataset_indices)
    n_noisy = int(n * noise_rate)

    original_labels = np.array([dataset[int(i)]['label'] for i in dataset_indices])
    corrupted_labels = original_labels.copy()

    noise_indices = np.random.choice(n, n_noisy, replace=False)
    noise_mask = np.zeros(n, dtype=bool)
    noise_mask[noise_indices] = True

    for idx in noise_indices:
        original = original_labels[idx]
        wrong_labels = [l for l in range(1000) if l != original]
        corrupted_labels[idx] = np.random.choice(wrong_labels)

    return corrupted_labels, noise_mask, original_labels
```

#### Simulated VL Cleaning
```python
def simulate_visual_layer_cleaning(corrupted_labels, original_labels,
                                    noise_mask, detection_rate=0.80):
    cleaned_labels = corrupted_labels.copy()
    noisy_indices = np.where(noise_mask)[0]
    n_detected = int(len(noisy_indices) * detection_rate)
    detected_indices = np.random.choice(noisy_indices, n_detected, replace=False)

    for idx in detected_indices:
        cleaned_labels[idx] = original_labels[idx]  # Fix label

    return cleaned_labels
```

#### Progress Tracking
```python
def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    total_batches = len(train_loader)
    last_logged_progress = 0

    for batch_idx, batch in enumerate(train_loader):
        # ... training code ...

        # Log at 10%, 20%, ..., 100%
        progress = int(((batch_idx + 1) / total_batches) * 100)
        progress = (progress // 10) * 10

        if progress > 0 and progress % 10 == 0 and progress != last_logged_progress:
            print(f"Epoch [{epoch}][{progress}/100] Loss: {loss:.4f} Acc: {acc:.2f}%")
            last_logged_progress = progress
```

#### W&B Logging
```python
if config.use_wandb:
    log_dict = {
        'epoch': epoch + 1,
        'train/loss': train_loss,
        'train/acc': train_acc,
        'val/loss': val_loss,
        'val/top1': val_top1,
        'val/top5': val_top5,
        'learning_rate': current_lr,
    }

    # Track weights and gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            log_dict[f'weights/{name}_mean'] = param.data.mean().item()
            log_dict[f'weights/{name}_std'] = param.data.std().item()
            if param.grad is not None:
                log_dict[f'gradients/{name}_mean'] = param.grad.mean().item()

    wandb.log(log_dict)
```

## Next Phase: Real VL Cleaning

### Key Differences from Simulation
1. **Real Issues**: VL will find actual mislabels, not random flips
2. **Multiple Issue Types**: Mislabels, outliers, duplicates, quality issues
3. **Confidence Scores**: VL provides confidence, not binary detection
4. **Manual Validation**: Need human review to confirm VL findings

### Integration Plan
```python
# Load cleaning artifacts
with open('manifests/imagenet1k_removal_list.txt') as f:
    removal_ids = set(int(line.strip()) for line in f)

with open('manifests/imagenet1k_relabel_map.json') as f:
    relabel_map = json.load(f)

# Modified dataset class
class CleanedImageNetDataset(Dataset):
    def __init__(self, hf_dataset, removal_ids, relabel_map, transform):
        self.dataset = hf_dataset
        self.relabel_map = relabel_map
        self.transform = transform

        # Filter out removed images
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

        # Use relabeled label if available
        label = self.relabel_map.get(str(real_idx), sample['label'])

        return self.transform(image), label
```

## Weights & Biases Projects

### Existing W&B Projects
- `imagenet-resnet18-15pct-thursday`
  - Exp 1: Clean (run: exp1_clean_15pct)
  - Exp 2: Noisy 20% (run: exp2_noisy20_15pct)
  - Exp 3: Cleaned 80% (run: buee49gy)
  - Exp 4: Noisy 40% (run: exp4_noisy40_15pct)

### Future Projects
- `imagenet-resnet18-vl-cleaned` (after real VL cleaning)
- `imagenet-resnet18-leak-removed` (after leak detection)

## Performance Benchmarks

### Expected Results (ImageNet1K, ResNet-18)
- **Clean baseline**: ~69-71% Top-1, ~89-91% Top-5
- **20% random noise**: ~X% Top-1 (drop of Y%)
- **After 80% cleaning**: ~Z% Top-1 (recovery of W%)

### Training Time Estimates
- **Per epoch**: 15-20 minutes (full ImageNet1K on GPU)
- **50 epochs**: 12-15 hours
- **Full 5 experiments**: 60-75 hours

## Tips for Next Training Runs

1. **Start with Subset**: Test on 1% of data first
2. **Monitor Early**: Check first 5 epochs for issues
3. **Use Kaggle/Modal**: Don't tie up local machine
4. **Version Experiments**: Clear naming in W&B
5. **Save Checkpoints**: Every 5-10 epochs
6. **Document Changes**: Note what's different in W&B description

## Resources

- **HuggingFace Dataset**: `evanarlian/imagenet_1k_resized_256`
- **W&B Account**: kush13
- **Compute**: Kaggle (30 hrs/week), Modal ($30 credit)
- **Google Drive**: Checkpoint storage, results sharing
