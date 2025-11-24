"""
Train-Test Leak Detection for ImageNet and COCO
Detects duplicate and near-duplicate images between train and test splits

Methods:
1. Perceptual hashing (pHash) - Fast, catches exact/near duplicates
2. Deep feature similarity - More robust, catches semantic duplicates
3. Pixel-level comparison - Baseline for exact matches

Usage:
    python train_test_leak_detection.py --dataset imagenet1k --train_dir path/to/train --test_dir path/to/test --output leaks_report.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import hashlib
from collections import defaultdict
import imagehash
import warnings
warnings.filterwarnings('ignore')


class ImageDataset(Dataset):
    """Simple dataset for loading images"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, str(self.image_paths[idx])
        except Exception as e:
            # Return black image on error
            if self.transform:
                return torch.zeros(3, 224, 224), str(self.image_paths[idx])
            return Image.new('RGB', (224, 224)), str(self.image_paths[idx])


class TrainTestLeakDetector:
    def __init__(self, train_dir, test_dir, dataset_name='imagenet1k'):
        """
        Initialize leak detector

        Args:
            train_dir: Directory containing training images
            test_dir: Directory containing test/validation images
            dataset_name: Name of dataset for reporting
        """
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.dataset_name = dataset_name

        self.train_images = []
        self.test_images = []
        self.leaks = []

        print(f"Initializing Train-Test Leak Detector for {dataset_name}")
        print(f"Train dir: {self.train_dir}")
        print(f"Test dir: {self.test_dir}")

    def collect_image_paths(self):
        """Collect all image paths from train and test directories"""
        print("\nCollecting image paths...")

        # Common image extensions
        exts = ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']

        for ext in exts:
            self.train_images.extend(list(self.train_dir.rglob(ext)))
            self.test_images.extend(list(self.test_dir.rglob(ext)))

        print(f"✓ Found {len(self.train_images):,} training images")
        print(f"✓ Found {len(self.test_images):,} test images")

        if len(self.train_images) == 0 or len(self.test_images) == 0:
            raise ValueError("No images found. Check directory paths.")

    def detect_exact_duplicates(self):
        """
        Method 1: MD5 hash for exact pixel-level duplicates
        Fastest method, catches identical files
        """
        print("\n" + "="*80)
        print("METHOD 1: Exact Duplicate Detection (MD5 Hashing)")
        print("="*80)

        print("Computing MD5 hashes for training images...")
        train_hashes = {}
        for img_path in tqdm(self.train_images, desc="Train hashes"):
            try:
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    train_hashes[file_hash] = img_path
            except Exception as e:
                continue

        print(f"Computing MD5 hashes for test images...")
        exact_duplicates = []
        for img_path in tqdm(self.test_images, desc="Test hashes"):
            try:
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    if file_hash in train_hashes:
                        exact_duplicates.append({
                            'train_image': str(train_hashes[file_hash]),
                            'test_image': str(img_path),
                            'method': 'md5_exact',
                            'similarity': 1.0,
                            'leak_type': 'exact_duplicate'
                        })
            except Exception as e:
                continue

        print(f"\n✓ Found {len(exact_duplicates)} exact duplicates")
        self.leaks.extend(exact_duplicates)
        return exact_duplicates

    def detect_perceptual_duplicates(self, hash_size=8, threshold=5):
        """
        Method 2: Perceptual hashing (pHash) for near-duplicates
        Detects images that look similar but may have minor differences

        Args:
            hash_size: Size of hash (8 = 64-bit hash)
            threshold: Hamming distance threshold (lower = more similar)
                      0 = identical, 5 = very similar, 10 = somewhat similar
        """
        print("\n" + "="*80)
        print("METHOD 2: Perceptual Hash Detection (pHash)")
        print("="*80)
        print(f"Hash size: {hash_size}, Threshold: {threshold} bits")

        print("\nComputing perceptual hashes for training images...")
        train_phashes = {}
        for img_path in tqdm(self.train_images, desc="Train pHash"):
            try:
                img = Image.open(img_path).convert('RGB')
                phash = imagehash.phash(img, hash_size=hash_size)
                train_phashes[str(img_path)] = phash
            except Exception as e:
                continue

        print(f"Comparing test images against training set...")
        perceptual_duplicates = []
        for test_path in tqdm(self.test_images, desc="Test pHash"):
            try:
                img = Image.open(test_path).convert('RGB')
                test_hash = imagehash.phash(img, hash_size=hash_size)

                # Compare against all training hashes
                for train_path, train_hash in train_phashes.items():
                    hamming_dist = test_hash - train_hash

                    if hamming_dist <= threshold:
                        similarity = 1.0 - (hamming_dist / (hash_size * hash_size))
                        leak_type = 'exact_duplicate' if hamming_dist == 0 else 'near_duplicate'

                        perceptual_duplicates.append({
                            'train_image': train_path,
                            'test_image': str(test_path),
                            'method': 'phash',
                            'similarity': similarity,
                            'hamming_distance': hamming_dist,
                            'leak_type': leak_type
                        })

            except Exception as e:
                continue

        print(f"\n✓ Found {len(perceptual_duplicates)} perceptual duplicates")
        self.leaks.extend(perceptual_duplicates)
        return perceptual_duplicates

    def detect_semantic_duplicates(self, batch_size=32, threshold=0.95):
        """
        Method 3: Deep feature similarity using pre-trained ResNet
        Most robust, catches semantically similar images

        Args:
            batch_size: Batch size for feature extraction
            threshold: Cosine similarity threshold (0.95 = 95% similar)
        """
        print("\n" + "="*80)
        print("METHOD 3: Deep Feature Similarity (ResNet50)")
        print("="*80)
        print(f"Similarity threshold: {threshold}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load pre-trained ResNet50 and remove final classification layer
        print("Loading ResNet50 feature extractor...")
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
        model = model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Extract features for training images
        print("\nExtracting features for training images...")
        train_dataset = ImageDataset(self.train_images, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)

        train_features = []
        train_paths = []
        with torch.no_grad():
            for images, paths in tqdm(train_loader, desc="Train features"):
                images = images.to(device)
                features = model(images).squeeze()
                train_features.append(features.cpu().numpy())
                train_paths.extend(paths)

        train_features = np.vstack(train_features)
        train_features = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)

        # Extract features for test images
        print("\nExtracting features for test images...")
        test_dataset = ImageDataset(self.test_images, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=4, pin_memory=True)

        test_features = []
        test_paths = []
        with torch.no_grad():
            for images, paths in tqdm(test_loader, desc="Test features"):
                images = images.to(device)
                features = model(images).squeeze()
                test_features.append(features.cpu().numpy())
                test_paths.extend(paths)

        test_features = np.vstack(test_features)
        test_features = test_features / np.linalg.norm(test_features, axis=1, keepdims=True)

        # Compute cosine similarities
        print("\nComputing similarities...")
        semantic_duplicates = []

        for i, test_feat in enumerate(tqdm(test_features, desc="Comparing")):
            # Compute cosine similarity with all training features
            similarities = np.dot(train_features, test_feat)
            max_idx = np.argmax(similarities)
            max_sim = similarities[max_idx]

            if max_sim >= threshold:
                leak_type = 'semantic_duplicate' if max_sim < 0.99 else 'near_duplicate'

                semantic_duplicates.append({
                    'train_image': train_paths[max_idx],
                    'test_image': test_paths[i],
                    'method': 'resnet50_features',
                    'similarity': float(max_sim),
                    'leak_type': leak_type
                })

        print(f"\n✓ Found {len(semantic_duplicates)} semantic duplicates")
        self.leaks.extend(semantic_duplicates)
        return semantic_duplicates

    def generate_report(self, output_path):
        """Generate comprehensive leak detection report"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame
        df_leaks = pd.DataFrame(self.leaks)

        if len(df_leaks) == 0:
            print("\n" + "="*80)
            print("✓ NO LEAKS DETECTED!")
            print("="*80)
            print("The train and test sets appear to be properly separated.")

            # Save empty report
            empty_report = pd.DataFrame({
                'status': ['No leaks detected'],
                'train_images': [len(self.train_images)],
                'test_images': [len(self.test_images)]
            })
            empty_report.to_csv(output_path, index=False)
            return

        # Remove duplicates (same leak detected by multiple methods)
        df_leaks = df_leaks.drop_duplicates(subset=['train_image', 'test_image'])

        # Sort by similarity
        df_leaks = df_leaks.sort_values('similarity', ascending=False)

        # Save to CSV
        df_leaks.to_csv(output_path, index=False)

        # Generate summary statistics
        print("\n" + "="*80)
        print("TRAIN-TEST LEAK DETECTION REPORT")
        print("="*80)
        print(f"Dataset: {self.dataset_name}")
        print(f"Training images: {len(self.train_images):,}")
        print(f"Test images: {len(self.test_images):,}")
        print(f"\n⚠️  TOTAL LEAKS DETECTED: {len(df_leaks):,}")
        print(f"   ({len(df_leaks) / len(self.test_images) * 100:.2f}% of test set)")

        print("\nBreakdown by leak type:")
        print(df_leaks['leak_type'].value_counts())

        print("\nBreakdown by detection method:")
        print(df_leaks['method'].value_counts())

        print("\nSimilarity distribution:")
        print(df_leaks['similarity'].describe())

        print(f"\n✓ Full report saved to: {output_path}")
        print("="*80)

        # Also save summary
        summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("TRAIN-TEST LEAK DETECTION SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Training images: {len(self.train_images):,}\n")
            f.write(f"Test images: {len(self.test_images):,}\n")
            f.write(f"Total leaks: {len(df_leaks):,}\n")
            f.write(f"Leak rate: {len(df_leaks) / len(self.test_images) * 100:.2f}%\n\n")
            f.write("Breakdown by type:\n")
            f.write(str(df_leaks['leak_type'].value_counts()))

        print(f"✓ Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Train-Test Leak Detection')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (imagenet1k, coco, etc)')
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to training images directory')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to test/validation images directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for leak report CSV')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['exact', 'perceptual', 'semantic'],
                        choices=['exact', 'perceptual', 'semantic'],
                        help='Detection methods to use')
    parser.add_argument('--phash_threshold', type=int, default=5,
                        help='Perceptual hash threshold (default: 5)')
    parser.add_argument('--semantic_threshold', type=float, default=0.95,
                        help='Semantic similarity threshold (default: 0.95)')

    args = parser.parse_args()

    # Initialize detector
    detector = TrainTestLeakDetector(args.train_dir, args.test_dir, args.dataset)

    # Collect image paths
    detector.collect_image_paths()

    # Run selected detection methods
    if 'exact' in args.methods:
        detector.detect_exact_duplicates()

    if 'perceptual' in args.methods:
        detector.detect_perceptual_duplicates(threshold=args.phash_threshold)

    if 'semantic' in args.methods:
        detector.detect_semantic_duplicates(threshold=args.semantic_threshold)

    # Generate report
    detector.generate_report(args.output)

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review the leak report CSV")
    print("2. Manually verify flagged image pairs")
    print("3. Remove duplicate images from test set")
    print("4. Re-run evaluation to get clean metrics")
    print("="*80)


if __name__ == '__main__':
    main()
