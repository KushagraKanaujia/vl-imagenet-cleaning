"""
Analyze Visual Layer Duplicate Export for Train-Test Leaks
Processes VL's duplicate clusters JSON to find train-test contamination

Usage:
    python analyze_vl_duplicates.py --vl_json path/to/vl_duplicates.json --output leaks_report.csv
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict


class VLDuplicateAnalyzer:
    def __init__(self, vl_json_path):
        """
        Initialize analyzer with VL duplicate export

        Args:
            vl_json_path: Path to Visual Layer duplicate export JSON
        """
        self.vl_json_path = Path(vl_json_path)
        self.duplicates = None
        self.train_test_leaks = []
        self.multi_label_clusters = []
        self.stats = defaultdict(int)

    def load_vl_export(self):
        """Load Visual Layer duplicate export"""
        print(f"Loading VL duplicate export: {self.vl_json_path}")

        with open(self.vl_json_path, 'r') as f:
            self.duplicates = json.load(f)

        print(f"✓ Loaded {len(self.duplicates)} duplicate clusters")
        return self.duplicates

    def is_validation_image(self, image_path):
        """
        Check if image is from validation set
        Guy's note: "Validation images have 'val' in filename"
        """
        path_str = str(image_path).lower()
        return 'val' in path_str or 'validation' in path_str

    def analyze_cluster_for_leaks(self, cluster):
        """
        Analyze a single duplicate cluster for train-test leaks

        Args:
            cluster: Dict with cluster info from VL
                Expected format:
                {
                    'cluster_id': str,
                    'images': [
                        {'path': str, 'label': int/str, 'id': str},
                        ...
                    ]
                }

        Returns:
            dict: Leak information if found, None otherwise
        """
        # Adapt this based on actual VL JSON structure
        # This is a template - adjust field names as needed
        images = cluster.get('images', cluster.get('members', []))

        has_train = False
        has_val = False
        labels = set()
        train_images = []
        val_images = []

        for img in images:
            # Adapt field names based on actual VL export
            img_path = img.get('path', img.get('image_path', img.get('filename', '')))
            img_label = img.get('label', img.get('class', img.get('category', '')))
            img_id = img.get('id', img.get('image_id', ''))

            labels.add(img_label)

            if self.is_validation_image(img_path):
                has_val = True
                val_images.append({
                    'path': img_path,
                    'label': img_label,
                    'id': img_id
                })
            else:
                has_train = True
                train_images.append({
                    'path': img_path,
                    'label': img_label,
                    'id': img_id
                })

        # Check for train-test leak
        if has_train and has_val:
            leak_type = 'same_label' if len(labels) == 1 else 'different_labels'

            return {
                'cluster_id': cluster.get('cluster_id', cluster.get('id', 'unknown')),
                'leak_type': leak_type,
                'num_labels': len(labels),
                'labels': list(labels),
                'num_train': len(train_images),
                'num_val': len(val_images),
                'train_images': train_images,
                'val_images': val_images,
                'action': 'REMOVE_VAL' if leak_type == 'same_label' else 'REVIEW_MANUAL'
            }

        # Check for multi-label cluster (ambiguous/mislabel)
        if len(labels) > 1:
            return {
                'cluster_id': cluster.get('cluster_id', cluster.get('id', 'unknown')),
                'leak_type': 'multi_label_no_leak',
                'num_labels': len(labels),
                'labels': list(labels),
                'num_train': len(train_images),
                'num_val': len(val_images),
                'train_images': train_images,
                'val_images': val_images,
                'action': 'REVIEW_LABELS'
            }

        return None

    def analyze_all_clusters(self):
        """Analyze all duplicate clusters"""
        print("\n" + "="*80)
        print("ANALYZING DUPLICATE CLUSTERS")
        print("="*80)

        if self.duplicates is None:
            raise ValueError("Must call load_vl_export() first")

        # Handle different possible VL export formats
        clusters = self.duplicates
        if isinstance(self.duplicates, dict):
            clusters = self.duplicates.get('clusters', self.duplicates.get('duplicates', []))

        print(f"\nProcessing {len(clusters)} clusters...")

        for cluster in clusters:
            result = self.analyze_cluster_for_leaks(cluster)

            if result:
                if 'leak' in result['leak_type'].lower() or result['leak_type'] in ['same_label', 'different_labels']:
                    self.train_test_leaks.append(result)
                    self.stats['train_test_leaks'] += 1

                    if result['leak_type'] == 'same_label':
                        self.stats['same_label_leaks'] += 1
                    else:
                        self.stats['different_label_leaks'] += 1

                if result['num_labels'] > 1:
                    self.multi_label_clusters.append(result)
                    self.stats['multi_label_clusters'] += 1

        print(f"\n✓ Analysis complete!")
        print(f"   Train-test leaks found: {self.stats['train_test_leaks']}")
        print(f"   Multi-label clusters: {self.stats['multi_label_clusters']}")

    def generate_leak_report(self, output_path):
        """Generate detailed train-test leak report"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Flatten leaks into rows
        leak_rows = []
        for leak in self.train_test_leaks:
            # Create row for each val image in leak
            for val_img in leak['val_images']:
                # Find corresponding train images
                for train_img in leak['train_images']:
                    leak_rows.append({
                        'cluster_id': leak['cluster_id'],
                        'leak_type': leak['leak_type'],
                        'train_image_path': train_img['path'],
                        'train_image_id': train_img['id'],
                        'train_label': train_img['label'],
                        'val_image_path': val_img['path'],
                        'val_image_id': val_img['id'],
                        'val_label': val_img['label'],
                        'labels_in_cluster': ','.join(map(str, leak['labels'])),
                        'recommended_action': leak['action'],
                        'priority': 'HIGH' if leak['leak_type'] == 'different_labels' else 'MEDIUM'
                    })

        df_leaks = pd.DataFrame(leak_rows)

        if len(df_leaks) > 0:
            df_leaks.to_csv(output_path, index=False)
            print(f"\n✓ Leak report saved: {output_path}")
        else:
            print(f"\n✓ No leaks detected!")
            # Save empty report
            pd.DataFrame({'status': ['No train-test leaks detected']}).to_csv(output_path, index=False)

        return df_leaks

    def generate_removal_list(self, output_path):
        """
        Generate list of validation images to remove
        (since they appear in training set)
        """
        output_path = Path(output_path)

        val_images_to_remove = set()

        for leak in self.train_test_leaks:
            for val_img in leak['val_images']:
                val_images_to_remove.add(val_img['id'])

        with open(output_path, 'w') as f:
            for img_id in sorted(val_images_to_remove):
                f.write(f"{img_id}\n")

        print(f"✓ Removal list saved: {output_path}")
        print(f"   Images to remove from validation: {len(val_images_to_remove)}")

        return val_images_to_remove

    def generate_multi_label_report(self, output_path):
        """Generate report of multi-label clusters (potential mislabels)"""
        output_path = Path(output_path)

        multi_label_rows = []
        for cluster in self.multi_label_clusters:
            all_images = cluster['train_images'] + cluster['val_images']

            for img in all_images:
                multi_label_rows.append({
                    'cluster_id': cluster['cluster_id'],
                    'image_path': img['path'],
                    'image_id': img['id'],
                    'current_label': img['label'],
                    'all_labels_in_cluster': ','.join(map(str, cluster['labels'])),
                    'num_different_labels': cluster['num_labels'],
                    'split': 'val' if self.is_validation_image(img['path']) else 'train',
                    'recommended_action': 'MANUAL_REVIEW',
                    'user_tags': 'ambiguous_class,needs_review',
                })

        df_multi = pd.DataFrame(multi_label_rows)

        if len(df_multi) > 0:
            df_multi.to_csv(output_path, index=False)
            print(f"✓ Multi-label report saved: {output_path}")
        else:
            print(f"✓ No multi-label clusters found")

        return df_multi

    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("TRAIN-TEST LEAK ANALYSIS SUMMARY")
        print("="*80)

        print(f"\nTotal duplicate clusters analyzed: {len(self.duplicates) if isinstance(self.duplicates, list) else len(self.duplicates.get('clusters', []))}")

        print(f"\n🚨 TRAIN-TEST LEAKS:")
        print(f"   Total leaks: {self.stats['train_test_leaks']}")
        print(f"   Same label leaks: {self.stats['same_label_leaks']}")
        print(f"   Different label leaks: {self.stats['different_label_leaks']}")

        print(f"\n⚠️  MULTI-LABEL CLUSTERS:")
        print(f"   Total: {self.stats['multi_label_clusters']}")

        if self.stats['train_test_leaks'] > 0:
            print(f"\n📋 RECOMMENDED ACTIONS:")
            print(f"   1. Remove {sum(len(leak['val_images']) for leak in self.train_test_leaks)} validation images")
            print(f"   2. Manually review {self.stats['different_label_leaks']} different-label leaks")
            print(f"   3. Review {self.stats['multi_label_clusters']} multi-label clusters for mislabels")

        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Analyze VL Duplicate Export for Train-Test Leaks')
    parser.add_argument('--vl_json', type=str, required=True,
                        help='Path to Visual Layer duplicate export JSON')
    parser.add_argument('--output', type=str, default='manifests/imagenet1k_leaks_from_vl.csv',
                        help='Output path for leak report')
    parser.add_argument('--removal_list', type=str, default='manifests/imagenet1k_val_removal_list.txt',
                        help='Output path for validation removal list')
    parser.add_argument('--multi_label_report', type=str, default='manifests/imagenet1k_multi_label_clusters.csv',
                        help='Output path for multi-label cluster report')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = VLDuplicateAnalyzer(args.vl_json)

    # Load VL export
    analyzer.load_vl_export()

    # Analyze for leaks
    analyzer.analyze_all_clusters()

    # Generate reports
    analyzer.generate_leak_report(args.output)
    analyzer.generate_removal_list(args.removal_list)
    analyzer.generate_multi_label_report(args.multi_label_report)

    # Print summary
    analyzer.print_summary()

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review leak report CSV")
    print("2. Use removal list to filter validation set")
    print("3. Manually review multi-label clusters")
    print("4. Tag images with user tags (see Drive schema)")
    print("5. Update training pipeline with cleaned data")
    print("="*80)


if __name__ == '__main__':
    main()
