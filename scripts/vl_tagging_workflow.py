"""
Visual Layer Tagging Workflow for ImageNet and COCO
Processes VL outputs and adds user tags for dataset cleaning

Usage:
    python vl_tagging_workflow.py --vl_export path/to/vl_export.csv --dataset imagenet1k --output tagged_manifest.csv
"""

import pandas as pd
import argparse
from pathlib import Path
import json
from datetime import datetime

# User tag taxonomy for dataset issues
USER_TAGS = {
    'mislabel_confirmed': 'Manual verification confirms wrong label',
    'mislabel_uncertain': 'Possible mislabel but unclear',
    'outlier_valid': 'Outlier but correctly labeled (edge case)',
    'outlier_invalid': 'Outlier and incorrectly labeled',
    'duplicate_exact': 'Exact duplicate in train/test',
    'duplicate_near': 'Near duplicate (>95% similar)',
    'low_quality_blur': 'Blurry or low resolution',
    'low_quality_corrupt': 'Corrupted or truncated image',
    'ambiguous_class': 'Could belong to multiple classes',
    'train_test_leak': 'Image appears in both train and test',
    'keep': 'Flagged by VL but should keep',
    'remove': 'Should remove from dataset',
    'relabel': 'Should relabel to different class',
}

class VLTaggingWorkflow:
    def __init__(self, vl_export_path, dataset_name):
        """
        Initialize the tagging workflow

        Args:
            vl_export_path: Path to Visual Layer export CSV
            dataset_name: Name of dataset (imagenet1k, coco, etc)
        """
        self.vl_export_path = Path(vl_export_path)
        self.dataset_name = dataset_name
        self.vl_data = None
        self.tagged_data = None

    def load_vl_export(self):
        """Load Visual Layer export file"""
        print(f"Loading VL export from: {self.vl_export_path}")

        # Try different formats VL might export
        if self.vl_export_path.suffix == '.csv':
            self.vl_data = pd.read_csv(self.vl_export_path)
        elif self.vl_export_path.suffix == '.json':
            self.vl_data = pd.read_json(self.vl_export_path)
        else:
            raise ValueError(f"Unsupported file format: {self.vl_export_path.suffix}")

        print(f"✓ Loaded {len(self.vl_data)} flagged samples")
        print(f"Columns: {list(self.vl_data.columns)}")
        return self.vl_data

    def create_tagging_manifest(self):
        """
        Create initial manifest for user tagging
        Converts VL output to standardized format
        """
        print("\nCreating tagging manifest...")

        # Standardize column names (adapt based on actual VL export format)
        manifest = pd.DataFrame({
            'image_id': self.vl_data.get('image_id', self.vl_data.get('id', range(len(self.vl_data)))),
            'image_path': self.vl_data.get('image_path', self.vl_data.get('path', '')),
            'dataset': self.dataset_name,
            'split': self.vl_data.get('split', 'train'),  # train/val/test
            'original_label': self.vl_data.get('label', self.vl_data.get('class', '')),
            'vl_flag_type': self.vl_data.get('issue_type', self.vl_data.get('flag', '')),
            'vl_confidence': self.vl_data.get('confidence', 0.0),
            'vl_suggested_label': self.vl_data.get('suggested_label', ''),
            'vl_reason': self.vl_data.get('reason', ''),
            'user_tags': '',  # To be filled by user
            'user_action': '',  # keep/remove/relabel
            'new_label': '',  # If relabeling
            'reviewer': '',  # Who reviewed this
            'review_date': '',
            'notes': '',
        })

        self.tagged_data = manifest
        print(f"✓ Created manifest with {len(manifest)} entries")
        return manifest

    def add_user_tags_batch(self, image_ids, tags, action, reviewer_name):
        """
        Add user tags to multiple images at once

        Args:
            image_ids: List of image IDs to tag
            tags: List of tag strings (from USER_TAGS keys)
            action: 'keep', 'remove', or 'relabel'
            reviewer_name: Name of person doing review
        """
        if self.tagged_data is None:
            raise ValueError("Must call create_tagging_manifest() first")

        mask = self.tagged_data['image_id'].isin(image_ids)
        self.tagged_data.loc[mask, 'user_tags'] = ','.join(tags)
        self.tagged_data.loc[mask, 'user_action'] = action
        self.tagged_data.loc[mask, 'reviewer'] = reviewer_name
        self.tagged_data.loc[mask, 'review_date'] = datetime.now().isoformat()

        print(f"✓ Tagged {mask.sum()} images with: {tags}, action: {action}")

    def export_manifest(self, output_path):
        """Export tagged manifest"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.tagged_data.to_csv(output_path, index=False)
        print(f"\n✓ Exported tagged manifest to: {output_path}")

        # Print summary statistics
        print("\n" + "="*80)
        print("TAGGING SUMMARY")
        print("="*80)
        print(f"Total flagged samples: {len(self.tagged_data)}")
        print(f"Reviewed: {(self.tagged_data['user_action'] != '').sum()}")
        print(f"Pending review: {(self.tagged_data['user_action'] == '').sum()}")
        print("\nActions breakdown:")
        print(self.tagged_data['user_action'].value_counts())
        print("\nVL flag types:")
        print(self.tagged_data['vl_flag_type'].value_counts())
        print("="*80)

    def generate_review_html(self, output_path, n_per_page=50):
        """
        Generate HTML interface for reviewing flagged images
        Useful for manual review with team
        """
        output_path = Path(output_path)

        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Visual Layer Review - {dataset}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .image-card {{
                    border: 2px solid #ddd;
                    padding: 15px;
                    margin: 20px 0;
                    border-radius: 8px;
                }}
                .flagged {{ border-color: #ff6b6b; background-color: #fff5f5; }}
                .image-info {{ display: flex; gap: 20px; }}
                .image-preview {{ max-width: 300px; max-height: 300px; }}
                .details {{ flex: 1; }}
                .tag-button {{
                    padding: 5px 10px;
                    margin: 5px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }}
                .keep {{ background-color: #51cf66; color: white; }}
                .remove {{ background-color: #ff6b6b; color: white; }}
                .relabel {{ background-color: #ffd43b; }}
                h2 {{ color: #333; }}
                .vl-flag {{
                    background-color: #e7f5ff;
                    padding: 10px;
                    border-radius: 4px;
                    margin: 10px 0;
                }}
            </style>
        </head>
        <body>
            <h1>Visual Layer Dataset Review: {dataset}</h1>
            <p>Total flagged: {total_flagged}</p>
            <hr>
        """.format(
            dataset=self.dataset_name,
            total_flagged=len(self.tagged_data)
        )

        for idx, row in self.tagged_data.head(n_per_page).iterrows():
            html += f"""
            <div class="image-card flagged">
                <h2>Image ID: {row['image_id']}</h2>
                <div class="image-info">
                    <div>
                        <img src="{row['image_path']}" class="image-preview"
                             onerror="this.src='placeholder.jpg'">
                    </div>
                    <div class="details">
                        <div class="vl-flag">
                            <strong>VL Flag:</strong> {row['vl_flag_type']}<br>
                            <strong>Confidence:</strong> {row['vl_confidence']:.2f}<br>
                            <strong>Reason:</strong> {row['vl_reason']}<br>
                            <strong>Original Label:</strong> {row['original_label']}<br>
                            <strong>Suggested Label:</strong> {row['vl_suggested_label']}
                        </div>
                        <div>
                            <h3>Review Actions:</h3>
                            <button class="tag-button keep">✓ Keep</button>
                            <button class="tag-button remove">✗ Remove</button>
                            <button class="tag-button relabel">↻ Relabel</button>
                        </div>
                        <div>
                            <h3>User Tags:</h3>
                            <select multiple style="width: 100%; height: 100px;">
                                {''.join(f'<option value="{k}">{v}</option>' for k, v in USER_TAGS.items())}
                            </select>
                        </div>
                    </div>
                </div>
            </div>
            """

        html += """
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html)

        print(f"\n✓ Generated review interface: {output_path}")
        print(f"   Open in browser to review flagged images")


def main():
    parser = argparse.ArgumentParser(description='Visual Layer Tagging Workflow')
    parser.add_argument('--vl_export', type=str, required=True,
                        help='Path to Visual Layer export file')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['imagenet1k', 'imagenet100', 'coco'],
                        help='Dataset name')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for tagged manifest')
    parser.add_argument('--generate_html', action='store_true',
                        help='Generate HTML review interface')

    args = parser.parse_args()

    # Initialize workflow
    workflow = VLTaggingWorkflow(args.vl_export, args.dataset)

    # Load VL export
    workflow.load_vl_export()

    # Create tagging manifest
    workflow.create_tagging_manifest()

    # Generate HTML review interface if requested
    if args.generate_html:
        html_path = Path(args.output).parent / 'review_interface.html'
        workflow.generate_review_html(html_path)

    # Export initial manifest
    workflow.export_manifest(args.output)

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Open review_interface.html in browser (if generated)")
    print("2. Review flagged images and add tags")
    print("3. Use add_user_tags_batch() in Python to batch tag similar issues")
    print("4. Re-export manifest with updated tags")
    print("5. Use manifest to clean dataset for training")
    print("="*80)


if __name__ == '__main__':
    main()
