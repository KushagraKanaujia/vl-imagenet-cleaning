#!/usr/bin/env bash
# Run all three cleaning tools end-to-end on a tiny synthetic dataset.
# No ImageNet download, no Visual Layer account, no API keys.
#
#   bash demo/run_demo.sh            # exact + perceptual leak detection (fast, CPU)
#   bash demo/run_demo.sh --semantic # also run ResNet50 feature similarity
#                                    # (downloads ~100 MB of torchvision weights)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
METHODS="exact perceptual"
if [[ "${1:-}" == "--semantic" ]]; then METHODS="exact perceptual semantic"; fi

echo "==> 1/4 Generating synthetic demo dataset"
$PY demo/make_demo_data.py --out demo_data

echo
echo "==> 2/4 Visual Layer tagging workflow (mock VL export -> manifest + review page)"
$PY scripts/vl_tagging_workflow.py \
  --vl_export demo_data/vl_export.csv \
  --dataset imagenet1k \
  --output demo_output/imagenet1k_tagged_manifest.csv \
  --generate_html

echo
echo "==> 3/4 Train-test leak detection on the image folders ($METHODS)"
# shellcheck disable=SC2086
$PY scripts/train_test_leak_detection.py \
  --dataset imagenet1k-demo \
  --train_dir demo_data/train \
  --test_dir demo_data/val \
  --output demo_output/imagenet1k_leaks.csv \
  --methods $METHODS \
  --num_workers 0

echo
echo "==> 4/4 Visual Layer duplicate-cluster analysis (mock VL duplicates JSON)"
$PY scripts/analyze_vl_duplicates.py \
  --vl_json demo_data/vl_duplicates.json \
  --output demo_output/imagenet1k_leaks_from_vl.csv \
  --removal_list demo_output/imagenet1k_val_removal_list.txt \
  --multi_label_report demo_output/imagenet1k_multi_label_clusters.csv

echo
echo "Done. Outputs are in demo_output/:"
ls -1 demo_output
echo
echo "Planted leaks (ground truth) are listed in demo_data/planted_leaks.json."
echo "Open demo_output/review_interface.html in a browser to see the review page."
