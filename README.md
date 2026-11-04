# VL ImageNet Cleaning

**Find mislabels and train/test leaks in image datasets, then turn them into a reviewable cleaning manifest.**

[![Live demo](https://img.shields.io/badge/live%20demo-netlify-00C7B7?logo=netlify&logoColor=white)](https://kush-vl-imagenet-cleaning.netlify.app)
[![Open in GitHub Codespaces](https://img.shields.io/badge/open%20in-Codespaces-181717?logo=github)](https://codespaces.new/KushagraKanaujia/vl-imagenet-cleaning?quickstart=1)
[![demo](https://github.com/KushagraKanaujia/vl-imagenet-cleaning/actions/workflows/demo.yml/badge.svg)](https://github.com/KushagraKanaujia/vl-imagenet-cleaning/actions/workflows/demo.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Part of a UCSB CS189A capstone project on label noise and dataset cleaning with the [Visual Layer](https://www.visual-layer.com/) platform.

## Try it in one click

| | What you get |
|---|---|
| **[Live demo](https://kush-vl-imagenet-cleaning.netlify.app)** | A static page with the real output of all three scripts, run on a tiny synthetic dataset with leaks planted on purpose. Nothing to install. |
| **[Open in Codespaces](https://codespaces.new/KushagraKanaujia/vl-imagenet-cleaning?quickstart=1)** | A ready Python 3.12 environment (CPU PyTorch, imagehash, pandas). The demo runs automatically when the Codespace is created; results land in `demo_output/`. |

The demo needs **no ImageNet download, no Visual Layer account and no API keys**. It generates 37 small synthetic images in three ImageNet class folders, copies a few training images into the validation split (byte-exact, edited, and edited + cropped), and writes a mock Visual Layer export in the format the scripts expect.

```bash
bash demo/run_demo.sh              # exact + perceptual leak detection, a few seconds on CPU
bash demo/run_demo.sh --semantic   # also ResNet50 feature similarity (downloads ~100 MB weights)
```

## What it does

Three command-line tools, each reading a dataset (or a Visual Layer export) and writing CSV reports:

| Script | Input | Output |
|---|---|---|
| `scripts/vl_tagging_workflow.py` | Visual Layer *flagged samples* export (CSV/JSON) | Tagging manifest with a fixed tag taxonomy (keep / remove / relabel, `mislabel_confirmed`, `duplicate_near`, ...) + an HTML review page |
| `scripts/train_test_leak_detection.py` | `train/` and `val/` image folders | Leak report of train/val image pairs that are copies, plus a summary |
| `scripts/analyze_vl_duplicates.py` | Visual Layer *duplicate clusters* export (JSON) | Leak report, val removal list, and multi-label clusters (possible mislabels) |

## How it works

- **Exact copies:** MD5 of the file bytes; any val file whose hash matches a train file is a leak.
- **Near copies:** 64-bit perceptual hash (`imagehash.phash`); pairs within a Hamming distance threshold (default 5 bits) are flagged.
- **Edited / cropped copies:** ResNet50 (ImageNet weights, final layer removed) embeds every image; each val image is matched to its most similar train image by cosine similarity (default threshold 0.95).
- **Visual Layer clusters:** a cluster that contains both train and val images is a leak. If its members share one label the val copy is marked `REMOVE_VAL`; if labels disagree it is `REVIEW_MANUAL` / HIGH priority. Train-only clusters with mixed labels are reported as possible mislabels.

In the bundled demo, MD5 catches the byte copies, pHash catches the brightness/resize edit, and only the ResNet50 method catches the cropped copy. See the [live demo](https://kush-vl-imagenet-cleaning.netlify.app) for the side-by-side images.

File formats and the tag taxonomy are documented in [`manifests/README.md`](manifests/README.md).

## Run on your own data

```bash
pip install -r scripts/requirements.txt

# 1. Visual Layer export -> tagging manifest + review page
python scripts/vl_tagging_workflow.py \
    --vl_export path/to/imagenet1k_vl_export.csv \
    --dataset imagenet1k \
    --output manifests/imagenet1k_tagged.csv \
    --generate_html

# 2. Train/val leak detection on image folders
python scripts/train_test_leak_detection.py \
    --dataset imagenet1k \
    --train_dir path/to/train \
    --test_dir path/to/val \
    --output manifests/imagenet1k_leaks.csv \
    --methods exact perceptual        # add "semantic" for ResNet50

# 3. Visual Layer duplicate clusters -> leaks + removal list
python scripts/analyze_vl_duplicates.py \
    --vl_json path/to/vl_duplicates.json \
    --output manifests/imagenet1k_leaks_from_vl.csv
```

Notes:
- The perceptual method compares every val hash against every train hash, so try a subset before running it on all of ImageNet.
- `analyze_vl_duplicates.py` decides the split from the path: anything containing `val` / `validation` is treated as validation.
- Column names in the Visual Layer exports are mapped with fallbacks (`image_path` or `path`, `label` or `class`, ...); adjust `create_tagging_manifest()` if your export differs.

## Project docs

- [`QUICK_START.md`](QUICK_START.md): checklist and copy-paste commands
- [`DATASET_CLEANING_GUIDE.md`](DATASET_CLEANING_GUIDE.md): the full cleaning workflow
- [`docs/WORKFLOW_DIAGRAM.md`](docs/WORKFLOW_DIAGRAM.md), [`docs/ANALYSIS_SUMMARY.md`](docs/ANALYSIS_SUMMARY.md): project context
- [`notebooks/imagenet_training_reference.md`](notebooks/imagenet_training_reference.md): notes from earlier ResNet-18 training runs

## Repository layout

```
scripts/        the three cleaning tools + requirements.txt
demo/           synthetic dataset generator, end-to-end demo runner, static site builder
manifests/      manifest format specs (actual data is git-ignored)
docs/           workflow diagram and analysis summary
.devcontainer/  Codespaces / Dev Container config
```

## Tech stack

Python 3, PyTorch + torchvision (ResNet50 features), imagehash (pHash), pandas, Pillow, tqdm. Demo page: plain HTML generated by `demo/build_site.py`, styled with Pico CSS, hosted on Netlify.

## Rebuilding the demo page

```bash
mkdir -p demo_output
bash demo/run_demo.sh --semantic 2>&1 | tee /tmp/run_log.txt
python demo/build_site.py --log /tmp/run_log.txt --out site   # every number is read from the outputs
```

## Credits

- [Visual Layer](https://www.visual-layer.com/): the data-quality platform whose exports these tools consume.
- [imagehash](https://github.com/JohannesBuchner/imagehash) (BSD-2-Clause), [PyTorch / torchvision](https://github.com/pytorch/vision) (BSD-3-Clause), [pandas](https://github.com/pandas-dev/pandas) (BSD-3-Clause), [Pillow](https://github.com/python-pillow/Pillow) (MIT-CMU).
- [Pico CSS](https://github.com/picocss/pico) (MIT): styling for the demo page.
- [devcontainers/images](https://github.com/devcontainers/images) (MIT): Codespaces base image.
- [shields.io](https://shields.io): badges.

## License

[MIT](LICENSE)
