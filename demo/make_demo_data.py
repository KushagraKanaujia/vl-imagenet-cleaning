"""
Generate a tiny, fully synthetic dataset for trying the cleaning tools
without downloading ImageNet or having a Visual Layer account.

It creates:
  demo_data/train/<class>/*.png      synthetic "training" images
  demo_data/val/<class>/*.png        synthetic "validation" images, some of which
                                     are planted copies of training images:
                                       - exact byte-for-byte copies
                                       - edited copies (brightness/contrast + resize)
                                       - edited + cropped copies
  demo_data/vl_export.csv            a mock Visual Layer "flagged samples" export
  demo_data/vl_duplicates.json       a mock Visual Layer duplicate-cluster export
  demo_data/planted_leaks.json       ground truth for the planted leaks

Everything here is generated procedurally with a fixed seed. None of it is real
ImageNet data and none of it is real Visual Layer output; the mock exports just
follow the column/field names the scripts expect (see manifests/README.md).

Usage:
    python demo/make_demo_data.py [--out demo_data] [--seed 0]
"""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance

CLASSES = ["n01440764_tench", "n01443537_goldfish", "n01484850_great_white_shark"]
SIZE = 128


def synth_image(rng: random.Random) -> Image.Image:
    """A random composition of a gradient background plus a few shapes."""
    c1 = tuple(rng.randrange(256) for _ in range(3))
    c2 = tuple(rng.randrange(256) for _ in range(3))
    img = Image.new("RGB", (SIZE, SIZE))
    px = img.load()
    horizontal = rng.random() < 0.5
    for y in range(SIZE):
        for x in range(SIZE):
            t = (x if horizontal else y) / (SIZE - 1)
            px[x, y] = tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(3, 7)):
        x0, y0 = rng.randrange(SIZE - 20), rng.randrange(SIZE - 20)
        x1, y1 = x0 + rng.randint(12, 60), y0 + rng.randint(12, 60)
        fill = tuple(rng.randrange(256) for _ in range(3))
        shape = rng.choice(["ellipse", "rectangle", "line"])
        if shape == "ellipse":
            draw.ellipse([x0, y0, x1, y1], fill=fill)
        elif shape == "rectangle":
            draw.rectangle([x0, y0, x1, y1], fill=fill)
        else:
            draw.line([x0, y0, x1, y1], fill=fill, width=rng.randint(2, 6))
    return img


def near_copy(img: Image.Image, crop: bool) -> Image.Image:
    """Simulate an edited copy of an image.

    crop=False: brightness tweak + resize (perceptual hashing usually catches it)
    crop=True:  also re-framed with a small crop (usually only deep features catch it)
    """
    out = ImageEnhance.Brightness(img).enhance(1.15)
    out = ImageEnhance.Contrast(out).enhance(0.85)
    if crop:
        out = out.crop((6, 6, SIZE - 6, SIZE - 6))
    return out.resize((SIZE - 16, SIZE - 16), Image.BILINEAR)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", default="demo_data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_per_class", type=int, default=8)
    ap.add_argument("--val_per_class", type=int, default=3)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    train, val = [], []  # (rel_path, class_idx)
    for ci, cls in enumerate(CLASSES):
        for split, n, bucket in (("train", args.train_per_class, train),
                                 ("val", args.val_per_class, val)):
            d = out / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                p = d / f"{cls.split('_')[0]}_{split}_{i:03d}.png"
                synth_image(rng).save(p)
                bucket.append((p.relative_to(out).as_posix(), ci))

    # Plant leaks: copy some training images into the validation split.
    planted = []
    picks = rng.sample(train, 4)
    for k, (rel, ci) in enumerate(picks):
        src = out / rel
        kind = ["exact", "exact", "near", "cropped"][k]
        dst_rel = f"val/{CLASSES[ci]}/leak_{kind}_{k}.png"
        dst = out / dst_rel
        if kind == "exact":
            shutil.copyfile(src, dst)
        else:
            near_copy(Image.open(src), crop=(kind == "cropped")).save(dst)
        val.append((dst_rel, ci))
        planted.append({"train_image": rel, "val_image": dst_rel, "kind": kind,
                        "label": ci})

    # Mock Visual Layer "flagged samples" export (CSV).
    # image_path is written relative to the demo output folder so the generated
    # review_interface.html can display the thumbnails.
    flagged = []
    issue_rows = [
        ("mislabel", 0.95, "Embedding closer to another class centroid"),
        ("mislabel", 0.91, "Label disagrees with nearest neighbours"),
        ("outlier", 0.87, "Unusual colour distribution for class"),
        ("outlier", 0.72, "Low similarity to rest of class"),
        ("duplicate", 0.99, "Near-identical image found in dataset"),
        ("blur", 0.66, "Low sharpness score"),
    ]
    sample = rng.sample(train, len(issue_rows))
    for i, ((rel, ci), (issue, conf, reason)) in enumerate(zip(sample, issue_rows)):
        suggested = (ci + 1) % len(CLASSES) if issue == "mislabel" else ""
        flagged.append({
            "image_id": 10000 + i,
            "image_path": f"../demo_data/{rel}",
            "label": ci,
            "issue_type": issue,
            "confidence": conf,
            "suggested_label": suggested,
            "reason": reason,
            "split": "train",
        })
    with open(out / "vl_export.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flagged[0].keys()))
        w.writeheader()
        w.writerows(flagged)

    # Mock Visual Layer duplicate-cluster export (JSON).
    clusters = []
    for k, leak in enumerate(planted):
        val_label = leak["label"]
        # Give one near-copy a different label to exercise the
        # "different_labels" (HIGH priority) branch.
        if k == 3:
            val_label = (leak["label"] + 1) % len(CLASSES)
        clusters.append({
            "cluster_id": f"dup_{k:03d}",
            "images": [
                {"id": f"train_{k}", "path": leak["train_image"], "label": leak["label"]},
                {"id": f"val_{k}", "path": leak["val_image"], "label": val_label},
            ],
        })
    # A train-only cluster whose members disagree on label (possible mislabel).
    a, b = rng.sample([t for t in train if t[1] == 0], 2)
    clusters.append({
        "cluster_id": "dup_train_only",
        "images": [
            {"id": "train_a", "path": a[0], "label": 0},
            {"id": "train_b", "path": b[0], "label": 2},
        ],
    })
    (out / "vl_duplicates.json").write_text(json.dumps({"clusters": clusters}, indent=2))
    (out / "planted_leaks.json").write_text(json.dumps(planted, indent=2))

    print(f"Wrote {len(train)} train and {len(val)} val images to {out}/")
    print(f"Planted {len(planted)} leaks "
          f"({sum(p['kind'] == 'exact' for p in planted)} exact, "
          f"{sum(p['kind'] == 'near' for p in planted)} edited, "
          f"{sum(p['kind'] == 'cropped' for p in planted)} edited+cropped)")
    print(f"Mock VL export: {out / 'vl_export.csv'} ({len(flagged)} rows)")
    print(f"Mock VL duplicates: {out / 'vl_duplicates.json'} ({len(clusters)} clusters)")


if __name__ == "__main__":
    main()
