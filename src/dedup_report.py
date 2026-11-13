"""Diagnostics for duplicate detection: exact (MD5) vs near-duplicate (dHash) matches.

    python -m src.dedup_report --data-dir /path/to/dataset [--montage out.jpg]
"""

import argparse
import hashlib
import os
from collections import Counter, defaultdict

import numpy as np

from .config import CLASSES
from .data import find_split_dirs, image_hash_bits, list_images


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--montage")
    args = ap.parse_args()

    tr_dir, te_dir = find_split_dirs(args.data_dir)
    tr_p, tr_y = list_images(tr_dir)
    te_p, te_y = list_images(te_dir)
    paths = tr_p + te_p
    y = np.concatenate([tr_y, te_y])
    split = np.array([0] * len(tr_p) + [1] * len(te_p))
    n_tr = len(tr_p)

    # Exact byte duplicates
    groups = defaultdict(list)
    for i, p in enumerate(paths):
        groups[md5(p)].append(i)
    dup_groups = [g for g in groups.values() if len(g) > 1]
    print(f"[exact MD5] groups with >1 file: {len(dup_groups)}, files in them: {sum(map(len, dup_groups))}")
    cross = sum(1 for g in dup_groups if 0 in split[g] and 1 in split[g])
    conflict = sum(1 for g in dup_groups if len(set(y[g])) > 1)
    print(f"[exact MD5] groups spanning train+test: {cross}, groups with conflicting labels: {conflict}")
    extra = Counter()
    for g in dup_groups:
        for i in g[1:]:
            extra[CLASSES[y[i]]] += 1
    print(f"[exact MD5] redundant copies by class: {dict(extra)}")

    # Near duplicates: nearest-neighbour Hamming distance (no chaining)
    bits = np.stack([image_hash_bits(p) for p in paths]).astype(np.float32)
    inv = 1 - bits
    nn_test = np.full(n_tr, 999.0)  # each train image -> closest test image
    nn_any = np.full(len(paths), 999.0)
    nn_idx = np.zeros(len(paths), dtype=int)
    for s in range(0, len(paths), 1024):
        a = bits[s:s + 1024]
        d = a @ inv.T + (1 - a) @ bits.T
        for r in range(len(a)):
            d[r, s + r] = 999
        nn_any[s:s + len(a)] = d.min(1)
        nn_idx[s:s + len(a)] = d.argmin(1)
        if s < n_tr:
            rows = min(len(a), n_tr - s)
            nn_test[s:s + rows] = d[:rows, n_tr:].min(1)

    print("\n[dHash] train images with a test image within distance t (no chaining):")
    for t in (0, 2, 4, 6, 8, 10, 14, 20):
        m = nn_test <= t
        per = {c: int(m[tr_y == i].sum()) for i, c in enumerate(CLASSES)}
        print(f"  t<={t:>2}: {int(m.sum()):>5}  {per}")
    print("\n[dHash] nearest-neighbour distance percentiles per class (all images):")
    for i, c in enumerate(CLASSES):
        v = nn_any[y == i]
        print(f"  {c:<11} p5={np.percentile(v,5):.0f} p25={np.percentile(v,25):.0f} "
              f"p50={np.percentile(v,50):.0f} p75={np.percentile(v,75):.0f}")

    if args.montage:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        rng = np.random.default_rng(0)
        rows = []
        for lo, hi in ((1, 4), (5, 8), (9, 12), (13, 18)):
            cand = np.where((nn_any >= lo) & (nn_any <= hi))[0]
            for i in rng.choice(cand, size=min(2, len(cand)), replace=False):
                rows.append((i, nn_idx[i], int(nn_any[i])))
        fig, axes = plt.subplots(len(rows), 2, figsize=(4, 2 * len(rows)))
        for r, (i, j, d) in enumerate(rows):
            for c, k in enumerate((i, j)):
                axes[r, c].imshow(Image.open(paths[k]).convert("L"), cmap="gray")
                axes[r, c].set_title(f"d={d} {CLASSES[y[k]]} {'tr' if split[k]==0 else 'te'}", fontsize=7)
                axes[r, c].axis("off")
        fig.tight_layout()
        fig.savefig(args.montage, dpi=70)
        print("montage saved", args.montage)


if __name__ == "__main__":
    main()
