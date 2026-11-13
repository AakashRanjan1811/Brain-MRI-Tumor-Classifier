"""Dataset listing, duplicate removal, stratified splitting and tf.data pipelines."""

import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

from .config import CLASSES, NUM_CLASSES, SEED, VAL_FRACTION

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_split_dirs(root):
    """Return (train_dir, test_dir), searching a few levels below `root`."""
    for dirpath, dirnames, _ in os.walk(root):
        if "Training" in dirnames and "Testing" in dirnames:
            return os.path.join(dirpath, "Training"), os.path.join(dirpath, "Testing")
    raise FileNotFoundError(f"Could not find Training/ and Testing/ folders under {root}")


def list_images(split_dir):
    """List image paths and integer labels for one split folder (one sub-folder per class)."""
    paths, labels = [], []
    for idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(split_dir, cls)
        for name in sorted(os.listdir(cls_dir)):
            if os.path.splitext(name)[1].lower() in IMG_EXTS:
                paths.append(os.path.join(cls_dir, name))
                labels.append(idx)
    return paths, np.array(labels, dtype=np.int64)


def image_hash_bits(path, hash_size=16):
    """256-bit difference hash (dHash) as a boolean vector.

    Robust to re-encoding and resizing, so it catches copies of the same scan that
    were saved at a different resolution or JPEG quality, not only byte-identical files.
    """
    with Image.open(path) as im:
        im = im.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        px = np.asarray(im, dtype=np.int16)
    return (px[:, 1:] > px[:, :-1]).flatten()


def duplicate_groups(paths, max_distance=10, chunk=1024):
    """Group near-identical images: Hamming distance between dHashes <= max_distance (of 256 bits).

    Returns one integer group id per path (union-find over all close pairs).
    """
    bits = np.stack([image_hash_bits(p) for p in paths]).astype(np.float32)
    n = len(paths)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    inv = 1.0 - bits
    for s in range(0, n, chunk):
        a = bits[s:s + chunk]
        dist = a @ inv.T + (1.0 - a) @ bits.T  # Hamming distance to every image
        rows, cols = np.nonzero(dist <= max_distance)
        for r, c in zip(rows + s, cols):
            if c > r:
                ra, rb = find(r), find(c)
                if ra != rb:
                    parent[rb] = ra
    return [int(find(i)) for i in range(n)]


def deduplicate(train_paths, train_labels, test_paths, test_labels, max_distance=10):
    """Remove duplicate images so that no scan is counted twice or leaks across splits.

    Rules (applied to groups of near-identical images):
      * a group containing more than one label is ambiguous -> dropped everywhere;
      * duplicates inside the test set -> keep the first copy;
      * training images that also appear in the test set -> dropped from training;
      * duplicates inside the training set -> keep the first copy.
    """
    groups = duplicate_groups(list(train_paths) + list(test_paths), max_distance)
    h_train, h_test = groups[:len(train_paths)], groups[len(train_paths):]

    labels_by_hash = defaultdict(set)
    for h, y in zip(h_train + h_test, list(train_labels) + list(test_labels)):
        labels_by_hash[h].add(int(y))
    conflicts = {h for h, ys in labels_by_hash.items() if len(ys) > 1}

    keep_test, seen_test = [], set()
    for i, h in enumerate(h_test):
        if h in conflicts or h in seen_test:
            continue
        seen_test.add(h)
        keep_test.append(i)

    keep_train, seen_train = [], set()
    n_train_in_test = 0
    for i, h in enumerate(h_train):
        if h in conflicts:
            continue
        if h in seen_test:
            n_train_in_test += 1
            continue
        if h in seen_train:
            continue
        seen_train.add(h)
        keep_train.append(i)

    stats = {
        "train_original": len(train_paths),
        "test_original": len(test_paths),
        "label_conflict_groups": len(conflicts),
        "train_removed_overlapping_test": n_train_in_test,
        "train_after_dedup": len(keep_train),
        "test_after_dedup": len(keep_test),
        "train_removed_total": len(train_paths) - len(keep_train),
        "test_removed_total": len(test_paths) - len(keep_test),
    }
    train_paths = [train_paths[i] for i in keep_train]
    test_paths = [test_paths[i] for i in keep_test]
    return (train_paths, train_labels[keep_train], test_paths, test_labels[keep_test], stats)


def prepare_splits(data_root, dedup=True):
    """Build train / validation / test splits. The test set is only used for the final evaluation."""
    train_dir, test_dir = find_split_dirs(data_root)
    tr_p, tr_y = list_images(train_dir)
    te_p, te_y = list_images(test_dir)
    if dedup:
        tr_p, tr_y, te_p, te_y, stats = deduplicate(tr_p, tr_y, te_p, te_y)
    else:
        stats = {"train_original": len(tr_p), "test_original": len(te_p)}

    tr_p, va_p, tr_y, va_y = train_test_split(
        tr_p, tr_y, test_size=VAL_FRACTION, stratify=tr_y, random_state=SEED
    )
    stats.update({"train": len(tr_p), "val": len(va_p), "test": len(te_p)})
    stats["class_counts"] = {
        split: {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}
        for split, y in (("train", tr_y), ("val", va_y), ("test", te_y))
    }
    return (tr_p, tr_y), (va_p, va_y), (te_p, te_y), stats


def make_dataset(paths, labels, img_size, batch_size=32, shuffle=False):
    """tf.data pipeline yielding float32 images in [0, 255] and one-hot labels.

    Pixel scaling is done inside each model (Rescaling layer for the baseline CNN,
    EfficientNet's built-in preprocessing for the transfer model), so the same
    pipeline works for both and inference code cannot forget to normalise.
    """

    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size), antialias=True)
        img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)  # cache compactly
        return img, tf.one_hot(label, NUM_CLASSES)

    ds = tf.data.Dataset.from_tensor_slices((list(paths), np.asarray(labels)))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).cache()
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).map(lambda x, y: (tf.cast(x, tf.float32), y))
    return ds.prefetch(tf.data.AUTOTUNE)


def load_image(path, img_size):
    """Load one image as a float32 array in [0, 255] (same preprocessing as the pipeline)."""
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size), antialias=True)
    return tf.clip_by_value(tf.round(img), 0, 255).numpy().astype(np.float32)
