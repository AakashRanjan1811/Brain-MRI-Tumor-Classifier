"""Test-set evaluation: accuracy, macro / per-class metrics, confusion matrix, ROC-AUC."""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from .config import CLASSES, NUM_CLASSES


def predict(model, ds, tta=False):
    """Class probabilities; with tta=True, averaged over the image and its horizontal flip."""
    y_prob = model.predict(ds, verbose=0)
    if tta:
        flipped = ds.map(lambda x, y: (tf.image.flip_left_right(x), y))
        y_prob = (y_prob + model.predict(flipped, verbose=0)) / 2
    y_true = np.concatenate([np.argmax(y, axis=1) for _, y in ds])
    return y_true, y_prob


def compute_metrics(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    onehot = np.eye(NUM_CLASSES)[y_true]
    report = classification_report(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), target_names=CLASSES,
        output_dict=True, zero_division=0,
    )
    try:
        auc = float(roc_auc_score(onehot, y_prob, multi_class="ovr", average="macro"))
        auc_per_class = {c: float(roc_auc_score(onehot[:, i], y_prob[:, i])) for i, c in enumerate(CLASSES)}
    except ValueError:  # a class missing from y_true (only possible on tiny smoke-test data)
        auc, auc_per_class = float("nan"), {}
    return {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_roc_auc": auc,
        "roc_auc_per_class": auc_per_class,
        "per_class": {
            c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score", "support")}
            for c in CLASSES
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist(),
        "n_errors": int((y_pred != y_true).sum()),
    }


def format_metrics(name, m):
    lines = [
        f"== {name} (test set, n={m['n_test']}) ==",
        f"accuracy        {m['accuracy']:.4f}",
        f"macro precision {m['macro_precision']:.4f}",
        f"macro recall    {m['macro_recall']:.4f}",
        f"macro F1        {m['macro_f1']:.4f}",
        f"macro ROC-AUC   {m['macro_roc_auc']:.4f}",
        f"errors          {m['n_errors']}",
        "per class (precision / recall / F1):",
    ]
    for c, v in m["per_class"].items():
        lines.append(f"  {c:<11} {v['precision']:.3f} / {v['recall']:.3f} / {v['f1-score']:.3f}")
    lines.append("confusion matrix (rows = true, cols = predicted, order = %s):" % CLASSES)
    lines += ["  " + str(r) for r in m["confusion_matrix"]]
    return "\n".join(lines)
