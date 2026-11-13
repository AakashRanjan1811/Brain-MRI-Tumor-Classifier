"""End-to-end pipeline: dedupe -> split -> train baseline + EfficientNet -> evaluate -> figures -> Grad-CAM.

Usage (Colab / any GPU machine):
    python -m src.run --data-dir /path/to/brain-tumor-mri-dataset --out results
"""

import argparse
import base64
import json
import os
import platform
import time

import keras
import numpy as np
import tensorflow as tf

from . import plots
from .config import CLASSES, SEED
from .data import load_image, make_dataset, prepare_splits
from .evaluate import compute_metrics, format_metrics, predict
from .gradcam import gradcam_figure, pick_examples
from .train import train_baseline, train_efficientnet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", help="Dataset root (contains Training/ and Testing/). Downloaded from Kaggle if omitted.")
    ap.add_argument("--out", default="results")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--baseline-size", type=int, default=150)
    ap.add_argument("--effnet-size", type=int, default=224)
    ap.add_argument("--baseline-epochs", type=int, default=40)
    ap.add_argument("--head-epochs", type=int, default=5)
    ap.add_argument("--finetune-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--baseline-patience", type=int, default=8)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--only-baseline", action="store_true",
                    help="Train only the baseline and merge it into an existing results/metrics.json.")
    ap.add_argument("--no-pretrained", action="store_true", help="Random init (offline smoke tests only).")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--print-export", action="store_true", help="Print metrics + Grad-CAM image (base64) to stdout.")
    args = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"TensorFlow {tf.__version__} | Keras {keras.__version__} | GPUs: {[g.name for g in gpus]}")

    if not args.data_dir:
        import kagglehub
        from .config import KAGGLE_DATASET
        args.data_dir = kagglehub.dataset_download(KAGGLE_DATASET)

    out = args.out
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)

    t0 = time.time()
    (tr_p, tr_y), (va_p, va_y), (te_p, te_y), split_stats = prepare_splits(args.data_dir, dedup=not args.no_dedup)
    print("Split stats:", json.dumps(split_stats, indent=1))

    # Balanced class weights: de-duplication leaves "notumor" with far fewer unique training images.
    counts = np.bincount(tr_y, minlength=len(CLASSES))
    class_weight = {i: float(len(tr_y) / (len(CLASSES) * c)) for i, c in enumerate(counts)}
    print("Class weights:", {CLASSES[i]: round(w, 3) for i, w in class_weight.items()})

    results = {
        "classes": CLASSES,
        "data": split_stats,
        "environment": {
            "tensorflow": tf.__version__, "keras": keras.__version__, "python": platform.python_version(),
            "gpu": tf.config.experimental.get_device_details(gpus[0]).get("device_name") if gpus else None,
            "mixed_precision": bool(gpus),
        },
        "class_weight": {CLASSES[i]: w for i, w in class_weight.items()},
        "models": {},
    }

    def record(name, model, hist, secs, size):
        test_ds = make_dataset(te_p, te_y, size, args.batch_size)
        y_true, y_prob = predict(model, test_ds)
        m = compute_metrics(y_true, y_prob)
        print(format_metrics(name, m))
        results["models"][name] = {
            "image_size": size, "params": int(model.count_params()), "train_seconds": round(secs, 1),
            "epochs_run": len(hist["loss"]), "history": hist, "test_metrics": m,
        }
        model.save(os.path.join(out, "models", f"{name}.keras"))
        return y_true, y_prob

    if not args.skip_baseline:
        s = args.baseline_size
        model, hist, secs = train_baseline(
            make_dataset(tr_p, tr_y, s, args.batch_size, shuffle=True),
            make_dataset(va_p, va_y, s, args.batch_size), out, s, args.baseline_epochs, args.baseline_patience, class_weight,
        )
        record("baseline_cnn", model, hist, secs, s)
        del model
        keras.backend.clear_session()

    if args.only_baseline:
        path = os.path.join(out, "metrics.json")
        with open(path) as f:
            merged = json.load(f)
        merged["models"]["baseline_cnn"] = results["models"]["baseline_cnn"]
        with open(path, "w") as f:
            json.dump(merged, f, indent=1)
        plots.make_all(out)
        print("Merged baseline into", path)
        return

    s = args.effnet_size
    model, hist, secs = train_efficientnet(
        make_dataset(tr_p, tr_y, s, args.batch_size, shuffle=True),
        make_dataset(va_p, va_y, s, args.batch_size), out, s,
        pretrained=not args.no_pretrained, head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs, patience=args.patience, class_weight=class_weight,
    )
    y_true, y_prob = record("efficientnet_b0", model, hist, secs, s)

    picks = pick_examples(y_true, y_prob)
    cam_path = os.path.join(out, "figures", "gradcam.jpg")
    gradcam_figure(model, [load_image(te_p[i], s) for i in picks], [int(y_true[i]) for i in picks], cam_path)

    results["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=1)
    plots.make_all(out)
    print(f"Done in {results['total_seconds'] / 60:.1f} min. Results in {out}/")

    if args.print_export:
        with open(cam_path, "rb") as f:
            cam_b64 = base64.b64encode(f.read()).decode()
        print("===METRICS_JSON_BEGIN===")
        print(json.dumps(results, separators=(",", ":")))
        print("===METRICS_JSON_END===")
        print("===GRADCAM_B64_BEGIN===")
        print(cam_b64)
        print("===GRADCAM_B64_END===")


if __name__ == "__main__":
    main()
