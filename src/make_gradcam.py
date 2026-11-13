"""Regenerate the Grad-CAM figure from a saved model, without retraining.

    python -m src.make_gradcam --model results_b3/models/efficientnet_b3.keras --out results_b3/figures/gradcam.jpg
"""

import argparse

import keras

from .data import load_image, make_dataset, prepare_splits
from .evaluate import predict
from .gradcam import gradcam_figure, pick_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir")
    ap.add_argument("--title", default="EfficientNetB3")
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    if not args.data_dir:
        import kagglehub
        from .config import KAGGLE_DATASET
        args.data_dir = kagglehub.dataset_download(KAGGLE_DATASET)

    model = keras.models.load_model(args.model)
    size = model.input_shape[1]
    _, _, (te_p, te_y), _ = prepare_splits(args.data_dir)  # same deterministic split as training
    y_true, y_prob = predict(model, make_dataset(te_p, te_y, size), tta=args.tta)
    picks = pick_examples(y_true, y_prob)
    gradcam_figure(model, [load_image(te_p[i], size) for i in picks], [int(y_true[i]) for i in picks],
                   args.out, title=args.title)
    print(f"test accuracy check: {(y_prob.argmax(1) == y_true).mean():.4f} | saved {args.out}")


if __name__ == "__main__":
    main()
