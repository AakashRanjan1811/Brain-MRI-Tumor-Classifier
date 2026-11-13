"""Training loops with early stopping, LR scheduling and best-checkpoint restore."""

import os
import time

import keras

from .models import build_baseline_cnn, build_efficientnet, unfreeze_backbone


def _callbacks(ckpt_path, patience):
    return [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_loss", save_best_only=True, save_weights_only=True
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=max(2, patience // 2), min_lr=1e-6, verbose=1
        ),
    ]


def _fit(model, train_ds, val_ds, epochs, ckpt_path, patience, phase, class_weight=None):
    start = time.time()
    hist = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        callbacks=_callbacks(ckpt_path, patience), class_weight=class_weight, verbose=2,
    )
    model.load_weights(ckpt_path)  # best epoch by validation loss, not the last epoch
    h = {k: [float(v) for v in vals] for k, vals in hist.history.items() if k != "learning_rate"}
    h["phase"] = [phase] * len(h["loss"])
    return h, time.time() - start


def _merge(histories):
    out = {}
    for h in histories:
        for k, v in h.items():
            out.setdefault(k, []).extend(v)
    return out


def train_baseline(train_ds, val_ds, out_dir, img_size=150, epochs=30, patience=6, class_weight=None):
    model = build_baseline_cnn(img_size)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    ckpt = os.path.join(out_dir, "baseline.weights.h5")
    hist, secs = _fit(model, train_ds, val_ds, epochs, ckpt, patience, "train", class_weight)
    return model, hist, secs


def train_efficientnet(train_ds, val_ds, out_dir, img_size=224, pretrained=True,
                       head_epochs=5, finetune_epochs=30, patience=6, class_weight=None):
    """Two-phase transfer learning: (1) train the new head on a frozen backbone, (2) fine-tune."""
    model, backbone = build_efficientnet(img_size, pretrained=pretrained)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    ckpt = os.path.join(out_dir, "efficientnet.weights.h5")

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
    h1, s1 = _fit(model, train_ds, val_ds, head_epochs, ckpt, patience, "head", class_weight)

    unfreeze_backbone(backbone)
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=loss, metrics=["accuracy"])
    h2, s2 = _fit(model, train_ds, val_ds, finetune_epochs, ckpt, patience, "finetune", class_weight)
    return model, _merge([h1, h2]), s1 + s2
