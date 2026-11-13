"""Grad-CAM heatmaps for the EfficientNet model (which image regions drove the prediction)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf

from .config import CLASS_DISPLAY, CLASSES
from .models import HEAD_LAYERS


def gradcam(model, image, layer=None):
    """Return (heatmap in [0, 1] at image resolution, predicted class index, confidence).

    `image` is a float32 array (H, W, 3) in [0, 255]. The augmentation block is skipped.
    `layer` names a layer inside the EfficientNet backbone to compute the map on; by default the
    backbone's last feature map is used.
    """
    backbone = next(l for l in model.layers if l.name.startswith("efficientnet"))
    dense = model.get_layer(HEAD_LAYERS[-1])
    kernel = tf.cast(dense.kernel, tf.float32)
    bias = tf.cast(dense.bias, tf.float32)
    if layer:
        backbone = keras.Model(backbone.input, [backbone.get_layer(layer).output, backbone.output])
    x = tf.convert_to_tensor(image[None], tf.float32)
    with tf.GradientTape() as tape:
        if layer:
            feats, top = backbone(x, training=False)
        else:
            feats = top = backbone(x, training=False)
        tape.watch(feats)
        # Gradient of the class *logit* (pre-softmax), as in the original Grad-CAM paper;
        # softmax probabilities saturate and give noisy maps. Dropout is inactive at inference.
        pooled = tf.reduce_mean(tf.cast(top, tf.float32), axis=(1, 2))  # = GlobalAveragePooling2D
        logits = tf.matmul(pooled, kernel) + bias
        probs = tf.nn.softmax(logits)
        cls = int(tf.argmax(logits[0]))
        score = logits[0, cls]
    grads = tf.cast(tape.gradient(score, feats), tf.float32)
    feats = tf.cast(feats, tf.float32)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(feats[0] * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], image.shape[:2], method="bilinear")[..., 0]
    return cam.numpy(), cls, float(probs[0, cls])


def pick_examples(y_true, y_prob, n_errors=2):
    """One confidently-correct example per class, plus up to `n_errors` misclassifications."""
    y_pred = y_prob.argmax(1)
    picks = []
    for c in range(len(CLASSES)):
        idx = np.where((y_true == c) & (y_pred == c))[0]
        if len(idx):
            picks.append(int(idx[np.argmax(y_prob[idx, c])]))
    wrong = np.where(y_pred != y_true)[0]
    if len(wrong):
        conf = y_prob[wrong].max(1)
        picks += [int(i) for i in wrong[np.argsort(-conf)][:n_errors]]
    return picks


def gradcam_figure(model, images, y_true, out_path, title="EfficientNet", layer=None):
    """Two rows: input MRI on top, Grad-CAM overlay below, one column per example."""
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6), squeeze=False)
    for j, (img, t) in enumerate(zip(images, y_true)):
        cam, p, conf = gradcam(model, img, layer)
        gray = img.mean(-1) / 255.0
        axes[0, j].imshow(gray, cmap="gray")
        axes[0, j].set_title(f"True: {CLASS_DISPLAY[CLASSES[t]]}", fontsize=9)
        axes[1, j].imshow(gray, cmap="gray")
        axes[1, j].imshow(cam, cmap="jet", alpha=0.4)
        ok = "correct" if p == t else "WRONG"
        axes[1, j].set_title(f"Pred: {CLASS_DISPLAY[CLASSES[p]]} ({conf:.0%}, {ok})", fontsize=8,
                             color="black" if p == t else "red")
        for a in axes[:, j]:
            a.axis("off")
    fig.suptitle(f"Grad-CAM: regions that drove the {title} prediction", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90, pil_kwargs={"quality": 80} if out_path.endswith(".jpg") else None)
    plt.close(fig)
