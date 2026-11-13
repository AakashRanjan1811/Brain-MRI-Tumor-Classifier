"""Model definitions: the original CNN (baseline) and an EfficientNetB0 transfer-learning model."""

import keras
from keras import layers

from .config import NUM_CLASSES

EFFNET_BASE_NAME = "efficientnetb0"
HEAD_LAYERS = ("gap", "head_dropout", "predictions")


def augmentation_block():
    """Light, anatomy-preserving augmentation. Active only during training."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )


def build_baseline_cnn(img_size=150):
    """The original project architecture (same layers, same 4,447,044 parameters).

    Changes vs. the original notebook: an explicit Input, pixel scaling to [0, 1], and He
    initialisation for the ReLU layers. Without BatchNorm, the default Glorot init shrinks the
    signal through this deep ReLU stack once inputs are in [0, 1] and training stalls at chance
    level; the original only trained because it was fed raw 0-255 pixels.
    """
    he = "he_normal"

    def conv(filters):
        return layers.Conv2D(filters, 3, activation="relu", kernel_initializer=he)

    inputs = keras.Input((img_size, img_size, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = conv(32)(x)
    x = conv(64)(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = conv(64)(x)
    x = conv(64)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = conv(128)(x)
    x = conv(128)(x)
    x = conv(128)(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = conv(128)(x)
    x = conv(256)(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu", kernel_initializer=he)(x)
    x = layers.Dense(512, activation="relu", kernel_initializer=he)(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def build_efficientnet(img_size=224, pretrained=True, dropout=0.3):
    """EfficientNetB0 backbone (ImageNet weights) + augmentation + small classification head.

    EfficientNet in Keras contains its own input normalisation and expects pixels in [0, 255].
    Returns (model, backbone) so the training loop can freeze / unfreeze the backbone.
    """
    inputs = keras.Input((img_size, img_size, 3))
    x = augmentation_block()(inputs)
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(img_size, img_size, 3),
    )
    assert backbone.name == EFFNET_BASE_NAME, backbone.name
    backbone.trainable = False
    # training=False keeps BatchNorm in inference mode even after unfreezing (standard fine-tuning practice).
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="predictions")(x)
    return keras.Model(inputs, outputs, name="efficientnetb0_classifier"), backbone


def unfreeze_backbone(backbone):
    """Unfreeze the whole backbone except BatchNormalization layers."""
    backbone.trainable = True
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
