#!/usr/bin/env python3"

import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

DATA_ROOT = "/content/drive/MyDrive/DeepLearning_Project/data/train_images"
DATA_ROOT2 = "/content/drive/MyDrive/DeepLearning_Project/data/train_masks"

AUTHENTIC_DIR = Path(DATA_ROOT) / "authentic"
FORGED_DIR = Path(DATA_ROOT) / "forged"
MASK_DIR = Path(DATA_ROOT2)

SPLIT_DIR = Path("/content/drive/MyDrive/DeepLearning_Project/splits")
TRAIN_CSV = SPLIT_DIR / "train_split.csv"
VAL_CSV = SPLIT_DIR / "val_split.csv"

# Validate inputs early (terminal-friendly)
if not TRAIN_CSV.exists():
    raise FileNotFoundError(f"Missing split file: {TRAIN_CSV}")
if not VAL_CSV.exists():
    raise FileNotFoundError(f"Missing split file: {VAL_CSV}")
if not MASK_DIR.exists():
    raise FileNotFoundError(f"Missing mask dir: {MASK_DIR}")

# Training params
TARGET_SIZE = (512, 512)  # (H,W)
BATCH_SIZE = 8            # adjust based on GPU RAM
EPOCHS = 20
LR = 1e-3
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ImageNet normalization (safe baseline)
MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def decode_image(path):
    """Decode PNG/JPG robustly -> float32 RGB in [0,1]."""
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    return img


def resize_image_mask(img, mask, target_size=TARGET_SIZE):
    h, w = target_size
    img = tf.image.resize(img, (h, w), method="bilinear")
    mask = tf.image.resize(mask, (h, w), method="nearest")
    mask = tf.cast(mask > 0.5, tf.float32)  # ensure mask is {0,1}
    return img, mask


def normalize_imagenet(img):
    return (img - MEAN) / STD


def mask_path_for_image_py(image_path_str):
    """
    Match image stem -> mask file in MASK_DIR.
    Logic mirrors your earlier approach: stem.npy or stem*.npy
    Returns '' if not found.
    """
    p = Path(image_path_str)
    stem = p.stem
    direct = MASK_DIR / f"{stem}.npy"
    if direct.exists():
        return str(direct)
    candidates = sorted(MASK_DIR.glob(f"{stem}*.npy"))
    return str(candidates[0]) if len(candidates) else ""


def load_mask_npy_py(mask_path_str, orig_h, orig_w):
    """
    Load .npy instance mask (N,H,W), collapse -> (H,W) float32.
    If mask_path_str is empty, return zeros.
    """
    if (mask_path_str is None) or (mask_path_str == ""):
        return np.zeros((orig_h, orig_w), dtype=np.float32)

    inst = np.load(mask_path_str)  # (N,H,W)
    if inst.ndim != 3:
        raise ValueError(f"Unexpected mask shape {inst.shape} for {mask_path_str}")

    bin_mask = (inst.max(axis=0) > 0).astype(np.float32)  # (H,W)
    return bin_mask


# Data pipeline: load images + collapse masks
def build_dataset(csv_path, training=True):
    df = pd.read_csv(csv_path)
    if ("path" not in df.columns) or ("label" not in df.columns):
        raise ValueError("CSV must have columns: path,label")

    paths = df["path"].astype(str).values
    labels = df["label"].astype(str).values  # "authentic" / "forged"

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(df), 2000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def _load_example(path, label_str):
        img = decode_image(path)
        orig_shape = tf.shape(img)
        orig_h = orig_shape[0]
        orig_w = orig_shape[1]

        is_forged = tf.equal(label_str, tf.constant("forged"))

        def _load_forged_mask():
            mpath = tf.py_function(
                func=lambda x: mask_path_for_image_py(x.numpy().decode("utf-8")),
                inp=[path],
                Tout=tf.string,
            )
            mpath.set_shape([])

            m = tf.py_function(
                func=lambda mp, h, w: load_mask_npy_py(
                    mp.numpy().decode("utf-8"),
                    int(h.numpy()),
                    int(w.numpy()),
                ),
                inp=[mpath, orig_h, orig_w],
                Tout=tf.float32,
            )
            m.set_shape([None, None])
            m = tf.expand_dims(m, axis=-1)  # (H,W,1)
            return m

        def _load_auth_mask():
            return tf.zeros([orig_h, orig_w, 1], dtype=tf.float32)

        mask = tf.cond(is_forged, _load_forged_mask, _load_auth_mask)

        img, mask = resize_image_mask(img, mask, TARGET_SIZE)
        img = normalize_imagenet(img)

        return img, mask

    ds = ds.map(_load_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


# Baseline U-Net
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def unet(input_shape=(512, 512, 3), base_filters=32):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPool2D()(c1)

    c2 = conv_block(p1, base_filters * 2)
    p2 = layers.MaxPool2D()(c2)

    c3 = conv_block(p2, base_filters * 4)
    p3 = layers.MaxPool2D()(c3)

    c4 = conv_block(p3, base_filters * 8)
    p4 = layers.MaxPool2D()(c4)

    # Bottleneck
    bn = conv_block(p4, base_filters * 16)

    # Decoder
    u4 = layers.UpSampling2D()(bn)
    u4 = layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, base_filters * 8)

    u3 = layers.UpSampling2D()(c5)
    u3 = layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, base_filters * 4)

    u2 = layers.UpSampling2D()(c6)
    u2 = layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, base_filters * 2)

    u1 = layers.UpSampling2D()(c7)
    u1 = layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, base_filters)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c8)
    return Model(inputs, outputs, name="UNetBaseline")

# Losses + metrics
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denom = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
    return tf.reduce_mean((2.0 * intersection + smooth) / (denom + smooth))


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


bce = tf.keras.losses.BinaryCrossentropy()


def bce_dice_loss(y_true, y_pred):
    return bce(y_true, y_pred) + dice_loss(y_true, y_pred)


def iou_coef(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    inter = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    union = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1) - inter
    return tf.reduce_mean((inter + smooth) / (union + smooth))


# Main
def main():
    train_ds = build_dataset(TRAIN_CSV, training=True)
    val_ds = build_dataset(VAL_CSV, training=False)

    model = unet(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3), base_filters=32)

    OUT_DIR = Path("/content/drive/MyDrive/DeepLearning_Project/outputs_unet_baseline")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = OUT_DIR / "unet_baseline_best.keras"
    log_path = OUT_DIR / "train_log.csv"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=bce_dice_loss,
        metrics=[dice_coef, iou_coef],
    )

    callbacks = [
        ModelCheckpoint(str(ckpt_path), monitor="val_dice_coef", mode="max", save_best_only=True),
        EarlyStopping(monitor="val_dice_coef", mode="max", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_dice_coef", mode="max", factor=0.5, patience=3, min_lr=1e-6),
        CSVLogger(str(log_path)),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,  # keep Keras progress; remove/0 if you want silent
    )


if __name__ == "__main__":
    # reduce TF logging noise (kept minimal, no prints)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()