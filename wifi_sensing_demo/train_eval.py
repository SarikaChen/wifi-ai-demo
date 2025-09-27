# wifi_sensing_demo/train_eval.py
# 完整版：EarlyStopping + 自動畫圖 + 匯出 H5/TFLite + (可選) 混淆矩陣
# 並且修正 processed.npz 的 meta 讀取相容性（支援 str / ndarray / scalar）

import os
import argparse
import ast
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="wifi_sensing_demo/data/processed.npz", help="processed.npz 路徑")
    p.add_argument("--use_synthetic", type=int, default=0, help="1=使用合成資料")
    p.add_argument("--epochs", type=int, default=100, help="最大 epoch，EarlyStopping 會自動提前終止")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="初始學習率")
    p.add_argument("--model_dir", default="wifi_sensing_demo/model", help="輸出資料夾")
    return p.parse_args()


def _parse_meta(meta_raw):
    """將 processed.npz 中的 meta 欄位轉成 dict。"""
    try:
        raw = meta_raw
        if isinstance(raw, np.ndarray):
            raw = raw[0]
        if isinstance(raw, str):
            return ast.literal_eval(raw)
        if hasattr(raw, "item"):
            raw = raw.item()
        if isinstance(raw, dict):
            return raw
        return {}
    except Exception:
        return {}


def load_processed_npz(path):
    d = np.load(path, allow_pickle=True)
    X_train = d["X_train"]; y_train = d["y_train"]
    X_val   = d["X_val"];   y_val   = d["y_val"]
    X_test  = d["X_test"];  y_test  = d["y_test"]
    meta = _parse_meta(d.get("meta", {}))
    return (X_train, y_train, X_val, y_val, X_test, y_test, meta)


def make_synthetic(n_train=1200, n_val=200, n_test=200, seq_len=128, channels=8, n_classes=3, seed=42):
    rng = np.random.default_rng(seed)
    def synth(n):
        X, y = [], []
        per = n // n_classes
        t = np.linspace(0, 2*np.pi, seq_len)[None, :, None]
        for c in range(n_classes):
            base = np.sin((c+1) * t + (c*0.5))
            noise = rng.normal(0, 0.3, size=(per, seq_len, channels))
            Xc = base + noise
            yc = np.full((per,), c, dtype=np.int64)
            X.append(Xc); y.append(yc)
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        idx = rng.permutation(len(X))
        return X[idx].astype(np.float32), y[idx]
    X_train, y_train = synth(n_train)
    X_val,   y_val   = synth(n_val)
    X_test,  y_test  = synth(n_test)
    mean = X_train.mean(axis=(0,1), keepdims=True)
    std  = X_train.std(axis=(0,1), keepdims=True) + 1e-6
    X_train = (X_train - mean)/std
    X_val   = (X_val   - mean)/std
    X_test  = (X_test  - mean)/std
    meta = {"seq_len": seq_len, "channels": channels, "class_names": [f"class_{i}" for i in range(n_classes)]}
    return X_train, y_train, X_val, y_val, X_test, y_test, meta


def build_model(input_shape, n_classes, lr=1e-3):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 5, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_learning_curves(history, out_path):
    hist = history.history
    plt.figure()
    plt.plot(hist.get("loss", []), label="train_loss")
    if "val_loss" in hist: plt.plot(hist["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_loss.png"), dpi=150)
    plt.close()

    if "accuracy" in hist:
        plt.figure()
        plt.plot(hist["accuracy"], label="train_acc")
        if "val_accuracy" in hist: plt.plot(hist["val_accuracy"], label="val_acc")
        plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


def maybe_confusion_matrix(model, X, y, out_path):
    try:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        y_pred = model.predict(X, verbose=0).argmax(axis=1)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        plt.figure()
        disp.plot(values_format="d")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"[WARN] 無法產生混淆矩陣（可選功能）：{e}")


def main():
    args = get_args()
    os.makedirs(args.model_dir, exist_ok=True)

    if args.use_synthetic:
        print("[INFO] 使用合成資料")
        X_train, y_train, X_val, y_val, X_test, y_test, meta = make_synthetic()
    else:
        if not os.path.exists(args.data):
            raise FileNotFoundError(f"找不到 {args.data}，請先產生 processed.npz。")
        X_train, y_train, X_val, y_val, X_test, y_test, meta = load_processed_npz(args.data)

    n_classes = int(np.max([y_train.max(), y_val.max(), y_test.max()])) + 1
    input_shape = X_train.shape[1:]

    print("[INFO] 資料形狀：")
    print("  X_train", X_train.shape, "y_train", y_train.shape)
    print("  X_val  ", X_val.shape,   "y_val  ", y_val.shape)
    print("  X_test ", X_test.shape,  "y_test ", y_test.shape)
    print("  classes:", n_classes, " input:", input_shape)
    print("  meta:", meta)

    model = build_model(input_shape, n_classes, lr=args.lr)
    model.summary()

    ckpt_path = os.path.join(args.model_dir, "best.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ModelCheckpoint(filepath=ckpt_path, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    curve_path = os.path.join(args.model_dir, "training_curve.png")
    plot_learning_curves(history, curve_path)
    print(f"[OK] 訓練曲線已輸出：{curve_path}")

    try:
        model.load_weights(ckpt_path)
    except Exception:
        pass

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    cm_path = os.path.join(args.model_dir, "confusion_matrix.png")
    maybe_confusion_matrix(model, X_test, y_test, cm_path)

    h5_path = os.path.join(args.model_dir, "csi_cnn.h5")
    model.save(h5_path)
    print(f"[OK] 已儲存 Keras 模型：{h5_path}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tfl_path = os.path.join(args.model_dir, "csi_cnn.tflite")
    with open(tfl_path, "wb") as f:
        f.write(tflite_model)
    print(f"[OK] 已匯出 TFLite：{tfl_path}")


if __name__ == "__main__":
    main()
