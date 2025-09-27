# tools/uci_har_to_processed.py
# Convert UCI HAR (UCI Machine Learning Repository) to wifi_sensing_demo/data/processed.npz
# - Reads X_train.txt/y_train.txt and X_test.txt/y_test.txt
# - Splits part of train as validation (default 15%)
# - Reshapes features to (N, 561, 1), z-score using train stats
# - Saves processed.npz for your train_eval.py

import os
import numpy as np

ROOT = os.environ.get("UCI_HAR_ROOT", "data_uci_har/UCI HAR Dataset")
OUT  = os.environ.get("UCI_HAR_OUT",  "wifi_sensing_demo/data/processed.npz")
VAL_RATIO = float(os.environ.get("UCI_HAR_VAL_RATIO", "0.15"))

def load_txt(path, dtype=float):
    return np.loadtxt(path, dtype=dtype)

def load_split(root):
    X_train = load_txt(os.path.join(root, "train", "X_train.txt"), dtype=float)
    y_train = load_txt(os.path.join(root, "train", "y_train.txt"), dtype=int)
    X_test  = load_txt(os.path.join(root, "test",  "X_test.txt"),  dtype=float)
    y_test  = load_txt(os.path.join(root, "test",  "y_test.txt"),  dtype=int)
    return X_train, y_train, X_test, y_test

def make_val(X_train, y_train, val_ratio=0.15, seed=42):
    n = len(X_train)
    n_val = int(round(n * val_ratio))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]
    return (X_train[tr_idx], y_train[tr_idx],
            X_train[val_idx], y_train[val_idx])

def reshape_seq(X):
    # UCI HAR has 561 features per sample -> (561,1)
    n, f = X.shape
    return X.reshape(n, f, 1).astype(np.float32)

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    X_tr_raw, y_tr_raw, X_test_raw, y_test = load_split(ROOT)

    # train/val split from official training set
    X_train_raw, y_train, X_val_raw, y_val = make_val(X_tr_raw, y_tr_raw, VAL_RATIO)

    # reshape to (N, 561, 1)
    X_train = reshape_seq(X_train_raw)
    X_val   = reshape_seq(X_val_raw)
    X_test  = reshape_seq(X_test_raw)

    # labels: 1..6 -> 0..5
    if y_train.min() == 1: y_train = y_train - 1
    if y_val.min()   == 1: y_val   = y_val   - 1
    if y_test.min()  == 1: y_test  = y_test  - 1

    # z-score using train stats
    mean = X_train.mean(axis=(0,1), keepdims=True)
    std  = X_train.std(axis=(0,1), keepdims=True) + 1e-6
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    meta = {
        "seq_len": int(X_train.shape[1]),
        "channels": int(X_train.shape[2]),
        "class_names": ["WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS","SITTING","STANDING","LAYING"],
        "splits": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))}
    }

    np.savez_compressed(OUT,
        X_train=X_train, y_train=y_train.astype(np.int64),
        X_val=X_val,     y_val=y_val.astype(np.int64),
        X_test=X_test,   y_test=y_test.astype(np.int64),
        meta=np.array([str(meta)], dtype=object)
    )

    print(f"Saved {OUT}")
    print("Shapes:",
          "X_train", X_train.shape,
          "X_val",   X_val.shape,
          "X_test",  X_test.shape)

if __name__ == "__main__":
    main()
