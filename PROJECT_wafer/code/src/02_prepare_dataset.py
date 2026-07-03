# src/02_prepare_dataset.py
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


UNLABELED_TOKEN = "__unlabeled__"

DEFECT8_CLASSES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
]

ALL9_CLASSES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "Normal",
]


def unwrap_nested_value(x):
    if x is None:
        return None

    if isinstance(x, float) and np.isnan(x):
        return None

    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        return unwrap_nested_value(x.flatten()[0])

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        return unwrap_nested_value(x[0])

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    s = str(x).strip()
    if s in ["", "[]", "None", "nan", "NaN"]:
        return None

    s = s.replace("[", "").replace("]", "").replace("'", "").replace('"', "").strip()

    if s == "":
        return None

    return s


def normalize_label(x):
    label = unwrap_nested_value(x)
    if label is None:
        return UNLABELED_TOKEN

    mapping = {
        "Edge-Loc": "Edge-Loc",
        "Edge-ring": "Edge-Ring",
        "Edge-Ring": "Edge-Ring",
        "Near-full": "Near-full",
        "Near-Full": "Near-full",
        "Loc": "Loc",
        "Random": "Random",
        "Scratch": "Scratch",
        "Center": "Center",
        "Donut": "Donut",
        "none": "none",
        "None": "none",
    }

    return mapping.get(label, label)


def normalize_train_test_label(x):
    label = unwrap_nested_value(x)
    if label is None:
        return None
    key = str(label).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if key in {"training", "train"}:
        return "train"
    if key in {"test", "testing"}:
        return "test"
    return None


def find_official_split_column(df):
    candidates = [
        "trianTestLabel",
        "trainTestLabel",
        "trainingTestLabel",
        "trainingOrTestLabel",
        "training or test set label",
    ]
    for name in candidates:
        if name in df.columns:
            return name
    for name in df.columns:
        key = str(name).lower().replace("_", "").replace("-", "").replace(" ", "")
        if ("train" in key or "trian" in key) and "test" in key:
            return name
    return None


def resize_wafer_map(wafer_map, image_size):
    """
    waferMap 通常是二维数组，值一般为 0/1/2。
    用最近邻插值，避免把离散类别插成小数。
    """
    arr = np.asarray(wafer_map)

    if arr.ndim != 2:
        raise ValueError(f"Invalid waferMap ndim={arr.ndim}, shape={arr.shape}")

    arr = arr.astype(np.uint8)

    img = Image.fromarray(arr)
    img = img.resize((image_size, image_size), resample=Image.Resampling.NEAREST)

    out = np.asarray(img).astype(np.uint8)
    return out


def canonical_label(label, label_to_id):
    if label in label_to_id:
        return label
    if label == "none" and "Normal" in label_to_id:
        return "Normal"
    if label == "Normal" and "none" in label_to_id:
        return "none"
    return label


def build_arrays(df, label_to_id, id_to_label, image_size):
    xs = []
    ys = []
    names = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing labeled wafers"):
        label = canonical_label(row["clean_label"], label_to_id)
        if label not in label_to_id:
            continue

        try:
            wafer = resize_wafer_map(row["waferMap"], image_size=image_size)
        except Exception as e:
            print(f"[WARN] Skip invalid waferMap: {e}")
            continue

        xs.append(wafer)
        ys.append(label_to_id[label])
        names.append(id_to_label[label_to_id[label]])

    x = np.stack(xs, axis=0).astype(np.uint8)
    y = np.asarray(ys, dtype=np.int64)
    names = np.asarray(names)

    return x, y, names


def build_unlabeled_arrays(df, image_size, max_unlabeled=None):
    if max_unlabeled is not None and len(df) > max_unlabeled:
        df = df.sample(n=max_unlabeled, random_state=42)

    xs = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing unlabeled wafers"):
        try:
            wafer = resize_wafer_map(row["waferMap"], image_size=image_size)
        except Exception as e:
            print(f"[WARN] Skip invalid waferMap: {e}")
            continue

        xs.append(wafer)

    if len(xs) == 0:
        return None

    x = np.stack(xs, axis=0).astype(np.uint8)
    return x


def to_one_hot_input(x, num_values=3):
    x = np.clip(x, 0, num_values - 1).astype(np.int64)
    x = np.eye(num_values, dtype=np.uint8)[x]
    return np.transpose(x, (0, 3, 1, 2))


def save_npz(path, x, y=None, names=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if y is None:
        np.savez_compressed(path, x=x)
    else:
        np.savez_compressed(path, x=x, y=y, label_names=names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default="data/raw/LSWMD.pkl",
                        help="Path to LSWMD.pkl")
    parser.add_argument("--out", type=str, default="data/processed",
                        help="Output directory")
    parser.add_argument("--image-size", type=int, default=32,
                        help="Resize waferMap to image_size x image_size")
    parser.add_argument("--classes", type=str, default="defect8",
                        choices=["defect8", "all9"],
                        help="defect8: only 8 defect classes; all9: include none class")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--split-mode", type=str, default="random", choices=["random", "official"],
                        help="random: stratified random split; official: use official Training/Test label for train/test")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save-unlabeled", action="store_true",
                        help="Save unlabeled samples for later pseudo-label training")
    parser.add_argument("--max-unlabeled", type=int, default=100000,
                        help="Max number of unlabeled samples to save")
    parser.add_argument("--max-none-samples", type=int, default=None,
                        help="Optional cap for labeled none samples when --classes all9 is used")
    parser.add_argument("--one-hot-input", action="store_true",
                        help="Save wafer maps as 3-channel one-hot tensors for values 0/1/2")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"Cannot find dataset: {raw_path}")

    if args.val_size <= 0 or args.val_size >= 1.0:
        raise ValueError("Require 0 < val_size < 1")
    if args.split_mode == "random" and (args.test_size <= 0 or args.test_size + args.val_size >= 1.0):
        raise ValueError("Require 0 < test_size, 0 < val_size, test_size + val_size < 1")

    print(f"[INFO] Loading dataset from: {raw_path}")
    df = pd.read_pickle(raw_path)
    df = df.copy()
    df["clean_label"] = df["failureType"].apply(normalize_label)

    class_names = DEFECT8_CLASSES if args.classes == "defect8" else ALL9_CLASSES
    label_to_id = {name: i for i, name in enumerate(class_names)}
    id_to_label = {i: name for name, i in label_to_id.items()}

    selected_labels = set(class_names)
    if args.classes == "all9":
        selected_labels.add("none")
    labeled_df = df[df["clean_label"].isin(selected_labels)].copy()
    unlabeled_df = df[df["clean_label"] == UNLABELED_TOKEN].copy()

    if args.classes == "all9" and args.max_none_samples is not None:
        none_df = labeled_df[labeled_df["clean_label"] == "none"]
        other_df = labeled_df[labeled_df["clean_label"] != "none"]
        if len(none_df) > args.max_none_samples:
            none_df = none_df.sample(n=args.max_none_samples, random_state=args.seed)
            labeled_df = pd.concat([other_df, none_df], axis=0).sample(
                frac=1.0,
                random_state=args.seed,
            )
            print(f"[INFO] Downsampled none samples to: {args.max_none_samples}")

    print("\n========== Selected Classes ==========")
    print(class_names)

    print("\n========== Labeled Counts ==========")
    display_counts = labeled_df["clean_label"].map(lambda item: canonical_label(item, label_to_id)).value_counts()
    print(display_counts)

    print(f"\n[INFO] Labeled samples selected: {len(labeled_df)}")
    print(f"[INFO] Unlabeled samples found:    {len(unlabeled_df)}")

    x, y, label_names = build_arrays(labeled_df, label_to_id, id_to_label, args.image_size)
    if args.one_hot_input:
        x = to_one_hot_input(x)

    print(f"\n[INFO] Processed x shape: {x.shape}, dtype={x.dtype}")
    print(f"[INFO] Processed y shape: {y.shape}, dtype={y.dtype}")

    # 第一次划分：train_val 和 test
    x_trainval, x_test, y_trainval, y_test, names_trainval, names_test = train_test_split(
        x,
        y,
        label_names,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    # 第二次划分：train 和 val
    val_ratio_in_trainval = args.val_size / (1.0 - args.test_size)

    x_train, x_val, y_train, y_val, names_train, names_val = train_test_split(
        x_trainval,
        y_trainval,
        names_trainval,
        test_size=val_ratio_in_trainval,
        random_state=args.seed,
        stratify=y_trainval,
    )

    print("\n========== Split Shapes ==========")
    print(f"Train: {x_train.shape}, {y_train.shape}")
    print(f"Val:   {x_val.shape}, {y_val.shape}")
    print(f"Test:  {x_test.shape}, {y_test.shape}")

    save_npz(out_dir / f"wafer_{args.image_size}_train.npz", x_train, y_train, names_train)
    save_npz(out_dir / f"wafer_{args.image_size}_val.npz", x_val, y_val, names_val)
    save_npz(out_dir / f"wafer_{args.image_size}_test.npz", x_test, y_test, names_test)

    label_info = {
        "classes_mode": args.classes,
        "image_size": args.image_size,
        "class_names": class_names,
        "label_to_id": label_to_id,
        "id_to_label": {str(k): v for k, v in id_to_label.items()},
        "num_classes": len(class_names),
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "input_channels": 3 if args.one_hot_input else 1,
        "one_hot_input": bool(args.one_hot_input),
        "max_none_samples": args.max_none_samples,
    }

    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_info, f, indent=2, ensure_ascii=False)

    # 保存无标签样本，供后面伪标签使用
    if args.save_unlabeled:
        if len(unlabeled_df) > 0:
            x_unlabeled = build_unlabeled_arrays(
                unlabeled_df,
                image_size=args.image_size,
                max_unlabeled=args.max_unlabeled,
            )
            if x_unlabeled is not None:
                if args.one_hot_input:
                    x_unlabeled = to_one_hot_input(x_unlabeled)
                save_npz(out_dir / f"wafer_{args.image_size}_unlabeled.npz", x_unlabeled)
                print(f"[INFO] Saved unlabeled x shape: {x_unlabeled.shape}")
        else:
            print("[WARN] No unlabeled samples found.")

    print("\n[INFO] Saved processed files to:")
    print(f"  {out_dir / f'wafer_{args.image_size}_train.npz'}")
    print(f"  {out_dir / f'wafer_{args.image_size}_val.npz'}")
    print(f"  {out_dir / f'wafer_{args.image_size}_test.npz'}")
    print(f"  {out_dir / 'label_map.json'}")


if __name__ == "__main__":
    main()
