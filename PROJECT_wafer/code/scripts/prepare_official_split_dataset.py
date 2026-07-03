import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


CODE_SRC = Path(__file__).resolve().parents[1] / "src"
if str(CODE_SRC) not in sys.path:
    sys.path.insert(0, str(CODE_SRC))

import importlib.util

PREPARE_PATH = CODE_SRC / "02_prepare_dataset.py"
spec = importlib.util.spec_from_file_location("prepare_dataset", PREPARE_PATH)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def normalize_split_label(value):
    label = prepare.unwrap_nested_value(value)
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
    raise ValueError(f"Cannot find official train/test split column. Columns: {df.columns.tolist()}")


def save_npz(path, x, y=None, names=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if y is None:
        np.savez_compressed(path, x=x)
    else:
        np.savez_compressed(path, x=x, y=y, label_names=names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/LSWMD.pkl")
    parser.add_argument("--out", default="data/processed64_onehot_official")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--classes", choices=["defect8", "all9"], default="defect8")
    parser.add_argument("--one-hot-input", action="store_true", default=False)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_path = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading dataset from: {raw_path}")
    df = pd.read_pickle(raw_path).copy()
    split_col = find_official_split_column(df)
    print(f"[INFO] Official split column: {split_col}")
    df["clean_label"] = df["failureType"].apply(prepare.normalize_label)
    df["official_split"] = df[split_col].apply(normalize_split_label)

    class_names = prepare.DEFECT8_CLASSES if args.classes == "defect8" else prepare.ALL9_CLASSES
    label_to_id = {name: i for i, name in enumerate(class_names)}
    id_to_label = {i: name for name, i in label_to_id.items()}

    selected_labels = set(class_names)
    if args.classes == "all9":
        selected_labels.add("none")
    labeled_df = df[df["clean_label"].isin(selected_labels)].copy()
    labeled_df = labeled_df[labeled_df["official_split"].isin(["train", "test"])].copy()

    print("\n========== Selected Classes ==========")
    print(class_names)
    print("\n========== Official Split Counts ==========")
    print(pd.crosstab(labeled_df["clean_label"].map(lambda item: prepare.canonical_label(item, label_to_id)), labeled_df["official_split"]))

    trainval_df = labeled_df[labeled_df["official_split"] == "train"].copy()
    test_df = labeled_df[labeled_df["official_split"] == "test"].copy()
    x_trainval, y_trainval, names_trainval = prepare.build_arrays(trainval_df, label_to_id, id_to_label, args.image_size)
    x_test, y_test, names_test = prepare.build_arrays(test_df, label_to_id, id_to_label, args.image_size)

    x_train, x_val, y_train, y_val, names_train, names_val = train_test_split(
        x_trainval,
        y_trainval,
        names_trainval,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_trainval,
    )

    if args.one_hot_input:
        x_train = prepare.to_one_hot_input(x_train)
        x_val = prepare.to_one_hot_input(x_val)
        x_test = prepare.to_one_hot_input(x_test)

    print("\n========== Split Shapes ==========")
    print(f"Train: {x_train.shape}, {y_train.shape}")
    print(f"Val:   {x_val.shape}, {y_val.shape}")
    print(f"Test:  {x_test.shape}, {y_test.shape}")

    save_npz(out_dir / f"wafer_{args.image_size}_train.npz", x_train, y_train, names_train)
    save_npz(out_dir / f"wafer_{args.image_size}_val.npz", x_val, y_val, names_val)
    save_npz(out_dir / f"wafer_{args.image_size}_test.npz", x_test, y_test, names_test)

    label_info = {
        "classes_mode": args.classes,
        "split_mode": "official",
        "official_split_column": split_col,
        "image_size": args.image_size,
        "class_names": class_names,
        "label_to_id": label_to_id,
        "id_to_label": {str(k): v for k, v in id_to_label.items()},
        "num_classes": len(class_names),
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "val_fraction_from_official_train": args.val_size,
        "input_channels": 3 if args.one_hot_input else 1,
        "one_hot_input": bool(args.one_hot_input),
    }
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_info, f, indent=2, ensure_ascii=False)

    print("\n[INFO] Saved official split files to:")
    print(f"  {out_dir / f'wafer_{args.image_size}_train.npz'}")
    print(f"  {out_dir / f'wafer_{args.image_size}_val.npz'}")
    print(f"  {out_dir / f'wafer_{args.image_size}_test.npz'}")
    print(f"  {out_dir / 'label_map.json'}")


if __name__ == "__main__":
    main()
