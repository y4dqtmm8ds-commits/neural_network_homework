# scripts/generate_cleanlab_oof_probs.py
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset


def load_train_module():
    code_src = Path(__file__).resolve().parents[1] / "src"
    if str(code_src) not in sys.path:
        sys.path.insert(0, str(code_src))
    from wafer_train import engine

    return engine


class ArrayWaferDataset(Dataset):
    def __init__(self, x, y, indices, train_module, augment=False):
        self.x = x[indices].astype(np.float32)
        max_val = self.x.max()
        if max_val > 0:
            self.x = self.x / max_val
        if self.x.ndim == 3:
            self.x = self.x[:, None, :, :]
        self.y = y[indices].astype(np.int64)
        self.indices = np.asarray(indices)
        self.train_module = train_module
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx])
        if self.augment:
            x = self.train_module.random_wafer_augment(x)
        return x, torch.tensor(self.y[idx], dtype=torch.long)


def build_model(train_module, args, num_classes, in_channels):
    if args.model != "dpfee_dual_focal_onehot":
        raise ValueError("This OOF script currently supports --model dpfee_dual_focal_onehot")
    return train_module.build_model(
        model_name="dpfee_dual",
        num_classes=num_classes,
        width=args.width,
        dropout=args.dropout,
        in_channels=in_channels,
        attention="none",
        use_edge_branch=False,
        edge_branch_type="fixed",
    )


@torch.no_grad()
def predict_fold(model, loader, device, train_module, use_amp=True, aux_eval_weight=0.3):
    probs = []
    targets = []
    model.eval()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = train_module.prediction_logits(model(x), aux_eval_weight=aux_eval_weight)
            prob = F.softmax(logits, dim=1)
        probs.append(prob.cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs, axis=0), np.concatenate(targets, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed64_onehot")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--epochs-per-fold", type=int, default=30)
    parser.add_argument("--model", default="dpfee_dual_focal_onehot")
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--one-hot", action="store_true", default=True)
    parser.add_argument("--output-dir", default="logs/cleanlab_oof")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=True)
    args = parser.parse_args()

    train_module = load_train_module()
    train_module.set_seed(args.seed)
    device = train_module.resolve_device(args.device)
    use_amp = train_module.amp_enabled(device, args.amp)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz_path = Path(args.data_dir) / f"wafer_{args.input_size}_train.npz"
    label_info_path = Path(args.data_dir) / "label_map.json"
    label_info = json.load(open(label_info_path, "r", encoding="utf-8"))
    class_names = label_info["class_names"]
    num_classes = int(label_info["num_classes"])
    data = np.load(npz_path, allow_pickle=True)
    x = data["x"]
    y = data["y"].astype(np.int64)
    sample_ids = np.arange(len(y)).astype(str)
    in_channels = x.shape[1] if x.ndim == 4 else 1

    oof_probs = np.zeros((len(y), num_classes), dtype=np.float32)
    fold_rows = []
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    for fold_idx, (train_idx, holdout_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        print(f"[INFO] Fold {fold_idx}/{args.num_folds}: train={len(train_idx)} holdout={len(holdout_idx)}")
        train_set = ArrayWaferDataset(x, y, train_idx, train_module, augment=True)
        holdout_set = ArrayWaferDataset(x, y, holdout_idx, train_module, augment=False)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
        holdout_loader = DataLoader(holdout_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

        model = build_model(train_module, args, num_classes, in_channels).to(device)
        criterion = train_module.build_criterion("ce", label_smoothing=0.03)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_per_fold, eta_min=args.lr * 0.03)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        hard_weights = train_module.build_hard_class_weights(class_names, ["Donut", "Edge-Loc", "Loc", "Scratch"], 2.0, device)
        for epoch in range(1, args.epochs_per_fold + 1):
            loss = train_module.train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler=scaler,
                ema=None,
                use_amp=use_amp,
                loss_name="ce",
                label_smoothing=0.03,
                aux_loss_weight=0.4,
                aux_focal_gamma=3.0,
                hard_sample_weights=hard_weights,
            )
            scheduler.step()
            if epoch == 1 or epoch == args.epochs_per_fold or epoch % 10 == 0:
                print(f"[INFO] fold={fold_idx} epoch={epoch}/{args.epochs_per_fold} loss={loss:.4f}")

        probs, targets = predict_fold(model, holdout_loader, device, train_module, use_amp=use_amp, aux_eval_weight=0.3)
        oof_probs[holdout_idx] = probs
        preds = probs.argmax(axis=1)
        recalls = recall_score(targets, preds, labels=list(range(num_classes)), average=None, zero_division=0)
        fold_metrics = {
            "fold": fold_idx,
            "fold_accuracy": float(accuracy_score(targets, preds)),
            "fold_macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
            "per_class_recall": {class_names[i]: float(recalls[i]) for i in range(num_classes)},
        }
        fold_rows.append(fold_metrics)
        np.save(out / f"fold{fold_idx}_pred_probs.npy", probs)
        torch.save({"model_state_dict": model.state_dict(), "fold": fold_idx, "args": vars(args)}, out / f"fold{fold_idx}_model.pth")
        print(json.dumps(fold_metrics, indent=2, ensure_ascii=False))

    np.save(out / "oof_pred_probs.npy", oof_probs)
    np.save(out / "oof_labels.npy", y)
    np.save(out / "oof_sample_ids.npy", sample_ids)
    with open(out / "oof_image_paths.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "dataset_index", "image_path", "label"])
        writer.writeheader()
        for idx, label in enumerate(y):
            writer.writerow({"sample_id": sample_ids[idx], "dataset_index": idx, "image_path": str(npz_path), "label": int(label)})
    summary = {
        "non_oof_pred_probs": False,
        "num_folds": int(args.num_folds),
        "epochs_per_fold": int(args.epochs_per_fold),
        "fold_metrics": fold_rows,
        "mean_fold_accuracy": float(np.mean([r["fold_accuracy"] for r in fold_rows])),
        "mean_fold_macro_f1": float(np.mean([r["fold_macro_f1"] for r in fold_rows])),
    }
    with open(out / "oof_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
