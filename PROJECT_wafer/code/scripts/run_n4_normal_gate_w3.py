# -*- coding: utf-8 -*-
import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler


def load_train_module():
    path = Path("code/src/wafer_train/engine.py")
    spec = importlib.util.spec_from_file_location("wafer_train_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BinaryNormalDefectDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, normal_idx):
        self.base_dataset = base_dataset
        self.normal_idx = int(normal_idx)
        self.y = (np.asarray(base_dataset.y) != self.normal_idx).astype(np.int64)
        self.x = base_dataset.x

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        x = item[0]
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y


def make_binary_loader(dataset, batch_size, shuffle, num_workers, pin_memory, balanced=False):
    sampler = None
    if balanced:
        counts = np.bincount(dataset.y, minlength=2).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        sample_weights = 1.0 / counts[dataset.y]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def run_binary_epoch(model, loader, optimizer, device, use_amp, label_smoothing):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total_loss = 0.0
    total_num = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=float(label_smoothing))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu()) * int(y.numel())
        total_num += int(y.numel())
    return total_loss / max(total_num, 1)


@torch.no_grad()
def evaluate_binary(model, loader, device, use_amp):
    model.eval()
    all_true = []
    all_pred = []
    all_probs = []
    total_loss = 0.0
    total_num = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        all_true.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())
        total_loss += float(loss.detach().cpu()) * int(y.numel())
        total_num += int(y.numel())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    probs = np.concatenate(all_probs)
    return {
        "loss": total_loss / max(total_num, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }, y_true, y_pred, probs


def load_w3_model(train_module, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    class_names = checkpoint.get("class_names")
    num_classes = int(checkpoint.get("num_classes", len(class_names)))
    image_size = int(ckpt_args.get("image_size") or checkpoint.get("image_size", 64))
    model = train_module.build_model(
        model_name=ckpt_args.get("model", "dpfee_dual"),
        num_classes=num_classes,
        width=int(ckpt_args.get("width", 48)),
        dropout=float(ckpt_args.get("dropout", 0.25)),
        vit_dim=int(ckpt_args.get("vit_dim", 128)),
        vit_depth=int(ckpt_args.get("vit_depth", 2)),
        vit_heads=int(ckpt_args.get("vit_heads", 4)),
        vit_patch_size=int(ckpt_args.get("vit_patch_size", 8)),
        in_channels=int(ckpt_args.get("input_channels", 3)),
        attention=ckpt_args.get("attention", "none"),
        use_edge_branch=bool(ckpt_args.get("use_edge_branch", False)),
        edge_branch_type=ckpt_args.get("edge_branch_type", "fixed"),
        image_size=image_size,
        use_unet_structure_branch=bool(ckpt_args.get("use_unet_structure_branch", False)),
        freeze_dpfee_backbone=bool(ckpt_args.get("freeze_dpfee_backbone", False)),
        unet_entropy_weight=float(ckpt_args.get("unet_entropy_weight", 1.0)),
        use_component_head=bool(ckpt_args.get("use_component_head", False)),
        component_dim=len(checkpoint.get("component_names", class_names)),
        use_geometric_features=bool(ckpt_args.get("use_geometric_features", False)),
        geo_feature_dim=int(ckpt_args.get("geo_feature_dim", 18)),
        geo_mlp_hidden=int(ckpt_args.get("geo_mlp_hidden", 64)),
        geo_dropout=float(ckpt_args.get("geo_dropout", 0.1)),
        use_scratchness_head=bool(ckpt_args.get("use_scratchness_head", False)),
        use_capsule_head=bool(ckpt_args.get("use_capsule_head", False)),
        capsule_hard_class_count=len(ckpt_args.get("capsule_hard_classes", ["Loc", "Scratch", "Edge-Loc", "Edge-Ring"])),
        capsule_dim=int(ckpt_args.get("capsule_dim", 8)),
        capsule_routing_iters=int(ckpt_args.get("capsule_routing_iters", 3)),
        dpfee_feature_map_size=int(ckpt_args.get("dpfee_feature_map_size", 0)),
    ).to(device)
    state = (
        checkpoint.get("selected_model_state_dict")
        or checkpoint.get("ema_model_state_dict")
        or checkpoint.get("model_state_dict")
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, checkpoint, ckpt_args, class_names


@torch.no_grad()
def predict_w3_for_defects(train_module, model, x, device, use_amp, aux_eval_weight, use_tta):
    x = x.to(device, non_blocking=True)
    with torch.cuda.amp.autocast(enabled=use_amp):
        if use_tta:
            logits = torch.stack(
                [
                    train_module.prediction_logits(model(x_aug), aux_eval_weight)
                    for x_aug in train_module.tta_variants(x)
                ],
                dim=0,
            ).mean(dim=0)
        else:
            logits = train_module.prediction_logits(model(x), aux_eval_weight)
    return logits.argmax(dim=1).detach().cpu().numpy()


@torch.no_grad()
def evaluate_n4_pipeline(
    train_module,
    binary_model,
    w3_model,
    loader,
    device,
    use_amp,
    normal_idx,
    w3_to_9,
    w3_aux_eval_weight,
    binary_threshold,
    w3_tta,
):
    binary_model.eval()
    w3_model.eval()
    y_true_all = []
    y_pred_all = []
    binary_true_all = []
    binary_pred_all = []
    for x, y in loader:
        x_device = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            binary_logits = binary_model(x_device)
            binary_probs = torch.softmax(binary_logits, dim=1)
        defect_prob = binary_probs[:, 1]
        binary_pred = (defect_prob >= float(binary_threshold)).long().detach().cpu().numpy()
        y_np = y.numpy().astype(np.int64)
        final_pred = np.full_like(y_np, fill_value=int(normal_idx))
        defect_rows = np.where(binary_pred == 1)[0]
        if len(defect_rows) > 0:
            w3_local_pred = predict_w3_for_defects(
                train_module,
                w3_model,
                x[defect_rows],
                device,
                use_amp,
                w3_aux_eval_weight,
                w3_tta,
            )
            final_pred[defect_rows] = np.asarray([w3_to_9[int(item)] for item in w3_local_pred], dtype=np.int64)
        y_true_all.append(y_np)
        y_pred_all.append(final_pred)
        binary_true_all.append((y_np != int(normal_idx)).astype(np.int64))
        binary_pred_all.append(binary_pred.astype(np.int64))
    return (
        np.concatenate(y_true_all),
        np.concatenate(y_pred_all),
        np.concatenate(binary_true_all),
        np.concatenate(binary_pred_all),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed64_onehot_all9")
    parser.add_argument("--w3-checkpoint", default="outputs64/exp50_W3_oof_cleanlab_caw_n05_c07_s02/best_model.pth")
    parser.add_argument("--out", default="outputs_code_9cls/N4_9cls_normal_gate_w3")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument("--balanced-binary-sampler", action="store_true", default=False)
    parser.add_argument("--w3-tta", action="store_true", default=False)
    args = parser.parse_args()

    train_module = load_train_module()
    train_module.set_seed(args.seed)
    device = train_module.resolve_device(args.device)
    use_amp = train_module.amp_enabled(device, args.amp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names, num_classes, image_size = train_module.load_label_info(args.data_dir)
    if num_classes != 9:
        raise ValueError(f"N4 expects 9-class data, got num_classes={num_classes} from {args.data_dir}")
    normal_idx = train_module.find_normal_class_idx(class_names)
    if normal_idx is None:
        raise ValueError(f"Cannot find Normal/none class in {class_names}")

    train_set_9 = train_module.WaferDataset(
        Path(args.data_dir) / f"wafer_{image_size}_train.npz",
        augment=True,
        class_names=class_names,
    )
    val_set_9 = train_module.WaferDataset(Path(args.data_dir) / f"wafer_{image_size}_val.npz")
    test_set_9 = train_module.WaferDataset(Path(args.data_dir) / f"wafer_{image_size}_test.npz")
    train_binary = BinaryNormalDefectDataset(train_set_9, normal_idx)
    val_binary = BinaryNormalDefectDataset(val_set_9, normal_idx)

    pin_memory = device.type == "cuda"
    train_loader = make_binary_loader(
        train_binary,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        balanced=args.balanced_binary_sampler,
    )
    val_loader = make_binary_loader(
        val_binary,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        balanced=False,
    )
    test_loader_9 = DataLoader(
        test_set_9,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    binary_model = train_module.build_model(
        model_name="dpfee",
        num_classes=2,
        width=args.width,
        dropout=args.dropout,
        in_channels=int(train_set_9.x.shape[1]),
        image_size=image_size,
    ).to(device)
    optimizer = torch.optim.AdamW(binary_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
    }
    best_val_macro_f1 = -1.0
    best_epoch = 0
    stale = 0
    start = time.perf_counter()
    best_path = out_dir / "best_binary_model.pth"
    print(f"[INFO] N4 Stage 1 binary training on {device}")
    print(f"[INFO] 9-class names: {class_names}; Normal idx={normal_idx}")
    print(f"[INFO] Binary train counts [Normal, Defect]: {np.bincount(train_binary.y, minlength=2).tolist()}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_binary_epoch(
            binary_model,
            train_loader,
            optimizer,
            device,
            use_amp,
            args.label_smoothing,
        )
        scheduler.step()
        val_metrics, _, _, _ = evaluate_binary(binary_model, val_loader, device, use_amp)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["val_macro_f1"].append(float(val_metrics["macro_f1"]))
        history["val_weighted_f1"].append(float(val_metrics["weighted_f1"]))
        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": binary_model.state_dict(),
                    "class_names": ["Normal", "Defect"],
                    "source_9cls_class_names": class_names,
                    "normal_class_idx": normal_idx,
                    "image_size": image_size,
                    "args": vars(args),
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_epoch": best_epoch,
                },
                best_path,
            )
        else:
            stale += 1
        if args.patience > 0 and stale >= args.patience:
            print(f"[INFO] Early stopping at epoch {epoch}.")
            break

    train_time_sec = time.perf_counter() - start
    torch.save(
        {
            "model_state_dict": binary_model.state_dict(),
            "class_names": ["Normal", "Defect"],
            "source_9cls_class_names": class_names,
            "normal_class_idx": normal_idx,
            "image_size": image_size,
            "args": vars(args),
        },
        out_dir / "last_binary_model.pth",
    )
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    binary_model.load_state_dict(checkpoint["model_state_dict"])

    w3_model, w3_checkpoint, w3_args, w3_class_names = load_w3_model(train_module, args.w3_checkpoint, device)
    w3_to_9 = []
    for name in w3_class_names:
        if name not in class_names:
            raise ValueError(f"W3 class {name!r} is not in 9-class label_map: {class_names}")
        w3_to_9.append(class_names.index(name))
    w3_aux_eval_weight = float(w3_args.get("aux_eval_weight", 0.0))
    print(f"[INFO] N4 Stage 2 W3 checkpoint: {args.w3_checkpoint}")
    print(f"[INFO] W3 class mapping to 9-class ids: {dict(zip(w3_class_names, w3_to_9))}")

    y_true, y_pred, binary_true, binary_pred = evaluate_n4_pipeline(
        train_module=train_module,
        binary_model=binary_model,
        w3_model=w3_model,
        loader=test_loader_9,
        device=device,
        use_amp=use_amp,
        normal_idx=normal_idx,
        w3_to_9=w3_to_9,
        w3_aux_eval_weight=w3_aux_eval_weight,
        binary_threshold=args.binary_threshold,
        w3_tta=args.w3_tta,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "defect_macro_f1": float(train_module.defect_macro_f1_score(y_true, y_pred, class_names)),
    }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    binary_cm = confusion_matrix(binary_true, binary_pred, labels=[0, 1])
    per_class_recall = train_module.plot_per_class_recall(
        y_true,
        y_pred,
        class_names,
        out_dir / "per_class_recall.png",
    )
    train_module.plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")
    train_module.plot_confusion_matrix(binary_cm, ["Normal", "Defect"], out_dir / "binary_confusion_matrix.png")
    train_module.plot_training_curves(history, out_dir / "training_curves.png")

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    per_class_metrics = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(num_classes)
    }
    with open(out_dir / "per_class_metrics.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        for name, values in per_class_metrics.items():
            writer.writerow({"class": name, **values})
    with open(out_dir / "train_log.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "val_weighted_f1"],
        )
        writer.writeheader()
        for idx in range(len(history["train_loss"])):
            writer.writerow({
                "epoch": idx + 1,
                "train_loss": history["train_loss"][idx],
                "val_loss": history["val_loss"][idx],
                "val_accuracy": history["val_accuracy"][idx],
                "val_macro_f1": history["val_macro_f1"][idx],
                "val_weighted_f1": history["val_weighted_f1"][idx],
            })

    normal_as_defect = int(binary_cm[0, 1])
    defect_as_normal = int(binary_cm[1, 0])
    results = {
        "test_metrics": metrics,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "class_names": class_names,
        "normal_class_idx": int(normal_idx),
        "normal_class_name": class_names[int(normal_idx)],
        "defect_class_names": [name for i, name in enumerate(class_names) if i != int(normal_idx)],
        "per_class_recall": {class_names[i]: float(per_class_recall[i]) for i in range(num_classes)},
        "per_class_metrics": per_class_metrics,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "binary_confusion_matrix": binary_cm.tolist(),
        "normal_as_defect_count": normal_as_defect,
        "defect_as_normal_count": defect_as_normal,
        "normal_as_defect_rate": float(normal_as_defect / max(binary_cm[0].sum(), 1)),
        "defect_as_normal_rate": float(defect_as_normal / max(binary_cm[1].sum(), 1)),
        "best_val_macro_f1": float(best_val_macro_f1),
        "best_epoch": int(best_epoch),
        "history": history,
        "train_time_sec": float(train_time_sec),
        "epochs_ran": len(history["train_loss"]),
        "pipeline": "N4_normal_gate_w3",
        "stage1": "binary Normal-vs-Defect DPFEE",
        "stage2": "existing 8-class W3 checkpoint",
        "w3_checkpoint": args.w3_checkpoint,
        "w3_class_names": w3_class_names,
        "w3_to_9_class_id": w3_to_9,
        "binary_threshold": float(args.binary_threshold),
        "w3_tta": bool(args.w3_tta),
        "args": vars(args),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("\n========== N4 Test Metrics ==========")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"macro_f1_9cls: {metrics['macro_f1']:.4f}")
    print(f"defect_macro_f1_8cls: {metrics['defect_macro_f1']:.4f}")
    print(f"Normal->Defect: {normal_as_defect}/{int(binary_cm[0].sum())} ({results['normal_as_defect_rate']:.4f})")
    print(f"Defect->Normal: {defect_as_normal}/{int(binary_cm[1].sum())} ({results['defect_as_normal_rate']:.4f})")
    print(f"[INFO] Saved N4 outputs to: {out_dir}")


if __name__ == "__main__":
    main()
