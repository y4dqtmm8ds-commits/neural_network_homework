# data/cleanlab_utils.py
# -*- coding: utf-8 -*-

import argparse
import csv
import inspect
import json
import sys
from pathlib import Path

import numpy as np


CONFUSING_PAIRS = {
    (4, 7),
    (7, 4),
    (4, 2),
    (2, 4),
    (7, 2),
    (2, 7),
    (1, 0),
    (0, 1),
    (2, 3),
    (3, 2),
}

CONFUSING_PAIR_NAMES = [
    ("Normal", "Random"),
    ("Random", "Normal"),
    ("Normal", "Loc"),
    ("Loc", "Normal"),
    ("Normal", "Center"),
    ("Center", "Normal"),
    ("Normal", "Near-full"),
    ("Near-full", "Normal"),
    ("none", "Random"),
    ("Random", "none"),
    ("none", "Loc"),
    ("Loc", "none"),
    ("none", "Center"),
    ("Center", "none"),
    ("none", "Near-full"),
    ("Near-full", "none"),
    ("Loc", "Scratch"),
    ("Scratch", "Loc"),
    ("Loc", "Edge-Loc"),
    ("Edge-Loc", "Loc"),
]


def load_train_module():
    code_src = Path(__file__).resolve().parents[1] / "src"
    if str(code_src) not in sys.path:
        sys.path.insert(0, str(code_src))
    from wafer_train import engine

    return engine


def load_cleanlab_api():
    try:
        from cleanlab.filter import find_label_issues
    except Exception as exc:
        raise RuntimeError(
            "cleanlab is required for label issue detection. Install it in the active env, "
            "for example: conda run -n pytorch2.3.1 python -m pip install cleanlab"
        ) from exc
    return find_label_issues, inspect.signature(find_label_issues)


def infer_pred_probs_from_checkpoint(data_npz, checkpoint_path, batch_size=256, device="auto", use_tta=False, num_workers=0):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    train_module = load_train_module()
    device = train_module.resolve_device(device)
    use_amp = train_module.amp_enabled(device, True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    dataset = train_module.WaferDataset(data_npz)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = train_module.build_model(
        model_name=ckpt_args.get("model", "simple"),
        num_classes=int(checkpoint["num_classes"]),
        width=int(ckpt_args.get("width", 48)),
        dropout=float(ckpt_args.get("dropout", 0.25)),
        vit_dim=int(ckpt_args.get("vit_dim", 128)),
        vit_depth=int(ckpt_args.get("vit_depth", 2)),
        vit_heads=int(ckpt_args.get("vit_heads", 4)),
        in_channels=int(ckpt_args.get("input_channels", dataset.x.shape[1])),
        attention=ckpt_args.get("attention", "none"),
        use_edge_branch=bool(ckpt_args.get("use_edge_branch", False)),
        edge_branch_type=ckpt_args.get("edge_branch_type", "fixed"),
    ).to(device)
    state_dict = checkpoint.get("selected_model_state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("ema_model_state_dict")
    if state_dict is None:
        state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    aux_eval_weight = float(ckpt_args.get("aux_eval_weight", 0.0))

    probs = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                if use_tta:
                    views = [
                        F.softmax(train_module.prediction_logits(model(x_aug), aux_eval_weight), dim=1)
                        for x_aug in train_module.tta_variants(x)
                    ]
                    prob = torch.stack(views, dim=0).mean(dim=0)
                else:
                    prob = F.softmax(train_module.prediction_logits(model(x), aux_eval_weight), dim=1)
            probs.append(prob.cpu().numpy())
    return np.concatenate(probs, axis=0)


def compute_label_quality_scores(labels, pred_probs):
    try:
        from cleanlab.rank import get_label_quality_scores

        return get_label_quality_scores(labels=labels, pred_probs=pred_probs, method="self_confidence")
    except Exception:
        return pred_probs[np.arange(len(labels)), labels]


def load_class_names(label_map_path=None):
    if not label_map_path:
        return None
    with open(label_map_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("class_names")


def build_confusing_pairs(class_names=None):
    pairs = set(CONFUSING_PAIRS)
    if class_names:
        name_to_idx = {name: idx for idx, name in enumerate(class_names)}
        for a, b in CONFUSING_PAIR_NAMES:
            if a in name_to_idx and b in name_to_idx:
                pairs.add((name_to_idx[a], name_to_idx[b]))
    return pairs


def pair_key(original, predicted, class_names=None):
    if class_names and int(original) < len(class_names) and int(predicted) < len(class_names):
        return f"{class_names[int(original)]}->{class_names[int(predicted)]}"
    return f"{int(original)}->{int(predicted)}"


def enforce_min_keep_per_class(keep_mask, labels, issue_order, min_keep_per_class):
    keep_mask = keep_mask.copy()
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        if keep_mask[cls_idx].sum() >= int(min_keep_per_class):
            continue
        cls_issues = [idx for idx in issue_order if labels[idx] == cls and not keep_mask[idx]]
        need = int(min_keep_per_class) - int(keep_mask[cls_idx].sum())
        for idx in cls_issues[-need:]:
            keep_mask[idx] = True
    return keep_mask


def clean_dataset_with_cleanlab(
    labels,
    image_ids=None,
    pred_probs=None,
    out_dir="logs",
    remove_frac=0.02,
    min_keep_per_class=1,
    non_oof_pred_probs=False,
    class_names=None,
):
    find_label_issues, signature = load_cleanlab_api()
    labels = np.asarray(labels).astype(int)
    pred_probs = np.asarray(pred_probs, dtype=np.float64)
    if pred_probs.shape[0] != labels.shape[0]:
        raise ValueError(f"pred_probs rows {pred_probs.shape[0]} != labels {labels.shape[0]}")
    if image_ids is None:
        image_ids = np.arange(len(labels)).astype(str)
    image_ids = np.asarray(image_ids).astype(str)

    kwargs = {}
    if "return_indices_ranked_by" in signature.parameters:
        kwargs["return_indices_ranked_by"] = "self_confidence"
    issue_order = find_label_issues(labels=labels, pred_probs=pred_probs, **kwargs)
    if issue_order.dtype == bool:
        issue_order = np.where(issue_order)[0]
    issue_order = np.asarray(issue_order, dtype=int)

    max_remove = int(np.floor(len(labels) * float(remove_frac)))
    max_remove = max(0, min(max_remove, len(issue_order)))
    remove_indices = issue_order[:max_remove]
    keep_mask = np.ones(len(labels), dtype=bool)
    keep_mask[remove_indices] = False
    keep_mask = enforce_min_keep_per_class(keep_mask, labels, issue_order, min_keep_per_class)

    label_quality = compute_label_quality_scores(labels, pred_probs)
    predicted = pred_probs.argmax(axis=1)
    max_prob = pred_probs.max(axis=1)
    confusing_pairs = build_confusing_pairs(class_names)
    focus_pair_counts = {}
    issue_pair_counts = {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "cleanlab_label_issues.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "sample_id",
                "dataset_index",
                "original_label",
                "predicted_label",
                "self_confidence",
                "label_quality_score",
                "max_pred_prob",
                "image_path",
                "is_confusing_pair",
                "will_remove",
            ],
        )
        writer.writeheader()
        for rank, idx in enumerate(issue_order, start=1):
            pair = (int(labels[idx]), int(predicted[idx]))
            key = pair_key(pair[0], pair[1], class_names)
            issue_pair_counts[key] = issue_pair_counts.get(key, 0) + 1
            if pair in confusing_pairs:
                focus_pair_counts[key] = focus_pair_counts.get(key, 0) + 1
            writer.writerow(
                {
                    "rank": rank,
                    "sample_id": image_ids[idx],
                    "dataset_index": int(idx),
                    "original_label": int(labels[idx]),
                    "predicted_label": int(predicted[idx]),
                    "self_confidence": float(pred_probs[idx, labels[idx]]),
                    "label_quality_score": float(label_quality[idx]),
                    "max_pred_prob": float(max_prob[idx]),
                    "image_path": image_ids[idx],
                    "is_confusing_pair": pair in confusing_pairs,
                    "will_remove": bool(not keep_mask[idx]),
                }
            )

    np.save(out_dir / "cleanlab_keep_mask.npy", keep_mask)
    summary = {
        "num_samples": int(len(labels)),
        "num_issue_candidates": int(len(issue_order)),
        "remove_frac": float(remove_frac),
        "num_removed": int((~keep_mask).sum()),
        "min_keep_per_class": int(min_keep_per_class),
        "non_oof_pred_probs": bool(non_oof_pred_probs),
        "find_label_issues_signature": str(signature),
        "csv_path": str(csv_path),
        "mask_path": str(out_dir / "cleanlab_keep_mask.npy"),
        "class_names": class_names,
        "focus_pair_counts": focus_pair_counts,
        "top_issue_pair_counts": dict(sorted(issue_pair_counts.items(), key=lambda item: item[1], reverse=True)[:30]),
    }
    with open(out_dir / "cleanlab_label_issues.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return keep_mask, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True, help="Training npz with x/y.")
    parser.add_argument("--pred-probs", type=str, default=None, help="Optional .npy pred_probs.")
    parser.add_argument("--cleanlab-pred-probs-path", type=str, default=None)
    parser.add_argument("--cleanlab-labels-path", type=str, default=None)
    parser.add_argument("--cleanlab-sample-ids-path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint for quick non-OOF inference.")
    parser.add_argument("--out", type=str, default="logs")
    parser.add_argument("--remove-frac", type=float, default=0.02)
    parser.add_argument("--min-keep-per-class", type=int, default=1)
    parser.add_argument("--label-map", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    if args.cleanlab_labels_path:
        labels = np.load(args.cleanlab_labels_path).astype(int)
    else:
        labels = data["y"].astype(int)
    if args.cleanlab_sample_ids_path:
        image_ids = np.load(args.cleanlab_sample_ids_path, allow_pickle=True).astype(str)
    else:
        image_ids = data["label_names"].astype(str) if "label_names" in data.files else np.arange(len(labels)).astype(str)
    non_oof = False
    pred_probs_path = args.cleanlab_pred_probs_path or args.pred_probs
    if pred_probs_path:
        pred_probs = np.load(pred_probs_path)
    elif args.checkpoint:
        pred_probs = infer_pred_probs_from_checkpoint(
            args.npz,
            args.checkpoint,
            batch_size=args.batch_size,
            device=args.device,
            use_tta=args.tta,
            num_workers=args.num_workers,
        )
        non_oof = True
    else:
        raise ValueError("Provide either --pred-probs or --checkpoint")

    _, summary = clean_dataset_with_cleanlab(
        labels=labels,
        image_ids=image_ids,
        pred_probs=pred_probs,
        out_dir=args.out,
        remove_frac=args.remove_frac,
        min_keep_per_class=args.min_keep_per_class,
        non_oof_pred_probs=non_oof,
        class_names=load_class_names(args.label_map),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
