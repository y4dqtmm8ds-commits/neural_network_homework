# src/summarize_experiments.py
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path


KEY_CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]
SUMMARY_9CLS_FIELDS = [
    "experiment_name",
    "accuracy",
    "macro_F1_9cls",
    "defect_macro_F1_8cls",
    "Normal_precision",
    "Normal_recall",
    "Donut_recall",
    "Loc_recall",
    "Scratch_recall",
    "Edge_Loc_recall",
    "best_epoch",
    "notes",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metric_value(metrics, key, default=None):
    test_metrics = metrics.get("test_metrics", {})
    return test_metrics.get(key, metrics.get(key, default))


def row_from_metrics(path, root):
    metrics = load_json(path)
    args = metrics.get("args", {})
    recalls = metrics.get("per_class_recall", {})
    human_stats = metrics.get("human_review_applied_stats", {})
    stats_path = path.parent / "human_review_applied_stats.json"
    if stats_path.exists():
        human_stats = load_json(stats_path)
    experiment = path.parent.relative_to(root).as_posix()
    row = {
        "experiment": experiment,
        "experiment_name": experiment,
        "accuracy": metric_value(metrics, "accuracy"),
        "macro_f1": metric_value(metrics, "macro_f1"),
        "weighted_f1": metric_value(metrics, "weighted_f1"),
        "train_time_sec": metrics.get("train_time_sec", ""),
        "final_train_loss": metrics.get("final_train_loss", (metrics.get("history", {}).get("train_loss") or [""])[-1]),
        "final_val_loss": metrics.get("final_val_loss", (metrics.get("history", {}).get("val_loss") or [""])[-1]),
        "loss_type": metrics.get("loss_type", args.get("loss_type", args.get("loss", ""))),
        "loss": args.get("loss", ""),
        "focal_gamma": args.get("focal_gamma", ""),
        "qfl_gamma": metrics.get("qfl_gamma", args.get("qfl_gamma", "")),
        "qfl_quality_min": metrics.get("qfl_quality_min", args.get("qfl_quality_min", "")),
        "qfl_quality_max": metrics.get("qfl_quality_max", args.get("qfl_quality_max", "")),
        "label_smoothing": args.get("label_smoothing", ""),
        "ema_decay": args.get("ema_decay", metrics.get("ema_decay", "")),
        "tta": metrics.get("tta", args.get("tta", "")),
        "use_component_head": metrics.get("use_component_head", args.get("use_component_head", "")),
        "component_loss_weight": metrics.get("component_loss_weight", args.get("component_loss_weight", "")),
        "use_geometric_features": metrics.get("use_geometric_features", args.get("use_geometric_features", "")),
        "use_hardclass_aug": metrics.get("use_hardclass_aug", args.get("use_hardclass_aug", "")),
        "use_scratch_aug": metrics.get("use_scratch_aug", args.get("use_scratch_aug", "")),
        "class_specific_aug": metrics.get("class_specific_aug", args.get("class_specific_aug", "")),
        "use_pseudo_mask": metrics.get("use_pseudo_mask", args.get("use_pseudo_mask", "")),
        "pseudo_mask_loss_weight": metrics.get("pseudo_mask_loss_weight", args.get("pseudo_mask_loss_weight", "")),
        "use_scratchness_head": metrics.get("use_scratchness_head", args.get("use_scratchness_head", "")),
        "scratchness_loss_weight": metrics.get("scratchness_loss_weight", args.get("scratchness_loss_weight", "")),
        "scratchness_start_epoch": metrics.get("scratchness_start_epoch", args.get("scratchness_start_epoch", "")),
        "use_teacher_distillation": metrics.get("use_teacher_distillation", args.get("use_teacher_distillation", "")),
        "distill_loss_weight": metrics.get("distill_loss_weight", args.get("distill_loss_weight", "")),
        "distill_temperature": metrics.get("distill_temperature", args.get("distill_temperature", "")),
        "use_capsule_head": metrics.get("use_capsule_head", args.get("use_capsule_head", "")),
        "capsule_fusion_mode": metrics.get("capsule_fusion_mode", args.get("capsule_fusion_mode", "")),
        "use_human_review": metrics.get("use_human_review", args.get("use_human_review", "")),
        "applied_human_count": human_stats.get("applied_sample_count", ""),
        "applied_sample_count": human_stats.get("applied_sample_count", ""),
        "keep_count": human_stats.get("keep_count", ""),
        "ambiguous_count": human_stats.get("ambiguous_count", ""),
        "relabel_count": human_stats.get("relabel_count", ""),
        "remove_count": human_stats.get("remove_count", ""),
        "ambiguous_weight": human_stats.get("ambiguous_weight", metrics.get("ambiguous_weight", args.get("ambiguous_weight", ""))),
        "relabel_weight": human_stats.get("relabel_weight", metrics.get("relabel_weight", args.get("relabel_weight", ""))),
        "aux_loss_weight": args.get("aux_loss_weight", ""),
        "aux_eval_weight": args.get("aux_eval_weight", ""),
        "weight_decay": args.get("weight_decay", ""),
        "attention": metrics.get("attention", args.get("attention", "none")),
        "use_edge_branch": metrics.get("use_edge_branch", args.get("use_edge_branch", False)),
        "edge_branch_type": metrics.get("edge_branch_type", args.get("edge_branch_type", "")),
        "use_confusion_margin": metrics.get("use_confusion_margin", args.get("use_confusion_margin", False)),
        "confusion_margin": metrics.get("confusion_margin", args.get("confusion_margin", "")),
        "confusion_lambda": metrics.get("confusion_lambda", args.get("confusion_lambda", "")),
        "cleanlab_weight_normal": metrics.get("cleanlab_weight_normal", args.get("cleanlab_weight_normal", "")),
        "cleanlab_weight_confusing": metrics.get("cleanlab_weight_confusing", args.get("cleanlab_weight_confusing", "")),
        "cleanlab_weight_strong": metrics.get("cleanlab_weight_strong", args.get("cleanlab_weight_strong", "")),
        "mask_entropy": metrics.get("mask_entropy", ""),
        "mask_activation_mean": metrics.get("mask_activation_mean", ""),
        "best_epoch": metrics.get("best_epoch", ""),
        "output_dir": str(path.parent),
        "model": args.get("model", metrics.get("model", "")),
        "cleanlab_mode": args.get("cleanlab_mode", metrics.get("cleanlab_mode", "")) if args.get("use_clean_labels", metrics.get("use_clean_labels", False)) else "",
        "pseudo_mode": args.get("pseudo_mode", metrics.get("pseudo_mode", "")),
        "ensemble_type": metrics.get(
            "ensemble_type",
            "class_aware" if metrics.get("class_aware") else "weighted" if metrics.get("weighted_ensemble") else "simple_avg" if metrics.get("ensemble_size") else "",
        ),
        "seed": args.get("seed", metrics.get("seed", "")),
        "notes": "",
    }
    for name in KEY_CLASSES:
        row[f"{name}_recall"] = recalls.get(name, "")
    return row


def row_from_metrics_9cls(path, root):
    metrics = load_json(path)
    args = metrics.get("args", {})
    report = metrics.get("classification_report", {})
    recalls = metrics.get("per_class_recall", {})
    class_names = metrics.get("class_names", [])
    normal_name = metrics.get("normal_class_name")
    if not normal_name:
        for candidate in ("Normal", "none"):
            if candidate in class_names:
                normal_name = candidate
                break
    experiment = path.parent.relative_to(root).as_posix()
    test_metrics = metrics.get("test_metrics", {})
    normal_report = report.get(normal_name or "Normal", {})
    row = {
        "experiment_name": experiment,
        "accuracy": metric_value(metrics, "accuracy"),
        "macro_F1_9cls": metric_value(metrics, "macro_f1"),
        "defect_macro_F1_8cls": test_metrics.get("defect_macro_f1", metrics.get("defect_macro_f1", "")),
        "Normal_precision": normal_report.get("precision", ""),
        "Normal_recall": normal_report.get("recall", recalls.get(normal_name or "Normal", "")),
        "Donut_recall": recalls.get("Donut", ""),
        "Loc_recall": recalls.get("Loc", ""),
        "Scratch_recall": recalls.get("Scratch", ""),
        "Edge_Loc_recall": recalls.get("Edge-Loc", ""),
        "best_epoch": metrics.get("best_epoch", ""),
        "notes": (
            "two_stage" if metrics.get("two_stage_normal_defect") else
            f"cleanlab={args.get('cleanlab_mode')}" if args.get("use_clean_labels") else
            "direct_9cls"
        ),
    }
    return row


def row_from_stacking(path):
    metrics = load_json(path)
    recalls = metrics.get("per_class_recall", {})
    row = {
        "experiment": path.parent.name,
        "experiment_name": path.parent.name,
        "output_dir": str(path.parent),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "train_time_sec": "",
        "final_train_loss": "",
        "final_val_loss": "",
        "loss_type": "",
        "loss": "",
        "focal_gamma": "",
        "qfl_gamma": "",
        "qfl_quality_min": "",
        "qfl_quality_max": "",
        "label_smoothing": "",
        "ema_decay": "",
        "tta": "",
        "use_component_head": "",
        "component_loss_weight": "",
        "use_geometric_features": "",
        "use_hardclass_aug": "",
        "use_scratch_aug": "",
        "class_specific_aug": "",
        "use_pseudo_mask": "",
        "pseudo_mask_loss_weight": "",
        "use_scratchness_head": "",
        "scratchness_loss_weight": "",
        "scratchness_start_epoch": "",
        "use_teacher_distillation": "",
        "distill_loss_weight": "",
        "distill_temperature": "",
        "use_capsule_head": "",
        "capsule_fusion_mode": "",
        "use_human_review": "",
        "applied_human_count": "",
        "applied_sample_count": "",
        "keep_count": "",
        "ambiguous_count": "",
        "relabel_count": "",
        "remove_count": "",
        "ambiguous_weight": "",
        "relabel_weight": "",
        "aux_loss_weight": "",
        "aux_eval_weight": "",
        "weight_decay": "",
        "attention": "",
        "use_edge_branch": "",
        "edge_branch_type": "",
        "use_confusion_margin": "",
        "confusion_margin": "",
        "confusion_lambda": "",
        "best_epoch": "",
        "mask_entropy": "",
        "mask_activation_mean": "",
        "model": "stacking",
        "cleanlab_mode": "",
        "pseudo_mode": "",
        "ensemble_type": metrics.get("ensemble_type", "stacking"),
        "seed": "",
        "notes": f"stacking_mode={metrics.get('stacking_mode', '')}",
    }
    for name in KEY_CLASSES:
        row[f"{name}_recall"] = recalls.get(name, "")
    return row


def write_markdown(rows, path, fieldnames):
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            values = []
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            f.write("| " + " | ".join(values) + " |\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs64")
    parser.add_argument("--extra-root", nargs="*", default=[])
    parser.add_argument("--exp_dirs", "--exp-dirs", nargs="*", default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="*/metrics.json")
    parser.add_argument("--exp50", action="store_true")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--md", type=str, default=None)
    parser.add_argument("--summary-9cls", action="store_true")
    args = parser.parse_args()

    if args.exp_dirs:
        rows = []
        for exp_dir in args.exp_dirs:
            metrics_path = Path(exp_dir) / "metrics.json"
            if metrics_path.exists():
                if args.summary_9cls:
                    rows.append(row_from_metrics_9cls(metrics_path, Path(exp_dir).parent))
                else:
                    rows.append(row_from_metrics(metrics_path, Path(exp_dir).parent))
        if not rows:
            raise RuntimeError("No metrics.json found in --exp_dirs")
    else:
        root = Path(args.root)
        pattern = "exp50_*/metrics.json" if args.exp50 else args.pattern
        paths = sorted(root.glob(pattern))
        if not paths:
            raise RuntimeError(f"No metrics.json files found under {root}")
        rows = [row_from_metrics_9cls(path, root) if args.summary_9cls else row_from_metrics(path, root) for path in paths]
        for extra_root in args.extra_root:
            extra_root = Path(extra_root)
            extra_pattern = "exp50_*/metrics.json" if args.exp50 else args.pattern
            for path in sorted(extra_root.glob(extra_pattern)):
                rows.append(row_from_metrics_9cls(path, extra_root) if args.summary_9cls else row_from_metrics(path, extra_root))
            for path in sorted(extra_root.glob("exp50_*/stacking_metrics.json" if args.exp50 else "*/stacking_metrics.json")):
                rows.append(row_from_stacking(path))
        rows.sort(key=lambda r: (r.get("accuracy") or 0.0, r.get("macro_f1") or 0.0), reverse=True)

    if rows and not args.summary_9cls:
        baseline = rows[0]
        for row in rows:
            for key in ["accuracy", "macro_f1", "Loc_recall", "Scratch_recall", "Edge-Loc_recall", "Donut_recall"]:
                try:
                    row[f"delta_{key}"] = float(row.get(key) or 0.0) - float(baseline.get(key) or 0.0)
                except ValueError:
                    row[f"delta_{key}"] = ""

    fieldnames = SUMMARY_9CLS_FIELDS if args.summary_9cls else [
        "experiment",
        "experiment_name",
        "output_dir",
        "model",
        "cleanlab_mode",
        "pseudo_mode",
        "ensemble_type",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "delta_accuracy",
        "delta_macro_f1",
        "Donut_recall",
        "Edge-Loc_recall",
        "Loc_recall",
        "Scratch_recall",
        "delta_Donut_recall",
        "delta_Edge-Loc_recall",
        "delta_Loc_recall",
        "delta_Scratch_recall",
        "Center_recall",
        "Edge-Ring_recall",
        "Near-full_recall",
        "Random_recall",
        "train_time_sec",
        "final_train_loss",
        "final_val_loss",
        "loss_type",
        "loss",
        "focal_gamma",
        "qfl_gamma",
        "qfl_quality_min",
        "qfl_quality_max",
        "label_smoothing",
        "ema_decay",
        "tta",
        "use_human_review",
        "applied_human_count",
        "use_component_head",
        "component_loss_weight",
        "use_geometric_features",
        "use_hardclass_aug",
        "use_scratch_aug",
        "class_specific_aug",
        "use_pseudo_mask",
        "pseudo_mask_loss_weight",
        "use_scratchness_head",
        "scratchness_loss_weight",
        "scratchness_start_epoch",
        "use_teacher_distillation",
        "distill_loss_weight",
        "distill_temperature",
        "use_capsule_head",
        "capsule_fusion_mode",
        "applied_sample_count",
        "keep_count",
        "ambiguous_count",
        "relabel_count",
        "remove_count",
        "ambiguous_weight",
        "relabel_weight",
        "aux_loss_weight",
        "aux_eval_weight",
        "weight_decay",
        "cleanlab_weight_normal",
        "cleanlab_weight_confusing",
        "cleanlab_weight_strong",
        "attention",
        "use_edge_branch",
        "edge_branch_type",
        "use_confusion_margin",
        "confusion_margin",
        "confusion_lambda",
        "mask_entropy",
        "mask_activation_mean",
        "best_epoch",
        "seed",
        "notes",
    ]

    csv_path = Path(args.output) if args.output else Path(args.csv) if args.csv else Path(args.root) / "summary_ablation.csv"
    md_path = Path(args.md) if args.md else csv_path.with_suffix(".md")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, md_path, fieldnames)

    print(f"[INFO] Wrote: {csv_path}")
    print(f"[INFO] Wrote: {md_path}")
    acc_key = "accuracy"
    macro_key = "macro_F1_9cls" if args.summary_9cls else "macro_f1"
    name_key = "experiment_name" if args.summary_9cls else "experiment"
    best_acc = max(rows, key=lambda r: r.get(acc_key) or 0.0)
    best_macro = max(rows, key=lambda r: r.get(macro_key) or 0.0)
    print(f"[INFO] Best accuracy: {best_acc[name_key]} = {best_acc[acc_key]:.4f}")
    print(f"[INFO] Best macro-F1: {best_macro[name_key]} = {best_macro[macro_key]:.4f}")


if __name__ == "__main__":
    main()
