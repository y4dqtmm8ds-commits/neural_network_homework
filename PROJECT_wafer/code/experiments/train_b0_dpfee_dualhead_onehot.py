from _run_train import run_training


run_training([
    "--data-dir", "data/processed64_onehot",
    "--out", "outputs_code/B0_dpfee_dual_onehot_50",
    "--epochs", "50",
    "--device", "cuda",
    "--model", "dpfee_dual",
    "--dual-head",
    "--width", "48",
    "--dropout", "0.25",
    "--batch-size", "128",
    "--lr", "0.0003",
    "--weight-decay", "0.0001",
    "--loss-type", "ce_ls",
    "--label-smoothing", "0.03",
    "--scheduler", "cosine",
    "--patience", "15",
    "--train-augment",
    "--tta",
    "--ema",
    "--ema-decay", "0.999",
    "--aux-loss-weight", "0.4",
    "--aux-focal-gamma", "3.0",
    "--aux-eval-weight", "0.3",
    "--hard-class-weight", "2.0",
])
