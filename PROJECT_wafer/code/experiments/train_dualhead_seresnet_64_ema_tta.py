from _run_train import run_training


run_training([
    "--data-dir", "data/processed64",
    "--out", "outputs_code/dualhead_seresnet_64_50_ema_tta",
    "--epochs", "50",
    "--device", "cuda",
    "--model", "dual_seresnet",
    "--dual-head",
    "--width", "48",
    "--dropout", "0.25",
    "--batch-size", "128",
    "--lr", "0.0003",
    "--weight-decay", "0.0001",
    "--loss-type", "ce",
    "--scheduler", "cosine",
    "--patience", "15",
    "--aux-loss-weight", "0.4",
    "--aux-eval-weight", "0.3",
    "--train-aug",
    "--ema",
    "--ema-decay", "0.999",
    "--tta",
])
