from _run_train import run_training


run_training([
    "--data-dir", "data/processed",
    "--out", "outputs_code/seresnet_32_50_ema_tta",
    "--epochs", "50",
    "--device", "cuda",
    "--model", "seresnet",
    "--width", "48",
    "--dropout", "0.25",
    "--batch-size", "256",
    "--lr", "0.0003",
    "--weight-decay", "0.0001",
    "--loss-type", "ce",
    "--scheduler", "cosine",
    "--patience", "15",
    "--train-aug",
    "--ema",
    "--ema-decay", "0.999",
    "--tta",
])
