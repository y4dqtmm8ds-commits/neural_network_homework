from _run_train import run_training


run_training([
    "--data-dir", "data/processed",
    "--out", "outputs_code/cnn_baseline_32_50",
    "--epochs", "50",
    "--device", "cuda",
    "--model", "simple",
    "--batch-size", "128",
    "--lr", "0.0003",
    "--weight-decay", "0.0001",
    "--loss-type", "ce",
    "--scheduler", "cosine",
    "--patience", "15",
])
