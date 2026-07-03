from _run_train import run_training


run_training([
    "--config", "code/configs/dpfeefocal_onehot_64_official.yaml",
    "--out", "outputs_code_official/W3_official_split_50",
    "--epochs", "50",
    "--device", "cuda",
])
