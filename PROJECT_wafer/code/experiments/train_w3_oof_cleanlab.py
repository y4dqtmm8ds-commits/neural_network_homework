from _run_train import run_training


run_training([
    "--config", "code/configs/dpfeefocal_onehot_64.yaml",
    "--out", "outputs_code/W3_oof_cleanlab_caw_50",
    "--epochs", "50",
    "--device", "cuda",
])
