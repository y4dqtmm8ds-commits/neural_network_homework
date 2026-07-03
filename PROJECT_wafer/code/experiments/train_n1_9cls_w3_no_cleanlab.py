from _run_train import run_training


run_training([
    "--config", "code/configs/dpfeefocal_onehot_64_9cls.yaml",
    "--out", "outputs_code_9cls/N1_9cls_w3_no_cleanlab",
    "--epochs", "20",
    "--device", "cuda",
    "--no-balanced-sampler",
])
