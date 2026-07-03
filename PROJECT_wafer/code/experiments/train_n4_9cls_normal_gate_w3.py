import runpy
import sys
from pathlib import Path


args = [
    "--data-dir", "data/processed64_onehot_all9",
    "--w3-checkpoint", "outputs_code/W3_oof_cleanlab_caw_50/best_model.pth",
    "--out", "outputs_code_9cls/N4_9cls_normal_gate_w3",
    "--epochs", "20",
    "--batch-size", "128",
    "--lr", "0.0003",
    "--weight-decay", "0.0001",
    "--width", "48",
    "--dropout", "0.25",
    "--label-smoothing", "0.03",
    "--device", "cuda",
    "--num-workers", "0",
    "--seed", "42",
    "--patience", "8",
    "--binary-threshold", "0.5",
]

script = Path("code/scripts/run_n4_normal_gate_w3.py")
sys.argv = [str(script), *args]
runpy.run_path(str(script), run_name="__main__")
