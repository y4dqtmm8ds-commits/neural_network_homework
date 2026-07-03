$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $ProjectRoot

conda run --no-capture-output -n pytorch2.3.1 python -u code\scripts\generate_cleanlab_oof_probs.py `
  --data-dir data\processed64_onehot_official `
  --output-dir logs\cleanlab_oof_official `
  --input-size 64 `
  --num-folds 5 `
  --epochs-per-fold 20 `
  --batch-size 128 `
  --device cuda `
  --amp `
  --num-workers 0

conda run --no-capture-output -n pytorch2.3.1 python code\data\cleanlab_utils.py `
  --npz data\processed64_onehot_official\wafer_64_train.npz `
  --cleanlab-pred-probs-path logs\cleanlab_oof_official\oof_pred_probs.npy `
  --cleanlab-labels-path logs\cleanlab_oof_official\oof_labels.npy `
  --cleanlab-sample-ids-path logs\cleanlab_oof_official\oof_sample_ids.npy `
  --label-map data\processed64_onehot_official\label_map.json `
  --out logs\cleanlab_oof_official `
  --remove-frac 0.02 `
  --min-keep-per-class 20
