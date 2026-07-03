$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $ProjectRoot

conda run --no-capture-output -n pytorch2.3.1 python code\scripts\prepare_official_split_dataset.py `
  --raw data\raw\LSWMD.pkl `
  --out data\processed64_onehot_official `
  --image-size 64 `
  --classes defect8 `
  --one-hot-input `
  --val-size 0.15 `
  --seed 42
