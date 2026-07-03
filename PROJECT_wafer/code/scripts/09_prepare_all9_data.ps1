$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $ProjectRoot

conda run --no-capture-output -n pytorch2.3.1 python code\src\02_prepare_dataset.py `
  --raw data\raw\LSWMD.pkl `
  --out data\processed64_onehot_all9 `
  --image-size 64 `
  --classes all9 `
  --one-hot-input `
  --seed 42
