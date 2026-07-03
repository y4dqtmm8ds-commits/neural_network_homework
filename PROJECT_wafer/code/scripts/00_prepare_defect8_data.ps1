$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $ProjectRoot

conda run --no-capture-output -n pytorch2.3.1 python code\src\02_prepare_dataset.py `
  --raw data\raw\LSWMD.pkl `
  --out data\processed `
  --image-size 32 `
  --classes defect8 `
  --seed 42

conda run --no-capture-output -n pytorch2.3.1 python code\src\02_prepare_dataset.py `
  --raw data\raw\LSWMD.pkl `
  --out data\processed64 `
  --image-size 64 `
  --classes defect8 `
  --seed 42

conda run --no-capture-output -n pytorch2.3.1 python code\src\02_prepare_dataset.py `
  --raw data\raw\LSWMD.pkl `
  --out data\processed64_onehot `
  --image-size 64 `
  --classes defect8 `
  --one-hot-input `
  --seed 42
