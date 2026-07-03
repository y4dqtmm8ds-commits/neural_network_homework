$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $ProjectRoot

conda run --no-capture-output -n pytorch2.3.1 python -u code\experiments\train_b0_dpfee_dualhead_onehot.py
