$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $ProjectRoot

if (!(Test-Path logs\cleanlab_oof\cleanlab_keep_mask.npy) -or !(Test-Path logs\cleanlab_oof\cleanlab_label_issues.csv)) {
  Write-Host "[WARN] logs\cleanlab_oof cleanlab files not found. Run code\scripts\06_generate_oof_cleanlab_for_w3.ps1 first."
}

conda run --no-capture-output -n pytorch2.3.1 python -u code\experiments\train_w3_oof_cleanlab.py
