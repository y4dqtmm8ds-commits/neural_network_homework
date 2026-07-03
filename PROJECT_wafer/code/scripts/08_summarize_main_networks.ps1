$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $ProjectRoot

conda run --no-capture-output -n pytorch2.3.1 python code\src\summarize_experiments.py `
  --exp-dirs `
    outputs_code\cnn_baseline_32_50 `
    outputs_code\seresnet_32_50_ema_tta `
    outputs_code\seresnet_64_50_ema_tta `
    outputs_code\dualhead_seresnet_64_50_ema_tta `
    outputs_code\B0_dpfee_dual_onehot_50 `
    outputs_code\W3_oof_cleanlab_caw_50 `
  --csv outputs_code\main_network_summary.csv `
  --md outputs_code\main_network_summary.md
