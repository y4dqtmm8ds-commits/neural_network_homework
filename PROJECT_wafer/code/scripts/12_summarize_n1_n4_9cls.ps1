$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $ProjectRoot

conda run --no-capture-output -n pytorch2.3.1 python code\src\summarize_experiments.py `
  --summary-9cls `
  --exp-dirs `
    outputs_code_9cls\N1_9cls_w3_no_cleanlab `
    outputs_code_9cls\N4_9cls_normal_gate_w3 `
  --csv outputs_code_9cls\summary_N1_N4_9cls.csv `
  --md outputs_code_9cls\summary_N1_N4_9cls.md
