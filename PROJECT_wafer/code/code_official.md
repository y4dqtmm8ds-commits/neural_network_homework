# 官方划分实验说明

本文档记录基于 LSWMD 官方 `trianTestLabel` 字段划分的 8 类和 9 类实验。这里的“官方划分”指：

- `trianTestLabel = Training` 的样本作为官方训练来源；
- `trianTestLabel = Test` 的样本作为官方测试集；
- validation 集不是官方单独提供的，而是从官方 Training 样本中按类别 stratified 再划出 15%。

## 1. 代码介绍

 `code/src/02_prepare_dataset.py` 只支持随机 stratified split，不支持官方 `trianTestLabel` 划分。

官方划分数据脚本：
code/scripts/prepare_official_split_dataset.py

8 类官方 W3 配置和入口：
code/configs/dpfeefocal_onehot_64_official.yaml
code/experiments/train_w3_official_split.py

运行脚本：
code/scripts/13_prepare_official_defect8_onehot.ps1
code/scripts/14_generate_oof_cleanlab_official.ps1
code/scripts/15_train_w3_official_split.ps1

除此之外，模型结构、W3 训练逻辑、N1/N4 逻辑仍然使用 `code/src/wafer_train` 中原来的训练代码。

## 2. 官方八类数据划分

生成命令：
powershell -ExecutionPolicy Bypass -File code\scripts\13_prepare_official_defect8_onehot.ps1

输出目录：data/processed64_onehot_official

样本数量：
| Split | Count |
|---|---:|
| Train | 14981 |
| Val | 2644 |
| Test | 7894 |

官方原始 train/test 类别数量：
| Class | Official Train | Official Test |
|---|---:|---:|
| Center | 3462 | 832 |
| Donut | 409 | 146 |
| Edge-Loc | 2417 | 2772 |
| Edge-Ring | 8554 | 1126 |
| Loc | 1620 | 1973 |
| Near-full | 54 | 95 |
| Random | 609 | 257 |
| Scratch | 500 | 693 |

可以看到官方 test 与官方 train 的类别分布差异很大，例如 `Edge-Loc / Loc / Scratch / Near-full` 在测试集中占比更高。

## 3. 官方八类 W3 训练

官方 W3 使用的配置：
code/configs/dpfeefocal_onehot_64_official.yaml

训练前先重新生成官方训练集对应的 OOF Cleanlab，不能复用随机划分的 `logs/cleanlab_oof`，因为 Cleanlab mask 必须和训练集顺序、长度一致。

生成官方 OOF Cleanlab：
powershell -ExecutionPolicy Bypass -File code\scripts\14_generate_oof_cleanlab_official.ps1

输出：logs/cleanlab_oof_official

Cleanlab 发现的问题样本数：67

主要问题对：
| Pair | Count |
|---|---:|
| Loc -> Scratch | 12 |
| Scratch -> Edge-Loc | 7 |
| Scratch -> Loc | 7 |
| Edge-Loc -> Loc | 7 |
| Loc -> Edge-Loc | 5 |

训练官方 W3：
powershell -ExecutionPolicy Bypass -File code\scripts\15_train_w3_official_split.ps1

输出目录：
outputs_code_official/W3_official_split_50

结果：
| Metric | Value |
|---|---:|
| Accuracy | 0.8342 |
| Macro-F1 | 0.8164 |
| Weighted-F1 | 0.8354 |
| Best epoch | 45 |

每类 Recall：
| Class | Recall |
|---|---:|
| Center | 0.9050 |
| Donut | 0.8014 |
| Edge-Loc | 0.9022 |
| Edge-Ring | 0.9094 |
| Loc | 0.6954 |
| Near-full | 0.9263 |
| Random | 0.7626 |
| Scratch | 0.7706 |

## 4. 官方九类数据划分

生成命令：
conda run --no-capture-output -n pytorch2.3.1 python code\scripts\prepare_official_split_dataset.py `
  --raw data\raw\LSWMD.pkl `
  --out data\processed64_onehot_all9_official `
  --image-size 64 `
  --classes all9 `
  --one-hot-input `
  --val-size 0.15 `
  --seed 42

输出目录：data/processed64_onehot_all9_official

样本数量：
| Split | Count |
|---|---:|
| Train | 46201 |
| Val | 8154 |
| Test | 118595 |

官方原始 train/test 类别数量：
| Class | Official Train | Official Test |
|---|---:|---:|
| Center | 3462 | 832 |
| Donut | 409 | 146 |
| Edge-Loc | 2417 | 2772 |
| Edge-Ring | 8554 | 1126 |
| Loc | 1620 | 1973 |
| Near-full | 54 | 95 |
| Random | 609 | 257 |
| Scratch | 500 | 693 |
| Normal | 36730 | 110701 |

官方 9 类测试集里 Normal 数量非常大：
Normal test = 110701
Defect test = 7894

因此 9 类 accuracy 和 weighted-F1 会强烈受 Normal 类影响，macro-F1 更能反映缺陷类性能。

## 5. 官方九类 N1 训练

N1 是直接 9 类 softmax 分类：
DPFEE dual-head
64x64 one-hot
9 类输出：8 类缺陷 + Normal
不使用 Cleanlab

运行命令：
conda run --no-capture-output -n pytorch2.3.1 python -c "import sys; sys.path.insert(0, 'code/src'); from wafer_train.cli import main; main(['--config','code/configs/dpfeefocal_onehot_64_9cls.yaml','--data-dir','data/processed64_onehot_all9_official','--out','outputs_code_official/N1_9cls_w3_no_cleanlab_20','--epochs','20','--device','cuda','--no-balanced-sampler'])"

输出目录：outputs_code_official/N1_9cls_w3_no_cleanlab_20

结果：
| Metric | Value |
|---|---:|
| Accuracy | 0.9646 |
| 9-class Macro-F1 | 0.7275 |
| Defect-only 8-class Macro-F1 | 0.6953 |
| Weighted-F1 | 0.9648 |
| Normal Precision | 0.9853 |
| Normal Recall | 0.9852 |
| Best epoch | 19 |

关键缺陷类 Recall：
| Class | Recall |
|---|---:|
| Donut | 0.7055 |
| Loc | 0.5748 |
| Scratch | 0.4675 |
| Edge-Loc | 0.7302 |

## 6. 官方九类 N4 训练

N4 是两阶段 pipeline：
Stage 1: 训练 Normal vs Defect 二分类 DPFEE
Stage 2: 对判为 Defect 的样本调用官方划分训练出的 8 类 W3

Stage 2 使用的 checkpoint：
outputs_code_official/W3_official_split_50/best_model.pth

运行命令：
conda run --no-capture-output -n pytorch2.3.1 python code\scripts\run_n4_normal_gate_w3.py `
  --data-dir data\processed64_onehot_all9_official `
  --w3-checkpoint outputs_code_official\W3_official_split_50\best_model.pth `
  --out outputs_code_official\N4_9cls_normal_gate_w3_20 `
  --epochs 20 `
  --batch-size 128 `
  --lr 0.0003 `
  --weight-decay 0.0001 `
  --width 48 `
  --dropout 0.25 `
  --label-smoothing 0.03 `
  --device cuda `
  --num-workers 0 `
  --seed 42 `
  --patience 8 `
  --binary-threshold 0.5 `
  --balanced-binary-sampler

输出目录：outputs_code_official/N4_9cls_normal_gate_w3_20

结果：
| Metric | Value |
|---|---:|
| Accuracy | 0.9598 |
| 9-class Macro-F1 | 0.7219 |
| Defect-only 8-class Macro-F1 | 0.6893 |
| Weighted-F1 | 0.9619 |
| Normal Precision | 0.9885 |
| Normal Recall | 0.9770 |
| Best epoch | 19 |

关键缺陷类 Recall：
| Class | Recall |
|---|---:|
| Donut | 0.7877 |
| Loc | 0.6178 |
| Scratch | 0.5873 |
| Edge-Loc | 0.7832 |

N4 二分类门控错误：
| Error Type | Count | Rate |
|---|---:|---:|
| Normal -> Defect | 2542 / 110701 | 2.30% |
| Defect -> Normal | 1263 / 7894 | 16.00% |

## 7. 官方九类汇总

汇总命令：
conda run --no-capture-output -n pytorch2.3.1 python code\src\summarize_experiments.py `
  --summary-9cls `
  --exp-dirs `
    outputs_code_official\N1_9cls_w3_no_cleanlab_20 `
    outputs_code_official\N4_9cls_normal_gate_w3_20 `
  --csv outputs_code_official\summary_N1_N4_official_9cls.csv `
  --md outputs_code_official\summary_N1_N4_official_9cls.md

输出：
outputs_code_official/summary_N1_N4_official_9cls.csv
outputs_code_official/summary_N1_N4_official_9cls.md

对比：
| Experiment | Accuracy | 9-class Macro-F1 | Defect-only Macro-F1 | Normal Recall |
|---|---:|---:|---:|---:|
| N1 | 0.9646 | 0.7275 | 0.6953 | 0.9852 |
| N4 | 0.9598 | 0.7219 | 0.6893 | 0.9770 |

N1 的总体 accuracy 和 macro-F1 略高；N4 对部分缺陷类召回更好，但 Stage 1 将 16.00% 的缺陷判成 Normal，导致最终 macro-F1 被拉低。

## 8. 为什么官方划分下八分类下降很多

之前项目里的主要实验使用的是随机 stratified split。随机划分会让 train/val/test 的类别比例、数据来源和形态分布更接近，因此测试集更像训练集。

官方划分不同，它不是重新随机分层，而是按照数据集原始的 `trianTestLabel` 列划分。这样会带来明显的 domain shift：

1. 官方 train/test 类别分布差异大。

例如 8 类中：

- `Edge-Loc`：train 2417，test 2772；
- `Loc`：train 1620，test 1973；
- `Scratch`：train 500，test 693；
- `Near-full`：train 54，test 95。

这些类别在测试集中相对更难、更重，且有些类测试样本甚至多于训练样本。

2. 官方 test 更接近真实泛化评估。

随机划分中同类样本的形态、批次、尺寸分布更容易混合到 train/test 两侧；官方划分可能把不同批次或不同分布的晶圆放在测试集中，因此模型泛化难度更高。

3. Hard classes 的混淆被放大。

官方划分下 `Loc / Scratch / Edge-Loc` 的混淆更明显。八类 W3 的官方 test 中：

- `Loc recall = 0.6954`
- `Scratch recall = 0.7706`
- `Donut precision = 0.5625`
- `Scratch precision = 0.5836`

这些都会明显拉低 macro-F1。

4. 九类中 Normal 占比过大，会掩盖缺陷类下降。

官方 9 类 test 中 Normal 有 110701 个，而 8 类缺陷总共 7894 个。因此：

- accuracy 很高，主要因为 Normal 很容易分对；
- weighted-F1 很高，受 Normal 支配；
- macro-F1 和 defect-only macro-F1 明显更低，才更能反映缺陷类真实性能。

