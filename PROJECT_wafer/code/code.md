# 代码整理与实验说明

本目录用于汇总 8 类晶圆缺陷分类主线实验代码。

每个网络实验都有一个短小的入口文件，放在：code/experiments/

这样阅读时可以先看具体实验入口；真正共享的模型、训练、评估逻辑由 `wafer_train` 包提供。

覆盖实验包括：

- CNN baseline
- SE-ResNet 32x32
- SE-ResNet 64x64
- Dual-head SE-ResNet 64x64
- DPFEE dual-head + one-hot baseline，也就是 B0
- W3，也就是 DPFEE dual-head + one-hot + OOF Cleanlab class-aware downweight

所有训练脚本默认使用：
conda run --no-capture-output -n pytorch2.3.1(cuda环境)

并显式使用 CUDA：
--device cuda

## 1. 目录结构

code/
  code.md
  configs/
    dpfeefocal_onehot_64.yaml
  experiments/
    _run_train.py
    train_cnn_baseline_32.py
    train_seresnet_32_ema_tta.py
    train_seresnet_64_ema_tta.py
    train_dualhead_seresnet_64_ema_tta.py
    train_b0_dpfee_dualhead_onehot.py
    train_w3_oof_cleanlab.py
  src/
    02_prepare_dataset.py
    summarize_experiments.py
    wafer_train/
      __init__.py
      __main__.py
      cli.py
      engine.py
      datasets.py
      models.py
      training.py
      evaluation.py
  data/
    dataset.py
    geometric_feature_utils.py
    pseudo_mask_utils.py
    cleanlab_utils.py
  models/
    base.py
    vit_wafer.py
    capsule_head.py
    dpfee_geometry_hybrid.py
    unet_dpfee_hybrid.py
    dpfeefocal.py
  scripts/
    00_prepare_defect8_data.ps1
    01_train_cnn_baseline_32.ps1
    02_train_seresnet_32_ema_tta.ps1
    03_train_seresnet_64_ema_tta.ps1
    04_train_dualhead_seresnet_64_ema_tta.ps1
    05_train_dpfee_dualhead_onehot_b0.ps1
    06_generate_oof_cleanlab_for_w3.ps1
    07_train_w3_oof_cleanlab.ps1
    08_summarize_main_networks.ps1
    generate_cleanlab_oof_probs.py

## 2. 模块功能说明

### 2.1 数据准备

`code/src/02_prepare_dataset.py`

从 `data/raw/LSWMD.pkl` 生成训练所需数据：
- `data/processed`：32x32 单通道 8 类数据
- `data/processed64`：64x64 单通道 8 类数据
- `data/processed64_onehot`：64x64 one-hot 三通道 8 类数据

当前实验使用自定义数据划分，而不是直接使用原始 `trianTestLabel`。

### 2.2 训练包 wafer_train

`code/src/wafer_train/engine.py`
共享训练引擎，包含原训练流程中的核心实现。负责：
- 参数解析
- 数据加载
- 模型构建
- 训练循环
- EMA / TTA
- Cleanlab 样本降权
- 指标、混淆矩阵和 checkpoint 保存

`code/src/wafer_train/models.py`
主要网络配置：
- `SimpleWaferCNN`
- `SEWaferCNN`
- `DualHeadSEWaferCNN`
- `DPFEELiteWaferCNN`
- `DualHeadDPFEELiteWaferCNN`
- `build_model`

`code/src/wafer_train/datasets.py`
数据集和标签工具：
- `WaferDataset`
- `load_label_info`
- `build_balanced_sampler`

`code/src/wafer_train/training.py`
训练相关工具：
- `set_seed`
- `build_criterion`
- `compute_class_weights`
- `ModelEma`
- `train_one_epoch`

`code/src/wafer_train/evaluation.py`
评估和绘图工具：
- `evaluate`
- `prediction_logits`
- `tta_variants`
- `plot_training_curves`
- `plot_confusion_matrix`
- `plot_per_class_recall`

`code/src/wafer_train/cli.py`
统一训练入口。

### 2.3 小型实验入口

`code/experiments/` 下每个文件只对应一个网络实验，只保留关键参数。

这些入口通过 code/experiments/_run_train.py 调用：
wafer_train.cli.main(...)

### 2.4 PowerShell 复现脚本

`code/scripts/*.ps1` 是可以直接运行的脚本。它们会进入项目根目录，并调用对应的 `code/experiments/*.py`。

## 3. 数据准备

运行：
.\code\scripts\00_prepare_defect8_data.ps1

生成：
data/processed
data/processed64
data/processed64_onehot

## 4. 各网络实验

### 4.1 CNN baseline

入口文件：code/experiments/train_cnn_baseline_32.py

运行脚本：
.\code\scripts\01_train_cnn_baseline_32.ps1

输出：outputs_code/cnn_baseline_32_50

网络结构：

输入：32x32 单通道

主要结构：
Conv-BN-ReLU-MaxPool
Conv-BN-ReLU-MaxPool
Conv-BN-ReLU
Conv-BN-ReLU
AdaptiveAvgPool
Dropout
Linear(128 -> 8)

### 4.2 SE-ResNet 32x32

入口文件：code/experiments/train_seresnet_32_ema_tta.py

运行脚本：
.\code\scripts\02_train_seresnet_32_ema_tta.ps1

输出：outputs_code/seresnet_32_50_ema_tta

网络结构：

输入：32x32 单通道

主要结构：
3×3 Conv stem
6 个带 SE 通道注意力的残差块
全局平均池化
dropout
8 类线性分类头
EMA
TTA

后续主要实验也默认使用 EMA 和 TTA。

### 4.3 SE-ResNet 64x64

入口文件：code/experiments/train_seresnet_64_ema_tta.py

运行脚本：
.\code\scripts\03_train_seresnet_64_ema_tta.ps1

输出：outputs_code/seresnet_64_50_ema_tta

输入从 32x32 提升到 64x64，用于保留更多缺陷空间细节。

### 4.4 Dual-head SE-ResNet 64x64

入口文件：code/experiments/train_dualhead_seresnet_64_ema_tta.py

运行脚本：
.\code\scripts\04_train_dualhead_seresnet_64_ema_tta.ps1

输出：
outputs_code/dualhead_seresnet_64_50_ema_tta

网络结构：

增加了 Dual-head 结构
  |-- main_head: 8-class logits
  |-- aux_head:  8-class logits

训练和推理：
total_loss = main_loss + 0.4 * aux_loss
final_logits = main_logits + 0.3 * aux_logits

### 4.5 DPFEE dual-head + one-hot baseline，B0

入口文件：code/experiments/train_b0_dpfee_dualhead_onehot.py

运行脚本：
.\code\scripts\05_train_dpfee_dualhead_onehot_b0.ps1

输出：outputs_code/B0_dpfee_dual_onehot_50

网络：

输入：64x64 one-hot 三通道

主要结构和策略：
DPFEE 双路径 backbone
dual-head
CE + label smoothing
train augmentation
EMA
TTA

### 4.6 生成 W3 所需 OOF Cleanlab

运行脚本：
.\code\scripts\06_generate_oof_cleanlab_for_w3.ps1

输出：
logs/cleanlab_oof/oof_pred_probs.npy
logs/cleanlab_oof/oof_labels.npy
logs/cleanlab_oof/cleanlab_label_issues.csv
logs/cleanlab_oof/cleanlab_keep_mask.npy

### 4.7 W3：DPFEE dual-head + one-hot + OOF Cleanlab

入口文件：code/experiments/train_w3_oof_cleanlab.py

运行脚本：
.\code\scripts\07_train_w3_oof_cleanlab.ps1

输出：outputs_code/W3_oof_cleanlab_caw_50

配置文件：code/configs/dpfeefocal_onehot_64.yaml

网络结构与策略：
DPFEE dual-head
64x64 one-hot 输入
CE + label smoothing
EMA
TTA
OOF Cleanlab class-aware downweight

Cleanlab 权重：
normal issue weight = 0.5
confusing pair weight = 0.7
strong issue weight = 0.2

## 5. 汇总实验结果

运行脚本：
.\code\scripts\08_summarize_main_networks.ps1

输出：
outputs_code/main_network_summary.csv
outputs_code/main_network_summary.md

## 6. 主要输出文件

每个训练目录会生成：
metrics.json
config.json
train_log.csv
training_curves.png
confusion_matrix.png
per_class_recall.png
per_class_metrics.csv
best_model.pth
last_model.pth

其中：
- `metrics.json`：最终测试指标和混淆矩阵数据。
- `training_curves.png`：训练/验证曲线。
- `confusion_matrix.png`：测试集混淆矩阵。
- `per_class_recall.png`：每类召回率柱状图。
- `best_model.pth`：验证集 macro-F1 最优 checkpoint。

## 7. 其他

由于先前做了很多尝试，所以代码较多，我们又重新整合了一下，在这个过程中可能还改变了一些参数，但是我都进行过复现，所以结果虽然会有些许差异（0.1%左右），但总体是一致的。

报告中提及的结果均在 outputs_code 文件夹中，但是比较早期的实验由于早期配置的原因，没有保存 config.json，所以没有记录配置，但是可以通过训练脚本查看。CNN_baseline 的结果由于太早被清理了，后续重新跑了一遍。

为了尽可能压缩工程，因为将处理后的所有数据集都提交 GitHub 会由于文件过大而无法上传，所以只保留了处理后的八类数据和部分九分类数据，如果要运行九分类代码，可能会需要一些时间来重新处理数据集。