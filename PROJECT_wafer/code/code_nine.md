# 九分类 N1/N4 代码说明

本文档说明 `code` 文件夹中新增加的九分类实验代码。新增内容用于复现加入 `Normal` 类后的 N1 和 N4 两个实验，不会覆盖原有 8 类实验结果。

## 1. 新增文件结构

code/
  configs/
    dpfeefocal_onehot_64_9cls.yaml
  experiments/
    train_n1_9cls_w3_no_cleanlab.py
    train_n4_9cls_normal_gate_w3.py
  scripts/
    09_prepare_all9_data.ps1
    10_train_n1_9cls_w3_no_cleanlab.ps1
    11_train_n4_9cls_normal_gate_w3.ps1
    12_summarize_n1_n4_9cls.ps1
    run_n4_normal_gate_w3.py

## 2. 数据准备脚本

文件：code/scripts/09_prepare_all9_data.ps1

功能：从原始 `LSWMD.pkl` 数据中重新生成 9 类数据集，包含 8 类缺陷和第 9 类 `Normal`。输出目录为：data/processed64_onehot_all9

主要配置：
- 输入尺寸：64x64
- 输入编码：one-hot
- 类别：9 类，包含 `Normal`
- 数据划分：按类别分层划分 train / val / test
- 随机种子：42

运行命令：
powershell -ExecutionPolicy Bypass -File code\scripts\09_prepare_all9_data.ps1

## 3. N1：直接九分类策略，无 Cleanlab

### 3.1 配置文件

code/configs/dpfeefocal_onehot_64_9cls.yaml

功能：

定义 9 类 DPFEE dual-head 训练配置。该配置基本沿用 8 类 W3(DPFEE Dual-head Cleanlab) 的训练策略，但不启用 Cleanlab。

核心配置：
- 数据目录：`data/processed64_onehot_all9`
- 输出目录默认：`outputs_code_9cls/config_default`
- 类别数：9
- 类别名：`Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, Normal`
- 输入尺寸：64x64
- 输入通道：one-hot，通常为 3 通道
- 模型：DPFEE dual-head
- 主损失：交叉熵 + label smoothing
- `label_smoothing=0.03`
- 训练增强：开启
- EMA：开启
- TTA：开启
- auxiliary head：开启，`aux_loss_weight=0.4`
- hard classes：`Donut, Edge-Loc, Loc, Scratch`
- Cleanlab：默认关闭

### 3.2 实验入口

文件：code/experiments/train_n1_9cls_w3_no_cleanlab.py

功能：调用 `code/configs/dpfeefocal_onehot_64_9cls.yaml`，训练直接 9 类分类模型。

输出目录：outputs_code_9cls/N1_9cls_w3_no_cleanlab

运行脚本：
powershell -ExecutionPolicy Bypass -File code\scripts\10_train_n1_9cls_w3_no_cleanlab.ps1

主要输出：
outputs_code_9cls/N1_9cls_w3_no_cleanlab

## 4. N4：Normal gate + 已训练 8 类 W3

### 4.1 方法说明

N4 是一个两阶段 pipeline：
Stage 1：训练 Normal vs Defect 二分类器
Stage 2：若样本被判为 Defect，则调用已有 8 类 W3 模型进行缺陷分类

与 N1 的区别：
- N1 是一个 9 类 softmax 模型，一次性输出 9 类结果。
- N4 先判断是否为 `Normal`，只有被判断为缺陷的样本才进入 8 类 W3 缺陷分类器。
- N4 的 Stage 2 使用已经训练好的 8 类 W3 checkpoint，默认不会在 N4 训练过程中更新。

### 4.2 N4 独立脚本

文件：code/scripts/run_n4_normal_gate_w3.py

功能：
- 读取 9 类数据
- 自动识别 `Normal` 类索引
- 构建二分类训练集
- 训练 Stage 1 Normal-vs-Defect 二分类 DPFEE
- 加载已有 8 类 W3 模型
- 对测试集执行两阶段推理
- 生成 9 类指标、二分类指标、混淆矩阵和训练曲线

默认 8 类 W3 checkpoint：outputs64/exp50_W3_oof_cleanlab_caw_n05_c07_s02/best_model.pth

如果使用其他 W3 权重，可以在命令行中修改 `--w3-checkpoint`。

### 4.3 实验入口

文件：code/experiments/train_n4_9cls_normal_gate_w3.py

功能：以固定参数调用 `code/scripts/run_n4_normal_gate_w3.py`，用于复现 N4 实验。

核心配置：
- 数据目录：`data/processed64_onehot_all9`
- 输出目录：`outputs_code_9cls/N4_9cls_normal_gate_w3`
- Stage 1 模型：DPFEE 二分类器
- Stage 1 类别：`Normal` vs `Defect`
- Stage 2 模型：已有 8 类 W3
- Stage 1 训练轮数：20
- batch size：128
- 学习率：`3e-4`
- weight decay：`1e-4`
- label smoothing：0.03
- binary threshold：0.5
- seed：42

运行脚本：
powershell -ExecutionPolicy Bypass -File code\scripts\11_train_n4_9cls_normal_gate_w3.ps1

主要输出：
outputs_code_9cls/N4_9cls_normal_gate_w3

其中：
- `confusion_matrix.png` 是最终 9 类分类混淆矩阵。
- `binary_confusion_matrix.png` 是 Stage 1 的 Normal-vs-Defect 二分类混淆矩阵。
- `metrics.json` 中包含最终 9 类 accuracy、9 类 macro-F1、defect-only macro-F1、Normal 误判为 Defect 的比例、Defect 误判为 Normal 的比例。

## 5. N1/N4 结果汇总

文件：code/scripts/12_summarize_n1_n4_9cls.ps1

功能：读取 N1 和 N4 的 `metrics.json`，生成九分类实验汇总表。

运行命令：
powershell -ExecutionPolicy Bypass -File code\scripts\12_summarize_n1_n4_9cls.ps1

输出文件：
outputs_code_9cls/summary_N1_N4_9cls.csv
outputs_code_9cls/summary_N1_N4_9cls.md

汇总字段包括：
- experiment_name
- accuracy
- macro_F1_9cls
- defect_macro_F1_8cls
- Normal_precision
- Normal_recall
- Donut_recall
- Loc_recall
- Scratch_recall
- Edge_Loc_recall
- best_epoch
- notes

## 6. N1 与 N4 的实验含义

N1：直接 9 类分类，将 `Normal` 与 8 类缺陷一起放入同一个 softmax 分类头中学习。它的优点是结构简单、推理流程直接；缺点是 `Normal` 类和缺陷类之间的样本量差异可能影响 macro-F1，且 Normal/Defect 边界需要由同一个 9 类分类器同时学习。

N4：先判断 Normal-vs-Defect，再对 Defect 样本调用已有 8 类 W3。它的优点是可以复用已经表现较好的 8 类缺陷分类器，并把 Normal 判断单独建模；缺点是 Stage 1 的二分类错误会直接影响最终结果，如果缺陷被误判为 Normal，Stage 2 没有机会纠正。

