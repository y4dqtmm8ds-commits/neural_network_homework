import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


# 随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 选择特征数量
def get_selected_features(feature_mode="all"):
    all_features = [
        "cement", "slag", "flyash", "water",
        "superplasticizer", "coarseaggregate",
        "fineaggregate", "age"
    ]

    high_corr_features = [
        "cement", "slag", "water",
        "superplasticizer", "age"
    ]

    core_features = [
        "cement", "water", "superplasticizer", "age"
    ]

    if feature_mode == "all":
        return all_features
    elif feature_mode == "high_corr":
        return high_corr_features
    elif feature_mode == "core":
        return core_features
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")


# 数据划分
def load_data(csv_path, feature_mode="all", use_y_scaling=False):
    df = pd.read_csv(csv_path)

    print("Missing values in each column:")
    print(df.isnull().sum())

    selected_cols = get_selected_features(feature_mode)
    print("\n当前特征模式:", feature_mode)
    print("使用特征:", selected_cols)

    X = df[selected_cols].values.astype(np.float32)
    y = df["csMPa"].values.astype(np.float32).reshape(-1, 1)

    # 前80%训练，后20%测试
    n = len(df)
    split_idx = int(0.8 * n)

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    # 标准化
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = None
    if use_y_scaling:
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)
    else:
        y_train_scaled = y_train
        y_test_scaled = y_test

    return (
        df,
        selected_cols,
        X_train, y_train, y_train_scaled,
        X_test, y_test, y_test_scaled,
        scaler_X, scaler_y
    )

def load_data_random(csv_path, feature_mode="all", use_y_scaling=False, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)

    print("Missing values in each column:")
    print(df.isnull().sum())

    selected_cols = get_selected_features(feature_mode)

    X = df[selected_cols].values.astype(np.float32)
    y = df["csMPa"].values.astype(np.float32).reshape(-1, 1)

    # 随机划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = None
    if use_y_scaling:
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)
    else:
        y_train_scaled = y_train
        y_test_scaled = y_test

    return (
        df,
        selected_cols,
        X_train, y_train, y_train_scaled,
        X_test, y_test, y_test_scaled,
        scaler_X, scaler_y
    )

def correlation_analysis(df, save_dir):
    corr = df.corr(numeric_only=True)

    print("\n与目标 csMPa 的相关性：")
    print(corr["csMPa"].sort_values(ascending=False))

    # # 热力图
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    # plt.title("混凝土数据集相关性热力图")
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "correlation_heatmap.png"), dpi=300)
    # plt.show()

    # 各特征与目标相关性条形图
    target_corr = corr["csMPa"].drop("csMPa").sort_values(ascending=False)

    plt.figure(figsize=(6, 5))
    target_corr.plot(kind="bar")
    plt.ylabel("相关系数")
    plt.title("各特征与混凝土强度的相关性")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_target_correlation.png"), dpi=300)
    plt.show()


def plot_train_test_distribution(df, save_dir):
    n = len(df)
    split_idx = int(0.8 * n)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(train_df["csMPa"], bins=20, alpha=0.7, label="训练集")
    plt.hist(test_df["csMPa"], bins=20, alpha=0.7, label="测试集")
    plt.xlabel("混凝土强度 csMPa")
    plt.ylabel("样本数")
    plt.title("训练集与测试集目标值分布")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(train_df["age"], bins=20, alpha=0.7, label="训练集")
    plt.hist(test_df["age"], bins=20, alpha=0.7, label="测试集")
    plt.xlabel("龄期 age")
    plt.ylabel("样本数")
    plt.title("训练集与测试集龄期分布")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_test_distribution.png"), dpi=300)
    plt.show()

def plot_feature_vs_target(df, save_dir):
    feature_cols = df.columns[:-1]
    target_col = df.columns[-1]

    plt.figure(figsize=(16, 10))
    for i, col in enumerate(feature_cols, 1):
        plt.subplot(3, 3, i)
        plt.scatter(df[col], df[target_col], alpha=0.6)
        plt.xlabel(col)
        plt.ylabel(target_col)
        plt.title(f"{col} 与 {target_col} 的关系")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_vs_target.png"), dpi=300)
    plt.show()


# 激活函数选择
def get_activation(activation_name="relu"):
    activation_name = activation_name.lower()

    if activation_name == "relu":
        return nn.ReLU()
    elif activation_name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01)
    elif activation_name == "elu":
        return nn.ELU()
    elif activation_name == "gelu":
        return nn.GELU()
    elif activation_name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation: {activation_name}")

# 构建神经网络
class ConcreteNet(nn.Module):
    def __init__(self, input_dim=8, activation_name="relu", dropout=0.0):
        super().__init__()
        act1 = get_activation(activation_name)
        act2 = get_activation(activation_name)

        layers = [
            nn.Linear(input_dim, 64),
            act1
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.extend([
            nn.Linear(64, 32),
            act2
        ])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(32, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# 训练函数
def train_model(model, train_loader, criterion, optimizer, epochs=300, device="cpu"):
    model.to(device)
    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}")

    return train_losses

# 测试函数
def evaluate_model(model, X_test, y_test, scaler_y=None, device="cpu"):
    model.eval()
    model.to(device)

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred_scaled = model(X_test_tensor).cpu().numpy()

    # 如果 y 做过标准化，这里还原回原始尺度
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
    else:
        y_pred = y_pred_scaled

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nTest Results:")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAE  = {mae:.4f}")
    print(f"R^2  = {r2:.4f}")

    return y_pred, mse, rmse, mae, r2


def plot_results(y_test, y_pred, train_losses, save_dir):
    # 训练损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses)
    plt.xlabel("训练轮数 Epoch")
    plt.ylabel("MSE 损失")
    plt.title("训练损失曲线")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_loss_curve.png"), dpi=300)
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("真实强度")
    plt.ylabel("预测强度")
    plt.title("混凝土强度真实值与预测值散点图")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "true_vs_predicted_scatter.png"), dpi=300)
    plt.show()

    # 前100个测试样本折线图
    show_num = min(100, len(y_test))
    plt.figure(figsize=(12, 5))
    plt.plot(range(show_num), y_test[:show_num], label="真实值", marker='o', markersize=3)
    plt.plot(range(show_num), y_pred[:show_num], label="预测值", marker='x', markersize=3)
    plt.xlabel("样本编号")
    plt.ylabel("混凝土强度")
    plt.title("前100个测试样本的真实值与预测值对比")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "true_vs_predicted_line.png"), dpi=300)
    plt.show()


# 保存结果
def save_metrics(save_dir, feature_mode, activation_name, use_y_scaling,
                 dropout, lr, epochs, mse, rmse, mae, r2, selected_cols):
    result_path = os.path.join(save_dir, "results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("混凝土强度回归实验结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"特征模式 feature_mode = {feature_mode}\n")
        f.write(f"使用特征 = {selected_cols}\n")
        f.write(f"激活函数 activation_name = {activation_name}\n")
        f.write(f"是否对 y 标准化 use_y_scaling = {use_y_scaling}\n")
        f.write(f"dropout = {dropout}\n")
        f.write(f"learning_rate = {lr}\n")
        f.write(f"epochs = {epochs}\n")
        f.write("-" * 50 + "\n")
        f.write(f"MSE  = {mse:.4f}\n")
        f.write(f"RMSE = {rmse:.4f}\n")
        f.write(f"MAE  = {mae:.4f}\n")
        f.write(f"R^2  = {r2:.4f}\n")

    print(f"\n结果已保存到: {result_path}")


# =========================
# 13. 主函数
# =========================
def main():
    set_seed(42)

    # ========= 可调参数区域 =========
    csv_path = "Concrete_Data_Yeh.csv"

    feature_mode = "high_corr"          # 可选: all / high_corr / core
    activation_name = "relu"      # 可选: relu / leakyrelu / elu / gelu / tanh
    use_y_scaling = True          # True / False
    dropout = 0.0                 # 例如 0.0 / 0.2
    batch_size = 32
    epochs = 300
    lr = 0.001
    weight_decay = 1e-4
    # ===============================

    save_dir = f"results_concrete_nn_{feature_mode}_{activation_name}_yscale_{use_y_scaling}"
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # # 读取数据
    # (
    #     df,
    #     selected_cols,
    #     X_train, y_train, y_train_scaled,
    #     X_test, y_test, y_test_scaled,
    #     scaler_X, scaler_y
    # ) = load_data(
    #     csv_path=csv_path,
    #     feature_mode=feature_mode,
    #     use_y_scaling=use_y_scaling
    # )

    # 读取数据
    (
        df,
        selected_cols,
        X_train, y_train, y_train_scaled,
        X_test, y_test, y_test_scaled,
        scaler_X, scaler_y
    ) = load_data_random(
        csv_path=csv_path,
        feature_mode=feature_mode,
        use_y_scaling=use_y_scaling
    )

    correlation_analysis(df, save_dir)
    plot_train_test_distribution(df, save_dir)
    plot_feature_vs_target(df, save_dir)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = ConcreteNet(
        input_dim=X_train.shape[1],
        activation_name=activation_name,
        dropout=dropout
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # 训练
    train_losses = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        device=device
    )

    # 测试
    y_pred, mse, rmse, mae, r2 = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        scaler_y=scaler_y if use_y_scaling else None,
        device=device
    )

    # 绘图
    plot_results(y_test, y_pred, train_losses, save_dir)

    # 保存结果
    save_metrics(
        save_dir=save_dir,
        feature_mode=feature_mode,
        activation_name=activation_name,
        use_y_scaling=use_y_scaling,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2=r2,
        selected_cols=selected_cols
    )


if __name__ == "__main__":
    main()