import os
import time
import copy
import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
set_seed(SEED)

TRAIN_MAT_PATH = "./data/train_32x32.mat"
TEST_MAT_PATH = "./data/test_32x32.mat"

BATCH_SIZE = 128
NUM_WORKERS = 2
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4

SAVE_DIR = "./runs_svhn_baseline"
os.makedirs(SAVE_DIR, exist_ok=True)


# 训练集做增强与标准化
class SVHNDataset(Dataset):
    def __init__(self, mat_path, transform=None):
        super().__init__()
        data = sio.loadmat(mat_path)

        self.images = data["X"]
        self.labels = data["y"].squeeze()

        # 将错误的标签改正
        self.labels[self.labels == 10] = 0

        # 转成 (N, 32, 32, 3)
        self.images = np.transpose(self.images, (3, 0, 1, 2))

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return img, label

# 数据预处理
def get_transforms():
    train_transform = transforms.Compose([
        # numpy 转换为 PIL，方便后续图像增强
        transforms.ToPILImage(),
        # 随机裁剪，增强鲁棒性
        transforms.RandomCrop(32, padding=4),
        # 轻微旋转
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        # 归一化到 [0,1]
        transforms.Normalize(
            mean=[0.4377, 0.4438, 0.4728],
            std=[0.1980, 0.2010, 0.1970]
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4377, 0.4438, 0.4728],
            std=[0.1980, 0.2010, 0.1970]
        )
    ])

    return train_transform, test_transform


def build_dataloaders(train_mat_path, test_mat_path,
                      batch_size=128, num_workers=2):
    train_transform, test_transform = get_transforms()

    train_dataset = SVHNDataset(train_mat_path, transform=train_transform)
    test_dataset = SVHNDataset(test_mat_path, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader

# CNN 网络构建，features：卷积特征提取部分 classifier：全连接分类部分
class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # block 1 提取低层特征
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # 对通道做归一化
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),   # 32x32 -> 16x16
            nn.Dropout(0.25),

            # block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),   # 16x16 -> 8x8
            nn.Dropout(0.25),

            # block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),   # 8x8 -> 4x4
            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 训练 / 测试函数
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0
    # 前向传播/计算损失/反向传播/更新参数
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total

    return epoch_loss, epoch_acc

def plot_curves(history, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("baseline CNN Train/Test Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=200)
    plt.close()

    # Accuracy 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["test_acc"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("baseline CNN Train/Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=200)
    plt.close()

def main():
    print("Device:", DEVICE)
    print("Loading data...")

    train_loader, test_loader = build_dataloaders(
        TRAIN_MAT_PATH,
        TEST_MAT_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches : {len(test_loader)}")

    model = BaselineCNN(num_classes=10).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )

        test_loss, test_acc = evaluate(
            model, test_loader, criterion, DEVICE
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(SAVE_DIR, "best_model.pth"))

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

    total_time = time.time() - start_time

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "last_model.pth"))

    # 恢复最优模型并再次评估
    model.load_state_dict(best_model_wts)
    best_test_loss, best_test_acc = evaluate(model, test_loader, criterion, DEVICE)

    # 保存训练记录
    np.save(os.path.join(SAVE_DIR, "history.npy"), history, allow_pickle=True)

    # 绘图
    plot_curves(history, SAVE_DIR)

    print("\nTraining finished.")
    print(f"Best Test Accuracy: {best_test_acc:.4f}")
    print(f"Best Test Loss    : {best_test_loss:.4f}")
    print(f"Total Time        : {total_time:.2f}s")
    print(f"Results saved to  : {SAVE_DIR}")


if __name__ == "__main__":
    main()