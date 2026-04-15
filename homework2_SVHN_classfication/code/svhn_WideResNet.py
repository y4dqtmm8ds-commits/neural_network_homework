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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
set_seed(SEED)

TRAIN_MAT_PATH = os.path.join(BASE_DIR, "data", "train_32x32.mat")
TEST_MAT_PATH = os.path.join(BASE_DIR, "data", "test_32x32.mat")

BATCH_SIZE = 128
NUM_WORKERS = 0
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 5e-4

SAVE_DIR = os.path.join(BASE_DIR, "runs_svhn_wideresnet_100epoch")
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


def get_transforms():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4377, 0.4438, 0.4728],
            std=[0.1980, 0.2010, 0.1970]
        ),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))
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


def build_dataloaders(train_mat_path, test_mat_path, batch_size=128, num_workers=0):
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

# WideResNet 结构
class WideBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate, stride):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, stride=stride, bias=False
            )

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):

    def __init__(self, depth=28, widen_factor=2, dropout_rate=0.3, num_classes=10):
        super().__init__()

        assert (depth - 4) % 6 == 0, "WideResNet depth should satisfy depth = 6n + 4"
        n = (depth - 4) // 6
        k = widen_factor

        n_channels = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(
            in_channels=n_channels[0],
            out_channels=n_channels[1],
            num_blocks=n,
            dropout_rate=dropout_rate,
            stride=1
        )
        self.layer2 = self._make_layer(
            in_channels=n_channels[1],
            out_channels=n_channels[2],
            num_blocks=n,
            dropout_rate=dropout_rate,
            stride=2
        )
        self.layer3 = self._make_layer(
            in_channels=n_channels[2],
            out_channels=n_channels[3],
            num_blocks=n,
            dropout_rate=dropout_rate,
            stride=2
        )

        self.bn = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_channels[3], num_classes)

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, dropout_rate, stride):
        layers = []
        for i in range(num_blocks):
            block_stride = stride if i == 0 else 1
            block_in_channels = in_channels if i == 0 else out_channels
            layers.append(
                WideBasicBlock(
                    in_channels=block_in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    stride=block_stride
                )
            )
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn(out))
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def build_wideresnet(num_classes=10):
    return WideResNet(depth=28, widen_factor=2, dropout_rate=0.3, num_classes=num_classes)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

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

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("WideResNet Train/Test Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["test_acc"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("WideResNet Train/Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=200)
    plt.close()


def main():
    print("DEVICE =", DEVICE)
    print("TRAIN_MAT_PATH =", TRAIN_MAT_PATH)
    print("TEST_MAT_PATH  =", TEST_MAT_PATH)
    print("train exists?  =", os.path.exists(TRAIN_MAT_PATH))
    print("test exists?   =", os.path.exists(TEST_MAT_PATH))

    train_loader, test_loader = build_dataloaders(
        TRAIN_MAT_PATH,
        TEST_MAT_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    model = build_wideresnet(num_classes=10).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # early stopping
    patience = 10
    no_improve = 0

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

        improved = test_acc > best_acc
        if improved:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(SAVE_DIR, "best_model.pth"))
            no_improve = 0
        else:
            no_improve += 1

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s"
        )

        if no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break

    total_time = time.time() - start_time

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "last_model.pth"))

    model.load_state_dict(best_model_wts)
    best_test_loss, best_test_acc = evaluate(model, test_loader, criterion, DEVICE)

    np.save(os.path.join(SAVE_DIR, "history.npy"), history, allow_pickle=True)
    plot_curves(history, SAVE_DIR)

    print("\nTraining finished.")
    print(f"Best Test Accuracy: {best_test_acc:.4f}")
    print(f"Best Test Loss    : {best_test_loss:.4f}")
    print(f"Total Time        : {total_time:.2f}s")
    print(f"Results saved to  : {SAVE_DIR}")


if __name__ == "__main__":
    main()