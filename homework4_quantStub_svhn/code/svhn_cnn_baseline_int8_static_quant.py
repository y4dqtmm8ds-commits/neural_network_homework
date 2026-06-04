import os
import time
import copy
import json
import random
import argparse
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

try:
    from torch.ao.quantization import (
        QuantStub,
        DeQuantStub,
        QConfig,
        MinMaxObserver,
        prepare,
        convert,
        fuse_modules,
    )
except Exception:
    from torch.quantization import (  # 兼容旧版 PyTorch
        QuantStub,
        DeQuantStub,
        QConfig,
        MinMaxObserver,
        prepare,
        convert,
        fuse_modules,
    )

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


SEED = 42
set_seed(SEED)

TRAIN_MAT_PATH = "./data/train_32x32.mat"
TEST_MAT_PATH = "./data/test_32x32.mat"

BATCH_SIZE = 128
NUM_WORKERS = 8
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4

SAVE_DIR = "runs_svhn_baseline_int8_batch128_2"
os.makedirs(SAVE_DIR, exist_ok=True)


# 线性量化函数
def linear_quantize(x: torch.Tensor, num_bits: int = 8):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    x = x.detach().float()
    q_min = 0
    q_max = 2 ** num_bits - 1

    x_min = x.min().item()
    x_max = x.max().item()

    if abs(x_max - x_min) < 1e-12:
        scale = 1.0
        zero_point = 0
        q = torch.zeros_like(x, dtype=torch.uint8)
        return q, scale, zero_point

    scale = (x_max - x_min) / float(q_max - q_min)
    zero_point = q_min - round(x_min / scale)
    zero_point = int(max(q_min, min(q_max, zero_point)))

    q = torch.round(x / scale + zero_point)
    q = torch.clamp(q, q_min, q_max).to(torch.uint8)
    return q, float(scale), int(zero_point)


def linear_dequantize(q: torch.Tensor, scale: float, zero_point: int):
    return scale * (q.float() - float(zero_point))


# 数据集与预处理
class SVHNDataset(Dataset):
    def __init__(self, mat_path, transform=None):
        super().__init__()
        data = sio.loadmat(mat_path)

        self.images = data["X"]
        self.labels = data["y"].squeeze()
        self.labels[self.labels == 10] = 0
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
        transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
    ])

    return train_transform, test_transform


def build_dataloaders(train_mat_path, test_mat_path, batch_size=128, num_workers=2, calib_size=1024):
    train_transform, test_transform = get_transforms()

    train_dataset = SVHNDataset(train_mat_path, transform=train_transform)
    train_dataset_for_calib = SVHNDataset(train_mat_path, transform=test_transform)
    test_dataset = SVHNDataset(test_mat_path, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    calib_size = min(calib_size, len(train_dataset_for_calib))
    calib_indices = list(range(calib_size))
    calib_dataset = Subset(train_dataset_for_calib, calib_indices)
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, test_loader, calib_loader


class QuantizableBaselineCNN(nn.Module):
    def __init__(self, num_classes=10, use_quant_stubs=False):
        super().__init__()
        self.use_quant_stubs = use_quant_stubs
        if use_quant_stubs:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=False)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.25)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(inplace=False)

        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.relu6 = nn.ReLU(inplace=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu_fc1 = nn.ReLU(inplace=False)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        if self.use_quant_stubs:
            x = self.quant(x)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool1(x))

        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.drop2(self.pool2(x))

        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.drop3(self.pool3(x))

        x = self.flatten(x)
        x = self.relu_fc1(self.fc1(x))
        x = self.drop_fc(x)
        x = self.fc2(x)

        if self.use_quant_stubs:
            x = self.dequant(x)
        return x

    def fuse_model(self):
        """融合 Conv-BN-ReLU 和 Linear-ReLU。必须在 eval() 模式下执行。"""
        fuse_modules(self, [["conv1", "bn1", "relu1"]], inplace=True)
        fuse_modules(self, [["conv2", "bn2", "relu2"]], inplace=True)
        fuse_modules(self, [["conv3", "bn3", "relu3"]], inplace=True)
        fuse_modules(self, [["conv4", "bn4", "relu4"]], inplace=True)
        fuse_modules(self, [["conv5", "bn5", "relu5"]], inplace=True)
        fuse_modules(self, [["conv6", "bn6", "relu6"]], inplace=True)
        fuse_modules(self, [["fc1", "relu_fc1"]], inplace=True)


def convert_old_baseline_state_dict(old_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    mapping = {
        "features.0": "conv1", "features.1": "bn1",
        "features.3": "conv2", "features.4": "bn2",
        "features.8": "conv3", "features.9": "bn3",
        "features.11": "conv4", "features.12": "bn4",
        "features.16": "conv5", "features.17": "bn5",
        "features.19": "conv6", "features.20": "bn6",
        "classifier.1": "fc1",
        "classifier.4": "fc2",
    }
    new_state = {}
    for k, v in old_state.items():
        replaced = False
        for old_prefix, new_prefix in mapping.items():
            if k.startswith(old_prefix + "."):
                new_state[k.replace(old_prefix, new_prefix, 1)] = v
                replaced = True
                break
        if not replaced:
            new_state[k] = v
    return new_state


def load_state_dict_flexible(model: nn.Module, path: str, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        state = convert_old_baseline_state_dict(state)
        model.load_state_dict(state, strict=True)

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

    return running_loss / total, running_correct / total


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

    return running_loss / total, running_correct / total


def plot_training_curves(history, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline CNN Train/Test Loss")
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
    plt.title("Baseline CNN Train/Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=200)
    plt.close()


def plot_quant_compare(metrics: Dict, save_dir: str):
    # 精度对比图
    plt.figure(figsize=(6, 5))
    names = ["FP32", "INT8"]
    values = [metrics["fp32_acc"], metrics["int8_acc"]]
    plt.bar(names, values)
    plt.ylabel("Accuracy")
    plt.title("FP32 vs INT8 Accuracy")
    plt.ylim(max(0.0, min(values) - 0.05), min(1.0, max(values) + 0.03))
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fp32_vs_int8_accuracy.png"), dpi=200)
    plt.close()

    # CPU 单张延迟对比图
    plt.figure(figsize=(6, 5))
    values = [metrics["fp32_latency_ms"], metrics["int8_latency_ms"]]
    plt.bar(names, values)
    plt.ylabel("Latency per image on CPU (ms)")
    plt.title("FP32 vs INT8 CPU Inference Latency")
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fp32_vs_int8_latency.png"), dpi=200)
    plt.close()


def plot_layer_mse(layer_mse: Dict[str, float], save_dir: str):
    if not layer_mse:
        return
    names = list(layer_mse.keys())
    values = [layer_mse[k] for k in names]
    plt.figure(figsize=(10, 5))
    plt.bar(names, values)
    plt.ylabel("MSE")
    plt.title("Layer Output MSE: FP32 fused vs INT8")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "layer_output_mse.png"), dpi=200)
    plt.close()

# 模型大小 / 延迟 / 层输出 MSE
def get_model_size_mb(model: nn.Module, save_path: str) -> float:
    torch.save(model.state_dict(), save_path)
    size_mb = os.path.getsize(save_path) / (1024 ** 2)
    return size_mb


@torch.no_grad()
def benchmark_latency_cpu(model: nn.Module, sample: torch.Tensor, warmup=50, repeat=300) -> float:
    model = model.cpu().eval()
    sample = sample.cpu()

    for _ in range(warmup):
        _ = model(sample)

    start = time.perf_counter()
    for _ in range(repeat):
        _ = model(sample)
    end = time.perf_counter()

    return (end - start) * 1000.0 / repeat


def _to_float_tensor(x):
    if isinstance(x, torch.Tensor):
        if x.is_quantized:
            return x.dequantize().detach().cpu().float()
        return x.detach().cpu().float()
    return x


@torch.no_grad()
def compute_layer_output_mse(fp32_fused_model: nn.Module,
                             int8_model: nn.Module,
                             loader: DataLoader,
                             layer_names: List[str],
                             num_batches: int = 5) -> Dict[str, float]:
    fp32_fused_model.cpu().eval()
    int8_model.cpu().eval()

    fp32_outputs = {}
    int8_outputs = {}
    handles = []

    def make_hook(store: Dict, name: str):
        def hook(module, inp, out):
            store[name] = _to_float_tensor(out)
        return hook

    for name in layer_names:
        m1 = dict(fp32_fused_model.named_modules()).get(name, None)
        m2 = dict(int8_model.named_modules()).get(name, None)
        if m1 is not None and m2 is not None:
            handles.append(m1.register_forward_hook(make_hook(fp32_outputs, name)))
            handles.append(m2.register_forward_hook(make_hook(int8_outputs, name)))

    mse_sum = {name: 0.0 for name in layer_names}
    mse_count = {name: 0 for name in layer_names}

    for batch_idx, (images, _) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        images = images.cpu()
        fp32_outputs.clear()
        int8_outputs.clear()

        _ = fp32_fused_model(images)
        _ = int8_model(images)

        for name in layer_names:
            if name in fp32_outputs and name in int8_outputs:
                a = fp32_outputs[name]
                b = int8_outputs[name]
                if a.shape == b.shape:
                    mse = torch.mean((a - b) ** 2).item()
                    mse_sum[name] += mse
                    mse_count[name] += 1

    for h in handles:
        h.remove()

    result = {}
    for name in layer_names:
        if mse_count[name] > 0:
            result[name] = mse_sum[name] / mse_count[name]
    return result

# 静态量化 PTQ
def build_static_qconfig():
    activation_observer = MinMaxObserver.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
    )
    weight_observer = MinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
    )
    return QConfig(activation=activation_observer, weight=weight_observer)


@torch.no_grad()
def calibrate(model: nn.Module, calib_loader: DataLoader, max_batches: int = 20):
    model.eval()
    for batch_idx, (images, _) in enumerate(calib_loader):
        if batch_idx >= max_batches:
            break
        _ = model(images.cpu())


def quantize_model_ptq(fp32_model: QuantizableBaselineCNN,
                       calib_loader: DataLoader,
                       backend: str = "fbgemm",
                       calib_batches: int = 20) -> Tuple[nn.Module, nn.Module]:
    if backend not in torch.backends.quantized.supported_engines:
        print(f"[Warning] backend={backend} not supported. Supported: {torch.backends.quantized.supported_engines}")
        backend = torch.backends.quantized.supported_engines[0]
    torch.backends.quantized.engine = backend

    # FP32 融合模型，用于 MSE 对比
    fp32_fused_model = copy.deepcopy(fp32_model).cpu().eval()
    fp32_fused_model.fuse_model()

    # INT8 准备模型，插入 QuantStub / DeQuantStub
    int8_model = copy.deepcopy(fp32_model).cpu().eval()
    int8_model.use_quant_stubs = True
    int8_model.quant = QuantStub()
    int8_model.dequant = DeQuantStub()
    int8_model.fuse_model()
    int8_model.qconfig = build_static_qconfig()

    for name, module in int8_model.named_modules():
        if isinstance(module, nn.Dropout):
            module.qconfig = None

    prepare(int8_model, inplace=True)
    calibrate(int8_model, calib_loader, max_batches=calib_batches)
    convert(int8_model, inplace=True)
    return fp32_fused_model, int8_model


def parse_args():
    parser = argparse.ArgumentParser(description="SVHN Baseline CNN + INT8 static quantization PTQ")
    parser.add_argument("--train_mat", type=str, default=TRAIN_MAT_PATH)
    parser.add_argument("--test_mat", type=str, default=TEST_MAT_PATH)
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--skip_train", action="store_true", help="跳过训练，直接加载 --weight_path")
    parser.add_argument("--weight_path", type=str, default="", help="已有 FP32 权重路径，例如 runs_svhn_baseline/best_model.pth")
    parser.add_argument("--calib_size", type=int, default=1024)
    parser.add_argument("--calib_batches", type=int, default=20)
    parser.add_argument("--backend", type=str, default="fbgemm", choices=["fbgemm", "qnnpack", "x86"])
    parser.add_argument("--latency_repeat", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.skip_train else "cpu")
    print("Device for training/eval:", device)
    print("Quantized INT8 inference will be evaluated on CPU.")

    print("Loading data...")
    train_loader, test_loader, calib_loader = build_dataloaders(
        args.train_mat,
        args.test_mat,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        calib_size=args.calib_size,
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches : {len(test_loader)}")
    print(f"Calib batches: {len(calib_loader)}")

    criterion = nn.CrossEntropyLoss()
    model = QuantizableBaselineCNN(num_classes=10, use_quant_stubs=False).to(device)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    if args.skip_train:
        if not args.weight_path:
            raise ValueError("使用 --skip_train 时必须提供 --weight_path")
        print(f"Loading FP32 checkpoint: {args.weight_path}")
        load_state_dict_flexible(model, args.weight_path, map_location=device)
        _, best_acc = evaluate(model, test_loader, criterion, device)
        best_model_wts = copy.deepcopy(model.state_dict())
    else:
        if args.weight_path and os.path.exists(args.weight_path):
            print(f"Loading initial checkpoint: {args.weight_path}")
            load_state_dict_flexible(model, args.weight_path, map_location=device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        start_time = time.time()
        for epoch in range(args.epochs):
            epoch_start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, os.path.join(args.save_dir, "best_model_fp32.pth"))

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch [{epoch+1:02d}/{args.epochs:02d}] | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )

        total_time = time.time() - start_time
        torch.save(model.state_dict(), os.path.join(args.save_dir, "last_model_fp32.pth"))
        np.save(os.path.join(args.save_dir, "history.npy"), history, allow_pickle=True)
        plot_training_curves(history, args.save_dir)
        print(f"Training time: {total_time:.2f}s")

    # 恢复最优 FP32 模型并评估
    model.load_state_dict(best_model_wts)
    model = model.cpu().eval()
    fp32_loss, fp32_acc = evaluate(model, test_loader, criterion, torch.device("cpu"))
    print(f"\nFP32 Test Loss: {fp32_loss:.4f} | FP32 Test Acc: {fp32_acc:.4f}")

    # 静态量化 PTQ
    print("\nPreparing INT8 static quantization...")
    fp32_fused_model, int8_model = quantize_model_ptq(
        model,
        calib_loader,
        backend=args.backend,
        calib_batches=args.calib_batches,
    )

    int8_loss, int8_acc = evaluate(int8_model, test_loader, criterion, torch.device("cpu"))
    print(f"INT8 Test Loss: {int8_loss:.4f} | INT8 Test Acc: {int8_acc:.4f}")

    # 模型大小
    fp32_size_mb = get_model_size_mb(fp32_fused_model, os.path.join(args.save_dir, "fp32_fused_state_dict.pth"))
    int8_size_mb = get_model_size_mb(int8_model, os.path.join(args.save_dir, "int8_quantized_state_dict.pth"))

    # CPU 单张延迟
    sample_images, _ = next(iter(test_loader))
    sample = sample_images[:1].cpu()
    fp32_latency_ms = benchmark_latency_cpu(fp32_fused_model, sample, repeat=args.latency_repeat)
    int8_latency_ms = benchmark_latency_cpu(int8_model, sample, repeat=args.latency_repeat)

    # 每层输出 MSE
    layer_names = ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "fc1", "fc2"]
    layer_mse = compute_layer_output_mse(
        fp32_fused_model,
        int8_model,
        test_loader,
        layer_names=layer_names,
        num_batches=5,
    )

    metrics = {
        "fp32_loss": fp32_loss,
        "fp32_acc": fp32_acc,
        "int8_loss": int8_loss,
        "int8_acc": int8_acc,
        "accuracy_drop": fp32_acc - int8_acc,
        "fp32_size_mb": fp32_size_mb,
        "int8_size_mb": int8_size_mb,
        "compression_ratio": fp32_size_mb / int8_size_mb if int8_size_mb > 0 else None,
        "fp32_latency_ms": fp32_latency_ms,
        "int8_latency_ms": int8_latency_ms,
        "speedup_ratio": fp32_latency_ms / int8_latency_ms if int8_latency_ms > 0 else None,
        "backend": torch.backends.quantized.engine,
        "calib_size": args.calib_size,
        "calib_batches": args.calib_batches,
        "layer_output_mse": layer_mse,
    }

    with open(os.path.join(args.save_dir, "quantization_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    plot_quant_compare(metrics, args.save_dir)
    plot_layer_mse(layer_mse, args.save_dir)

    print("\n========== Quantization Summary ==========")
    print(f"FP32 Acc       : {fp32_acc:.4f}")
    print(f"INT8 Acc       : {int8_acc:.4f}")
    print(f"Accuracy Drop  : {fp32_acc - int8_acc:.4f}")
    print(f"FP32 Size      : {fp32_size_mb:.3f} MB")
    print(f"INT8 Size      : {int8_size_mb:.3f} MB")
    print(f"Compression    : {metrics['compression_ratio']:.3f}x")
    print(f"FP32 Latency   : {fp32_latency_ms:.3f} ms/image")
    print(f"INT8 Latency   : {int8_latency_ms:.3f} ms/image")
    print(f"Speedup        : {metrics['speedup_ratio']:.3f}x")
    print("Layer MSE      :")
    for k, v in layer_mse.items():
        print(f"  {k}: {v:.8f}")
    print(f"\nResults saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
