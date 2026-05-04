import os
import re
import json
import random
import argparse
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt

# 加入押韵判断
try:
    from pypinyin import lazy_pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False

# 固定随机种子
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_chinese_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"

# 只保留汉字，在后面生成阶段再插入标点
def clean_poem_text(text: str) -> str:
    return "".join([ch for ch in text if is_chinese_char(ch)])

# 读取数据集，不强行筛选绝句，而是清洗后作为连续字符语料。
def load_json_poems(data_dir: str, min_len: int = 20) -> List[str]:
    json_files = []

    for name in os.listdir(data_dir):
        if name.endswith(".json") and (
            name.startswith("poet.song") or name.startswith("poet.tang")
        ):
            json_files.append(os.path.join(data_dir, name))

    json_files = sorted(json_files)

    if len(json_files) == 0:
        raise FileNotFoundError(
            f"在 {data_dir} 中没有找到 poet.song*.json 或 poet.tang*.json 文件。"
        )

    print("使用数据文件：")
    for f in json_files:
        print("  ", f)

    poems = []
    # 读取每个文件
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            paragraphs = item.get("paragraphs", [])
            if not isinstance(paragraphs, list):
                continue

            raw_text = "".join(paragraphs)
            clean_text = clean_poem_text(raw_text)
            # 只保留长度足够的诗词
            if len(clean_text) >= min_len:
                poems.append(clean_text)

    poems = sorted(list(set(poems)))

    if len(poems) == 0:
        raise ValueError("没有得到可用诗歌，请检查 json 文件内容。\n")

    print("样例：\n")
    for p in poems[:5]:
        print(p[:80])

    return poems

# 创建字符表，统计所有出现过的汉字
def build_vocab(
    poems: List[str],
    min_freq: int = 1,
) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    freq = {}

    for poem in poems:
        for ch in poem:
            freq[ch] = freq.get(ch, 0) + 1
    # 过滤低频的字
    chars = [ch for ch, c in freq.items() if c >= min_freq]
    chars = sorted(chars)

    vocab = ["<PAD>", "<UNK>"] + chars

    char2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2char = {i: ch for ch, i in char2idx.items()}

    print(f"\n词表大小：{len(vocab)}")

    return char2idx, idx2char, vocab

# 将诗歌文本转换为索引序列
def poems_to_sequences(
    poems: List[str],
    char2idx: Dict[str, int],
) -> List[List[int]]:
    unk_idx = char2idx["<UNK>"]
    sequences = []

    for poem in poems:
        seq = [char2idx.get(ch, unk_idx) for ch in poem]
        if len(seq) > 1:
            sequences.append(seq)

    return sequences


# 滑动窗口训练样本，序列预测
class PoetryDataset(Dataset):
    def __init__(
        self,
        sequences: List[List[int]],
        seq_len: int = 32,
        stride: int = 1, # 默认每次移动一个字，样本数量足够
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.data = []

        for seq in sequences:
            if len(seq) <= seq_len:
                continue

            for i in range(0, len(seq) - seq_len, stride):
                x = seq[i:i + seq_len]
                y = seq[i + 1:i + 1 + seq_len]
                self.data.append((x, y))

        if len(self.data) == 0:
            raise ValueError(
                "没有构造出训练样本，减小 --seq_len。\n"
            )

        print(f"滑动窗口训练样本数：{len(self.data)}\n")
        print(f"seq_len = {seq_len}, stride = {stride}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.LongTensor(x), torch.LongTensor(y)


# 模型结构，字符编号 → Embedding → RNN/LSTM/GRU → Linear → 每个字的概率分布
class PoetryRNNModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        model_type: str = "gru",
        dropout: float = 0.2,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type.lower()
        # 把一个字的编号转换成一个向量
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        rnn_dropout = dropout if num_layers > 1 else 0.0

        if self.model_type == "rnn":
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity="tanh",
                dropout=rnn_dropout,
                batch_first=False,
            )
        elif self.model_type == "gru":
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=rnn_dropout,
                batch_first=False,
            )
        elif self.model_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=rnn_dropout,
                batch_first=False,
            )
        else:
            raise ValueError("model_type 只能是 rnn / gru / lstm")

        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        output, hidden = self.rnn(emb, hidden)
        logits = self.linear(output)
        return logits, hidden


# 获取汉字的拼音韵母，用于押韵
def get_pinyin_final(ch: str) -> str:

    if not HAS_PYPINYIN:
        return ""

    if not is_chinese_char(ch):
        return ""

    try:
        finals = lazy_pinyin(ch, style=Style.FINALS, strict=False)
        if len(finals) == 0:
            return ""
        return finals[0]
    except Exception:
        return ""

# 判断两个字是否押韵
def normalize_final(final: str) -> str:
    if final is None:
        return ""

    final = final.lower().strip()

    final = final.replace("ü", "v")

    rhyme_groups = {
        # an 韵
        "an": {"an", "ian", "uan", "van"},

        # ang 韵
        "ang": {"ang", "iang", "uang"},

        # en/in/un/ün 可视为近韵
        "en": {"en", "in", "un", "vn"},

        # eng/ing/ong/iong 可视为近韵
        "eng": {"eng", "ing", "ong", "iong"},

        # ai/uai
        "ai": {"ai", "uai"},

        # ei/ui/uei
        "ei": {"ei", "ui", "uei"},

        # ao/iao
        "ao": {"ao", "iao"},

        # ou/iu/iou
        "ou": {"ou", "iu", "iou"},

        # a/ia/ua
        "a": {"a", "ia", "ua"},

        # o/uo
        "o": {"o", "uo"},

        # e/ie/ve/ue
        "e": {"e", "ie", "ve", "ue"},

        # er 单独一类
        "er": {"er"},
    }

    for group_name, finals in rhyme_groups.items():
        if final in finals:
            return group_name

    return final


def same_rhyme(ch1: str, ch2: str) -> bool:

    f1 = get_pinyin_final(ch1)
    f2 = get_pinyin_final(ch2)

    if f1 == "" or f2 == "":
        return False

    return normalize_final(f1) == normalize_final(f2)

# 加叠字位置约束，防止生成的诗词不通顺
def is_repeat_allowed(
    current_line_chars: List[str],
    candidate_char: str,
    line_length: int,
) -> bool:
    if len(current_line_chars) == 0:
        return True

    prev_char = current_line_chars[-1]

    # 不是叠字
    if candidate_char != prev_char:
        return True

    # 当前候选字在句内的位置
    current_pos = len(current_line_chars) + 1
    prev_pos = current_pos - 1

    # 位置约束
    if prev_pos == 3 and current_pos == 4:
        return True
    if prev_pos == 6 and current_pos == 7:
        return True

    return False

# 检查约束
def is_candidate_valid(
    candidate_char: str,
    current_line_chars: List[str],
    line_idx: int,
    pos_in_line: int,
    line_length: int,
    rhyme_char: Optional[str] = None,
    enforce_rhyme: bool = True,
) -> bool:
    # 必须是汉字
    if not is_chinese_char(candidate_char):
        return False

    # 特殊 token 不允许
    if candidate_char in ["<PAD>", "<UNK>"]:
        return False

    # 叠字位置约束
    if not is_repeat_allowed(current_line_chars, candidate_char, line_length):
        return False

    # 押韵约束
    if enforce_rhyme:
        is_last_char = pos_in_line == line_length

        if line_idx == 3 and is_last_char and rhyme_char is not None:
            if not same_rhyme(candidate_char, rhyme_char):
                return False

    return True


# 带约束的采样
@torch.no_grad()
def constrained_sample_from_logits(
    logits: torch.Tensor,
    idx2char: Dict[int, str],
    current_line_chars: List[str],
    line_idx: int,
    pos_in_line: int,
    line_length: int,
    rhyme_char: Optional[str] = None,
    enforce_rhyme: bool = True,
    temperature: float = 0.7,
    top_k: int = 20,
    max_retry_candidates: int = 300,
) -> int:
    logits = logits.float()

    if temperature > 0:
        logits = logits / temperature

    probs = torch.softmax(logits, dim=-1)
    # 只取 candidate_count 个概率最高的字
    candidate_count = min(max_retry_candidates, probs.size(-1))
    values, indices = torch.topk(probs, k=candidate_count)

    valid_indices = []
    valid_probs = []

    for prob, idx in zip(values, indices):
        idx_int = int(idx.item())
        ch = idx2char.get(idx_int, "")

        if is_candidate_valid(
            candidate_char=ch,
            current_line_chars=current_line_chars,
            line_idx=line_idx,
            pos_in_line=pos_in_line,
            line_length=line_length,
            rhyme_char=rhyme_char,
            enforce_rhyme=enforce_rhyme,
        ):
            valid_indices.append(idx_int)
            valid_probs.append(float(prob.item()))

        if top_k is not None and top_k > 0 and len(valid_indices) >= top_k:
            break

    # 约束太严格，退化
    if len(valid_indices) == 0:
        for prob, idx in zip(values, indices):
            idx_int = int(idx.item())
            ch = idx2char.get(idx_int, "")

            if is_candidate_valid(
                candidate_char=ch,
                current_line_chars=current_line_chars,
                line_idx=line_idx,
                pos_in_line=pos_in_line,
                line_length=line_length,
                rhyme_char=None,
                enforce_rhyme=False,
            ):
                valid_indices.append(idx_int)
                valid_probs.append(float(prob.item()))

            if top_k is not None and top_k > 0 and len(valid_indices) >= top_k:
                break

    if len(valid_indices) == 0:
        for idx in indices:
            idx_int = int(idx.item())
            ch = idx2char.get(idx_int, "")
            if is_chinese_char(ch):
                return idx_int

        return int(torch.argmax(logits).item())

    valid_probs_tensor = torch.tensor(valid_probs, dtype=torch.float)
    valid_probs_tensor = valid_probs_tensor / valid_probs_tensor.sum()

    # temperature <= 0：选择合法候选中概率最高的
    if temperature <= 0:
        return valid_indices[0]

    sampled_pos = torch.multinomial(valid_probs_tensor, num_samples=1).item()
    return valid_indices[sampled_pos]


# 固定格式生成函数
@torch.no_grad()
def generate_fixed_poem_constrained(
    model: nn.Module,
    start_text: str,
    char2idx: Dict[str, int],
    idx2char: Dict[int, str],
    device,
    line_length: int = 7,
    line_count: int = 4,
    temperature: float = 0.6,
    top_k: int = 8,
    enforce_rhyme: bool = True,
    show_rhyme_info: bool = True,
) -> str:
    model.eval()

    start_text = clean_poem_text(start_text)
    if len(start_text) == 0:
        start_text = "明月"

    if len(start_text) > line_length:
        start_text = start_text[:line_length]

    unk_idx = char2idx["<UNK>"]

    # 把起始文本输入模型，得到初始 hidden
    input_ids = [char2idx.get(ch, unk_idx) for ch in start_text]

    # 如果起始词完全不在词表中，回退到 <UNK>
    if len(input_ids) == 0:
        input_ids = [unk_idx]

    input_tensor = torch.LongTensor(input_ids).view(-1, 1).to(device)
    _, hidden = model(input_tensor, hidden=None)

    lines = [[] for _ in range(line_count)]

    # 把起始词放入第一句
    for ch in start_text:
        if len(lines[0]) < line_length:
            lines[0].append(ch)

    current_input_id = input_ids[-1]
    current_input = torch.LongTensor([[current_input_id]]).to(device)

    rhyme_char = None

    for line_idx in range(line_count):
        while len(lines[line_idx]) < line_length:
            pos_in_line = len(lines[line_idx]) + 1

            logits, hidden = model(current_input, hidden)
            next_logits = logits[-1, 0]

            next_id = constrained_sample_from_logits(
                logits=next_logits,
                idx2char=idx2char,
                current_line_chars=lines[line_idx],
                line_idx=line_idx,
                pos_in_line=pos_in_line,
                line_length=line_length,
                rhyme_char=rhyme_char,
                enforce_rhyme=enforce_rhyme,
                temperature=temperature,
                top_k=top_k,
            )

            next_char = idx2char.get(next_id, "")

            if not is_chinese_char(next_char):
                continue

            lines[line_idx].append(next_char)
            current_input = torch.LongTensor([[next_id]]).to(device)

        # 第二句末字作为韵脚
        if enforce_rhyme and line_idx == 1:
            rhyme_char = lines[line_idx][-1]

    # 格式化输出
    if line_count == 4:
        poem = (
            "".join(lines[0]) + "，" +
            "".join(lines[1]) + "。\n" +
            "".join(lines[2]) + "，" +
            "".join(lines[3]) + "。"
        )
    else:
        parts = []
        for i, line in enumerate(lines):
            punct = "，" if i % 2 == 0 else "。"
            parts.append("".join(line) + punct)
        poem = "".join(parts)

    # 韵脚检查
    if (
        show_rhyme_info
        and enforce_rhyme
        and line_count >= 4
        and HAS_PYPINYIN
    ):
        r2 = lines[1][-1]
        r4 = lines[3][-1]
        f2 = get_pinyin_final(r2)
        f4 = get_pinyin_final(r4)
        poem += f"\n韵脚检查：第2句「{r2}」({f2})，第4句「{r4}」({f4})"

    return poem


# 训练函数
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    loss_fn,
    device,
    grad_clip: float = 5.0,
    log_interval: int = 100,
    epoch: int = 1,
    total_epochs: int = 1,
) -> float:
    model.train()

    total_loss = 0.0
    total_batches = 0

    for batch_idx, (x, y) in enumerate(dataloader, start=1):
        x = x.to(device)
        y = y.to(device)

        # RNN 输入：[seq_len, batch]
        x = x.transpose(0, 1)

        optimizer.zero_grad()
        # 前向传播
        logits, _ = model(x)

        logits_for_loss = logits.transpose(0, 1).transpose(1, 2)
        # 计算 loss
        loss = loss_fn(logits_for_loss, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip,
        )
        # 反向传播和梯度裁剪，防止梯度爆炸
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if batch_idx % log_interval == 0:
            print(
                f"Epoch [{epoch}/{total_epochs}], "
                f"Step [{batch_idx}/{len(dataloader)}], "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / max(total_batches, 1)
    return avg_loss


# 保存词表
def save_vocab(
    char2idx: Dict[str, int],
    idx2char: Dict[int, str],
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, "vocab.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "char2idx": char2idx,
                "idx2char": {str(k): v for k, v in idx2char.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"词表保存到：{path}")


def save_loss_curve(losses: List[float], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    loss_json = os.path.join(out_dir, "loss_history.json")
    with open(loss_json, "w", encoding="utf-8") as f:
        json.dump(losses, f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"Loss 曲线保存到：{fig_path}")


# 训练流程
def train(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print(f"当前使用设备：{device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # 读取并清洗数据
    poems = load_json_poems(
        data_dir=args.data_dir,
        min_len=args.min_poem_len,
    )

    corpus_path = os.path.join(args.out_dir, "clean_poems.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for p in poems:
            f.write(p + "\n")


    # 构建词表
    char2idx, idx2char, vocab = build_vocab(
        poems=poems,
        min_freq=args.min_freq,
    )

    save_vocab(char2idx, idx2char, args.out_dir)

    # 转索引序列
    sequences = poems_to_sequences(poems, char2idx)

    # 构造滑动窗口数据集
    dataset = PoetryDataset(
        sequences=sequences,
        seq_len=args.seq_len,
        stride=args.stride,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # 模型
    model = PoetryRNNModel(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        model_type=args.model_type,
        dropout=args.dropout,
    ).to(device)

    print("\n模型结构：")
    print(model)

    # 失和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    best_loss = float("inf")

    # 训练
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        losses.append(avg_loss)

        print(f"\n===== Epoch {epoch} Average Loss: {avg_loss:.4f} =====")

        demo_poem = generate_fixed_poem_constrained(
            model=model,
            start_text=args.start_text,
            char2idx=char2idx,
            idx2char=idx2char,
            device=device,
            line_length=args.line_length,
            line_count=args.line_count,
            temperature=args.temperature,
            top_k=args.top_k,
            enforce_rhyme=args.enforce_rhyme,
            show_rhyme_info=True,
        )

        print("【生成演示】")
        print(demo_poem)
        print()

        ckpt = {
            "model_state_dict": model.state_dict(),
            "char2idx": char2idx,
            "idx2char": idx2char,
            "vocab": vocab,
            "args": vars(args),
            "epoch": epoch,
            "loss": avg_loss,
        }

        last_path = os.path.join(args.out_dir, "last_model.pth")
        torch.save(ckpt, last_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.out_dir, "best_model.pth")
            torch.save(ckpt, best_path)
            print(f"当前为最佳模型，已保存到：{best_path}")

    # 保存 loss 曲线
    save_loss_curve(losses, args.out_dir)

    # 保存最终生成结果
    final_poems = []

    for temp in [0.3, 0.5, 0.7, 1.0]:
        poem = generate_fixed_poem_constrained(
            model=model,
            start_text=args.start_text,
            char2idx=char2idx,
            idx2char=idx2char,
            device=device,
            line_length=args.line_length,
            line_count=args.line_count,
            temperature=temp,
            top_k=args.top_k,
            enforce_rhyme=args.enforce_rhyme,
            show_rhyme_info=True,
        )

        final_poems.append(f"temperature={temp}\n{poem}\n")

    result_path = os.path.join(args.out_dir, "generated_poems.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(final_poems))

    print("\n训练完成。")
    print(f"输出目录：{args.out_dir}")
    print(f"最终生成结果保存到：{result_path}")


# 训练完成加载模型生成古诗
def load_checkpoint_and_generate(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print(f"当前使用设备：{device}")

    ckpt = torch.load(args.checkpoint, map_location=device)

    char2idx = ckpt["char2idx"]
    idx2char_raw = ckpt["idx2char"]

    idx2char = {}
    for k, v in idx2char_raw.items():
        idx2char[int(k)] = v

    old_args = ckpt.get("args", {})

    vocab_size = len(char2idx)

    model = PoetryRNNModel(
        vocab_size=vocab_size,
        embedding_dim=old_args.get("embedding_dim", args.embedding_dim),
        hidden_size=old_args.get("hidden_size", args.hidden_size),
        num_layers=old_args.get("num_layers", args.num_layers),
        model_type=old_args.get("model_type", args.model_type),
        dropout=old_args.get("dropout", args.dropout),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"已加载模型：{args.checkpoint}")

    for temp in args.generate_temperatures:
        poem = generate_fixed_poem_constrained(
            model=model,
            start_text=args.start_text,
            char2idx=char2idx,
            idx2char=idx2char,
            device=device,
            line_length=args.line_length,
            line_count=args.line_count,
            temperature=temp,
            top_k=args.top_k,
            enforce_rhyme=args.enforce_rhyme,
            show_rhyme_info=True,
        )

        print("=" * 50)
        print(f"temperature = {temp}")
        print(poem)


# 命令行参数
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "generate"],
        help="train 表示训练；generate 表示加载 checkpoint 单独生成",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="mode=generate 时使用的模型路径，例如 outputs/best_model.pth",
    )

    # 数据路径
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset",
        help="存放 poet.song*.json 或 poet.tang*.json 的文件夹",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="./outputs_constrained_poem_gru",
        help="输出目录",
    )

    # 数据参数
    parser.add_argument(
        "--min_poem_len",
        type=int,
        default=20,
        help="清洗后诗歌最小长度",
    )

    parser.add_argument(
        "--min_freq",
        type=int,
        default=1,
        help="词表最小字频",
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=32,
        help="滑动窗口序列长度",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="滑动窗口步长",
    )

    # 模型参数
    parser.add_argument(
        "--model_type",
        type=str,
        default="lstm",
        choices=["rnn", "gru", "lstm"],
        help="循环网络类型",
    )

    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )

    # 训练参数
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--grad_clip",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
    )

    # 生成参数
    parser.add_argument(
        "--start_text",
        type=str,
        default="明月",
        help="起始词",
    )

    parser.add_argument(
        "--line_length",
        type=int,
        default=7,
        help="每句字数，7 表示七言，5 表示五言",
    )

    parser.add_argument(
        "--line_count",
        type=int,
        default=4,
        help="生成句数，4 表示绝句",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="生成温度，越低越保守",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=8,
        help="每步从前 k 个合法候选中采样",
    )

    parser.add_argument(
        "--enforce_rhyme",
        action="store_true",
        help="启用押韵约束：第4句末字与第2句末字押韵",
    )

    parser.add_argument(
        "--generate_temperatures",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7, 1.0],
        help="mode=generate 时测试的多个温度",
    )

    # 其他
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="强制使用 CPU",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "generate":
        if args.checkpoint == "":
            raise ValueError("mode=generate 时必须指定 --checkpoint")
        load_checkpoint_and_generate(args)