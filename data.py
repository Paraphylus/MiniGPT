from pathlib import Path

import torch
from torch.utils.data import Dataset


DEFAULT_CHARS = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def load_text(path="tiny_shakespeare.txt"):
    return Path(path).read_text(encoding="utf-8")


def build_vocab(text=None):
    chars = sorted(set(text)) if text is not None else list(DEFAULT_CHARS)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, stoi, itos


def encode(text, stoi):
    missing = sorted({ch for ch in text if ch not in stoi})
    if missing:
        raise ValueError(f"Unsupported characters: {missing}")
    return [stoi[ch] for ch in text]


def decode(token_ids, itos):
    return "".join(itos[int(i)] for i in token_ids)


class GPTDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def text_to_tensor(text, stoi):
    return torch.tensor(encode(text, stoi), dtype=torch.long)
