import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import GPTDataset, build_vocab, load_text, text_to_tensor
from minigpt import MiniGPT


def train(
    text_path="tiny_shakespeare.txt",
    checkpoint_path="MiniGPT_Tiny_Shakespeare.pth",
    block_size=256,
    batch_size=256,
    epochs=1,
    lr=5e-4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = load_text(text_path)
    chars, stoi, _ = build_vocab(text)
    data = text_to_tensor(text, stoi)
    loader = DataLoader(GPTDataset(data, block_size), batch_size=batch_size, shuffle=True)

    model = MiniGPT(len(chars)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, len(chars)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx > 0 and batch_idx % 100 == 0:
                print(f"epoch {epoch + 1} batch {batch_idx}/{len(loader)} loss {loss.item():.4f}")

    torch.save(model.state_dict(), checkpoint_path)
    return model


if __name__ == "__main__":
    train()
