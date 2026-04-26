import torch
import torch.nn as nn


class DecoderBlock(nn.Module):
    def __init__(self, nhead, d_model):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, attention_mask):
        x_normalized = self.ln1(x)
        attn_out = self.attn(
            x_normalized,
            x_normalized,
            x_normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )[0]
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=384, num_layers=4, nhead=6):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(1024, d_model)
        self.layers = nn.ModuleList(
            [DecoderBlock(nhead, d_model) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        _, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)

        attention_mask = torch.triu(
            torch.ones(t, t, device=idx.device), diagonal=1
        ).bool()

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.ln_f(x)
        return self.head(x)
