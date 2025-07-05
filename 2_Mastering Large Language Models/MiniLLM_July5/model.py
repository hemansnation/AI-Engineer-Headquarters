import torch
import torch.nn as nn
import math

class GPTConfig:
    def __init__(self, vocab_size=50257, seq_len=128, embed_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, embed_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(config.seq_len, config.seq_len)).view(1, 1, config.seq_len, config.seq_len))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(self.embed_dim, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        out = self.proj(out)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, 4 * config.embed_dim)
        self.fc2 = nn.Linear(4 * config.embed_dim, config.embed_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_enc = PositionalEncoding(config.embed_dim, config.seq_len, config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x, targets=None):
        x = self.token_emb(x)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

class GPTClassifier(nn.Module):
    def __init__(self, gpt_model, num_classes=2):
        super().__init__()
        self.gpt = gpt_model
        self.classifier = nn.Linear(gpt_model.config.embed_dim, num_classes)
        # Freeze GPT layers
        for param in self.gpt.parameters():
            param.requires_grad = False
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, labels=None):
        # Get last token's embedding
        x = self.gpt.token_emb(x)
        x = self.gpt.pos_enc(x)
        for block in self.gpt.blocks:
            x = block(x)
        x = self.gpt.ln_f(x)
        cls_embedding = x[:, -1, :]  # Use last token
        logits = self.classifier(cls_embedding)
        if labels is None:
            return logits, None
        loss = nn.CrossEntropyLoss()(logits, labels)
        return logits, loss