import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import List, Tuple, Dict

class LionHeart:
    def __init__(self, vocab_size: int = 50000, n_embd: int = 768, n_head: int = 12, 
                 n_layer: int = 12, block_size: int = 1024, dropout: float = 0.1):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.dropout = dropout

class MultiHeadAttention(nn.Module):
    def __init__(self, config: LionHeart):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.head_dim))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class FeedForward(nn.Module):
    def __init__(self, config: LionHeart):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config: LionHeart):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self, config: LionHeart):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def load_data(file_path: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return list(text), stoi, itos

def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi[c] for c in text]

def decode(ids: List[int], itos: Dict[int, str]) -> str:
    return ''.join([itos[i] for i in ids])

def get_batch(data: torch.Tensor, block_size: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def train(model: nn.Module, data: torch.Tensor, config: LionHeart, 
          lr: float = 3e-4, max_iters: int = 10000, eval_interval: int = 500, 
          batch_size: int = 64) -> None:
    optimizer = Adam(model.parameters(), lr=lr)
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, data, config, eval_iters=200)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch(data, config.block_size, batch_size)
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

@torch.no_grad()
def estimate_loss(model: nn.Module, data: torch.Tensor, config: LionHeart, eval_iters: int = 200) -> Dict[str, float]:
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, config.block_size, batch_size=32)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def save_model(model: nn.Module, config: LionHeart, vocab: Dict[str, int], save_dir: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_path = os.path.join(save_dir, 'language_model.pth')
    config_path = os.path.join(save_dir, 'model_config.pkl')
    vocab_path = os.path.join(save_dir, 'vocab.pkl')
    
    torch.save(model.state_dict(), model_path)
    
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    print(f"Model saved in {save_dir}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and preprocess data
    text, stoi, itos = load_data('./data/training_wiki_data.txt')
    data = torch.tensor(encode(text, stoi), dtype=torch.long, device=device)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Initialize model
    config = LionHeart(vocab_size=len(stoi), n_embd=384, n_head=6, n_layer=6, block_size=256)
    model = LanguageModel(config).to(device)

    # Train model
    train(model, train_data, config)

    # Save the model
    save_model(model, config, stoi, './models')

    # Generate text
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=10)
    print(decode(generated[0].tolist(), itos))

if __name__ == "__main__":
    main()
