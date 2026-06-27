# %%
# imports

import torch
import torch.nn as nn
from torch.nn import Dropout, functional as F


# %%
# setup
with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read()
device = "cuda" if torch.cuda.is_available() else "cpu"
len(text)
text[:100]

# %%
# chars

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])


# %%
# encode text

data = torch.tensor(encode(text), dtype=torch.long)
data.shape
data.dtype
data[:100]

# %%
# 0
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
BLOCK_SIZE = 8
BATCH_SIZE = 4
n_embd = 32

# %%
# 0


def get_batch(split):
    data = train_data if split == "train" else val_data
    data = data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x, y


xb, yb = get_batch("train")
for batch in range(BATCH_SIZE):
    for t in range(BLOCK_SIZE):
        context = xb[batch, : t + 1]
        target = yb[batch, t]

# %%
# 0
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
x.shape

# %%
# 0

xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1]
        xbow[b, t] = torch.mean(xprev, 0)

avg_mul = torch.tril(torch.ones(T, T))
avg_mul /= avg_mul.sum(dim=1, keepdim=True)
xbow2 = avg_mul @ x
# %%
# 0
torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)
q = query(x)
wei = q @ k.transpose(-2, -1) * head_size**-0.5


tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=1)
v = value(x)
out = wei @ v
out[0]
q.var()
k.var()
wei.var()

# %%
# 0


BATCH_SIZE = 64
BLOCK_SIZE = 256
N_EMBD = 384
N_HEAD = 6
MAX_ITERS = 1000
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_LAYERS = 6
dropout = 0.2


class ChadgramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.pos_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.linear_head = nn.Linear(N_EMBD, vocab_size)
        self.attention_head = Head(N_EMBD)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_embedding_table(torch.arange(T))
        x = token_emb + pos_emb
        x = self.attention_head(x)
        logits = self.linear_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -BLOCK_SIZE:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# %%
# 0
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(in_features=N_EMBD, out_features=head_size, bias=False)
        self.query = nn.Linear(in_features=N_EMBD, out_features=head_size, bias=False)
        self.value = nn.Linear(in_features=N_EMBD, out_features=head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.mask: torch.Tensor
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.key(x)
        v = self.key(x)
        wei = k @ q.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=2)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, N_EMBD):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(N_EMBD * 4, N_EMBD),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True, unbiased=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.self_attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class MoggeramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.pos_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.classifer = nn.Linear(N_EMBD, vocab_size)
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYERS)])
        self.layer_norm = nn.LayerNorm(N_EMBD)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        idx = idx.to(device)
        if targets is not None:
            targets = targets.to(device)
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.classifer(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -BLOCK_SIZE:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# %%
# 0

moggeram_v0 = MoggeramLanguageModel().to(device)
optimiser = torch.optim.AdamW(params=moggeram_v0.parameters(), lr=LR)
# %%
# 0
for epoch in range(MAX_ITERS):
    xb, yb = get_batch("train")
    _, loss = moggeram_v0(xb.to(device), yb.to(device))
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    if epoch % 5 == 0:
        print(f"Train: {loss.item()}")
        with torch.inference_mode():
            moggeram_v0.eval()
            xt, yt = get_batch("val")
            _, loss = moggeram_v0(xt, yt)
            print(f"Val: {loss.item()}")


# %%
# 0
itos = {i: ch for i, ch in enumerate(chars)}


def decoder(lst):
    lst = list(lst.squeeze())
    out = "".join([itos[int(i)] for i in lst])
    return out


out = moggeram_v0.generate(
    torch.zeros(size=(1, 1), dtype=torch.long, device=device), 1000
)
out = decoder(out)
print(out)
