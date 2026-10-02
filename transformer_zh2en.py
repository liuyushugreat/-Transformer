# -*- coding: utf-8 -*-
"""
从零实现 Transformer（Vaswani et al., 2017），完成中译英：
    我要吃苹果  ->  i want to eat an apple
依赖：Python 3.8+，PyTorch 1.12+。CPU 上约 1 分钟训练完毕。
运行：python transformer_zh2en.py
"""
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 第 1 部分：数据 —— 一个迷你中英平行语料
# ============================================================
# 中文按“词”切分（用空格分隔，等价于做好了分词），英文按空格切分。
PAIRS = [
    ("我 要 吃 苹果",   "i want to eat an apple"),
    ("我 要 吃 香蕉",   "i want to eat a banana"),
    ("我 要 喝 水",     "i want to drink water"),
    ("我 要 喝 牛奶",   "i want to drink milk"),
    ("你 要 吃 苹果",   "you want to eat an apple"),
    ("你 要 喝 水",     "you want to drink water"),
    ("他 要 吃 香蕉",   "he wants to eat a banana"),
    ("他 要 喝 牛奶",   "he wants to drink milk"),
    ("她 要 吃 苹果",   "she wants to eat an apple"),
    ("她 要 喝 水",     "she wants to drink water"),
    ("我 喜欢 苹果",    "i like apples"),
    ("我 喜欢 香蕉",    "i like bananas"),
    ("你 喜欢 牛奶",    "you like milk"),
    ("他 喜欢 苹果",    "he likes apples"),
    ("她 喜欢 水",      "she likes water"),
    ("我 吃 苹果",      "i eat an apple"),
    ("你 吃 香蕉",      "you eat a banana"),
    ("他 吃 苹果",      "he eats an apple"),
    ("她 喝 牛奶",      "she drinks milk"),
    ("我 不 要 吃 苹果", "i do not want to eat an apple"),
    ("我 不 要 喝 水",  "i do not want to drink water"),
    ("你 不 要 吃 香蕉", "you do not want to eat a banana"),
]

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]


class Vocab:
    """词表：token <-> id 的双向映射"""

    def __init__(self, sentences):
        self.itos = list(SPECIALS)
        for s in sentences:
            for tok in s.split():
                if tok not in self.itos:
                    self.itos.append(tok)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, sentence, add_bos_eos=True):
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in sentence.split()]
        if add_bos_eos:
            ids = [self.stoi[BOS]] + ids + [self.stoi[EOS]]
        return ids

    def decode(self, ids):
        toks = []
        for i in ids:
            t = self.itos[i]
            if t == EOS:
                break
            if t not in (PAD, BOS):
                toks.append(t)
        return " ".join(toks)


SRC_VOCAB = Vocab([p[0] for p in PAIRS])
TGT_VOCAB = Vocab([p[1] for p in PAIRS])
PAD_IDX = 0  # 两个词表中 <pad> 都是 0


def make_batch(pairs):
    """把若干句子对补齐成同长度张量"""
    src = [SRC_VOCAB.encode(s) for s, _ in pairs]
    tgt = [TGT_VOCAB.encode(t) for _, t in pairs]
    max_s, max_t = max(map(len, src)), max(map(len, tgt))
    src = [x + [PAD_IDX] * (max_s - len(x)) for x in src]
    tgt = [x + [PAD_IDX] * (max_t - len(x)) for x in tgt]
    return torch.tensor(src, device=DEVICE), torch.tensor(tgt, device=DEVICE)


# ============================================================
# 第 2 部分：模型组件
# ============================================================
class TokenEmbedding(nn.Module):
    """词嵌入，并按论文乘以 sqrt(d_model)"""

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.d_model = d_model

    def forward(self, x):                       # x: (B, L)
        return self.emb(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """正弦位置编码：PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(...)"""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()           # (max_len, 1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))           # (d_model/2,)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))                # (1, max_len, d)

    def forward(self, x):                       # x: (B, L, d)
        return self.dropout(x + self.pe[:, : x.size(1)])


def scaled_dot_product_attention(q, k, v, mask=None):
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    q,k,v: (B, h, L, d_k)；mask: 可广播到 (B, h, Lq, Lk)，True 表示“允许看”
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)              # (B,h,Lq,Lk)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ v, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn = None  # 保存最近一次注意力权重，便于可视化

    def _split(self, x):                        # (B,L,d) -> (B,h,L,d_k)
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        q = self._split(self.w_q(query))
        k = self._split(self.w_k(key))
        v = self._split(self.w_v(value))
        out, self.attn = scaled_dot_product_attention(q, k, v, mask)
        B, _, L, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(B, L, self.h * self.d_k)
        return self.dropout(self.w_o(out))


class FeedForward(nn.Module):
    """逐位置前馈网络：FFN(x) = W2 · ReLU(W1 x + b1) + b2"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    """Post-LN（与原论文一致）：x = LN(x + Sublayer(x))"""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        x = self.ln1(x + self.self_attn(x, x, x, src_mask))
        x = self.ln2(x + self.ffn(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)   # 带因果掩码
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)  # 看编码器
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, y, memory, tgt_mask, src_mask):
        y = self.ln1(y + self.self_attn(y, y, y, tgt_mask))
        y = self.ln2(y + self.cross_attn(y, memory, memory, src_mask))
        y = self.ln3(y + self.ffn(y))
        return y


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, n_heads=4,
                 n_layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        self.src_emb = TokenEmbedding(src_vocab, d_model)
        self.tgt_emb = TokenEmbedding(tgt_vocab, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.generator = nn.Linear(d_model, tgt_vocab)   # 输出投影到英文词表
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ---------- 掩码 ----------
    @staticmethod
    def make_src_mask(src):                     # (B,Ls) -> (B,1,1,Ls)
        return (src != PAD_IDX).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_tgt_mask(tgt):                     # (B,Lt) -> (B,1,Lt,Lt)
        Lt = tgt.size(1)
        pad = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)                    # (B,1,1,Lt)
        causal = torch.tril(torch.ones(Lt, Lt, dtype=torch.bool, device=tgt.device))
        return pad & causal                                                   # 下三角

    # ---------- 前向 ----------
    def encode(self, src, src_mask):
        x = self.pos(self.src_emb(src))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, tgt_mask, src_mask):
        y = self.pos(self.tgt_emb(tgt))
        for layer in self.decoder:
            y = layer(y, memory, tgt_mask, src_mask)
        return y

    def forward(self, src, tgt_in):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt_in)
        memory = self.encode(src, src_mask)
        out = self.decode(tgt_in, memory, tgt_mask, src_mask)
        return self.generator(out)              # (B, Lt, |V_tgt|) 的 logits


# ============================================================
# 第 3 部分：训练
# ============================================================
def train(model, epochs=300, batch_size=8, lr=5e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    model.train()
    for ep in range(1, epochs + 1):
        random.shuffle(PAIRS)
        total = 0.0
        for i in range(0, len(PAIRS), batch_size):
            src, tgt = make_batch(PAIRS[i:i + batch_size])
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]   # Teacher forcing：右移一位
            logits = model(src, tgt_in)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        if ep % 50 == 0 or ep == 1:
            print(f"epoch {ep:4d} | loss {total / math.ceil(len(PAIRS) / batch_size):.4f}")


# ============================================================
# 第 4 部分：推理（贪心解码）
# ============================================================
@torch.no_grad()
def translate(model, zh_sentence, max_len=20, verbose=False):
    model.eval()
    src = torch.tensor([SRC_VOCAB.encode(zh_sentence)], device=DEVICE)
    src_mask = model.make_src_mask(src)
    memory = model.encode(src, src_mask)                     # 编码器只跑一次
    ys = torch.tensor([[TGT_VOCAB.stoi[BOS]]], device=DEVICE)
    for step in range(max_len):
        out = model.decode(ys, memory, model.make_tgt_mask(ys), src_mask)
        prob = torch.softmax(model.generator(out[:, -1]), dim=-1)
        p, next_id = prob.max(dim=-1)
        if verbose:
            print(f"  step {step + 1}: 已生成 {TGT_VOCAB.decode(ys[0].tolist())!r:32s}"
                  f" -> 预测 {TGT_VOCAB.itos[next_id.item()]!r} (p={p.item():.3f})")
        ys = torch.cat([ys, next_id.view(1, 1)], dim=1)
        if next_id.item() == TGT_VOCAB.stoi[EOS]:
            break
    return TGT_VOCAB.decode(ys[0].tolist())


def show_cross_attention(model, zh_sentence):
    """打印最后一层解码器的交叉注意力（多头平均），看每个英文词“对齐”到哪个中文词"""
    en = translate(model, zh_sentence)
    src = torch.tensor([SRC_VOCAB.encode(zh_sentence)], device=DEVICE)
    tgt = torch.tensor([TGT_VOCAB.encode(en)[:-1]], device=DEVICE)   # <bos> + 译文
    with torch.no_grad():
        model(src, tgt)
    attn = model.decoder[-1].cross_attn.attn[0].mean(0)              # (Lt, Ls)
    src_toks = [BOS] + zh_sentence.split() + [EOS]
    out_toks = en.split() + [EOS]
    print("\n交叉注意力（行=生成的英文词，列=中文源词）")
    print(" " * 10 + "".join(f"{t:>8s}" for t in src_toks))
    for i, tok in enumerate(out_toks):
        print(f"{tok:>10s}" + "".join(f"{w:8.2f}" for w in attn[i].tolist()))


if __name__ == "__main__":
    print(f"中文词表 {len(SRC_VOCAB)} 个 token，英文词表 {len(TGT_VOCAB)} 个 token")
    model = Transformer(len(SRC_VOCAB), len(TGT_VOCAB)).to(DEVICE)
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

    train(model)

    print("\n===== 翻译：我要吃苹果 =====")
    result = translate(model, "我 要 吃 苹果", verbose=True)
    print(f"\n我要吃苹果  ->  {result}")

    show_cross_attention(model, "我 要 吃 苹果")

    print("\n===== 其他测试 =====")
    for s in ["她 要 吃 苹果", "我 不 要 喝 牛奶", "他 要 喝 水"]:
        print(f"{s.replace(' ', ''):8s} -> {translate(model, s)}")
