import random
from dataclasses import dataclass,replace

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

########################
# Config
########################


@dataclass
class Config:
    N: int = 2
    d_model: int = 128
    d_ff: int = 512
    h: int = 4
    d_k: int = 32
    d_v: int = 32
    P_drop: float = 0.1

    # training
    eps_ls: float = 0.1
    beta1: float = 0.1
    beta2: float = 0.9
    adam_eps: float = 1e-9
    warmup_steps: int = 400
    tokens_per_batch: int = 1500

    epochs: int = 20

    # decoding
    beam_width: int = 4
    alpha: float = 0.6
    max_extra_len: int = 50

    # data
    task: str = "reverse"  # "reverse" | "sort"
    num_symbols: int = 20
    min_len: int = 3
    max_len: int = 12
    train_pairs: int = 10000
    valid_pairs: int = 300
    max_positions: int = 512

    overfit_steps: int = 400
    seed: int = 0


########################
# Data
########################

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]


class Vocab:
    def __init__(self, tokens):
        self.itos = list(SPECIALS) + sorted(set(tokens) - set(SPECIALS))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, UNK) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]


def symbols(cfg):
    return [chr(ord("a") + i) for i in range(cfg.num_symbols)]


def make_pair(rng, cfg, syms):
    syms = symbols(cfg)
    rng = random.Random(cfg.seed)
    train = [make_pair(rng, cfg, syms) for _ in range(cfg.train_pairs)]
    rng_valid = random.Random(cfg.seed + 10_000)
    valid = [make_pair(rng_valid, cfg, syms) for _ in range(cfg.valid_pairs)]
    vocab = Vocab(syms)
    return train, valid, vocab


def tensorize(pair, vocab):
    src_tokens, tgt_tokens = pair
    src_ids = vocab.encode(src_tokens) + [EOS]
    tgt_ids = [SOS] + vocab.encode(tgt_tokens) + [EOS]
    return src_ids, tgt_ids


def collate(chunk):
    """pads a list of (src_ids, tgt_ids) into src (B, T_x) and tgt (B, T_y)."""
    src_len = max(len(s) for s, _ in chunk)
    tgt_len = max(len(t) for _, t in chunk)
    src = torch.full((len(chunk), src_len), PAD, dtype=torch.long)
    tgt = torch.full((len(chunk), tgt_len), PAD, dtype=torch.long)
    for i, (s, t) in enumerate(chunk):
        src[i, : len(s)] = torch.tensor(s)
        tgt[i, : len(t)] = torch.tensor(t)
    return src, tgt


def make_batches(data, cfg):
    data = sorted(data, key=lambda x: (len(x[0]), len(x[1])))
    batches, cur, longest = [], [], 0
    for pair in data:
        new_longest = max(longest, len(pair[0]), len(pair[1]))
        if cur and new_longest * (len(cur) + 1) > cfg.tokens_per_batch:
            batches.append(cur)
            cur, new_longest = [], max(len(pair[0]), len(pair[1]))
        cur.append(pair)
        longest = new_longest
    if cur:
        batches.append(cur)
    return batches


def iterate_batches(batches, rng):
    order = list(range(len(batches)))
    rng.shuffle(order)  # batch order reshuffled every epoch
    for i in order:
        yield collate(batches[i])


########################
# Masks
########################


def make_src_mask(src):
    return (src != PAD)[:, None, None, :]


def make_tgt_mask(tgt):
    T = tgt.size(1)
    pad = (tgt != PAD)[:, None, None, :]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tgt.device))
    return pad & causal


############################
# Model
############################


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ V, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        self.h, self.d_k, self.d_v = cfg.h, cfg.d_k, cfg.d_v
        self.W_Q = nn.Linear(cfg.d_model, cfg.h * cfg.d_k, bias=False)
        self.W_K = nn.Linear(cfg.d_model, cfg.h * cfg.d_k, bias=False)
        self.W_V = nn.Linear(cfg.d_model, cfg.h * cfg.d_v, bias=False)
        self.W_O = nn.Linear(cfg.h * cfg.d_v, cfg.d_model, bias=False)
        self.attn = None

    def forward(self, q_in, kv_in, mask):
        B, T_q = q_in.shape
        T_k = kv_in.size(1)
        Q = self.W_Q(q_in).view(B, T_q, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(kv_in).view(B, T_k, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(kv_in).view(B, T_k, self.h, self.d_v).transpose(1, 2)
        out, weights = scaled_dot_product_attention(Q, K, V, mask)
        self.attn = weights.detach()

        out = out.transpose(1, 2).reshape(
            B, T_q, self.h * self.d_v
        )  # Concat(head_1..head_h)
        return self.W_O(out)


class PositionwiseFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.W_1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.W_2 = nn.Linear(cfg.d_ff, cfg.d_model)

    def forward(self, x):
        return self.W_2(F.relu(self.W_1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        pos = torch.arange(cfg.max_positions, dtype=torch.float).unsqueeze(1)  # (P, 1)
        two_i = torch.arange(0, cfg.d_model, 2, dtype=torch.float)  # (d/2,)
        div = torch.pow(10000.0, two_i / cfg.d_model)
        pe = torch.zeros(cfg.max_positions, cfg.d_model)
        pe[:, 0::2] = torch.sin(pos / div)
        pe[:, 1::2] = torch.cos(pos / div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class SublayerConnection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.P_drop)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg)
        self.ffn = PositionwiseFFN(cfg)
        self.sub1 = SublayerConnection(cfg)
        self.sub2 = SublayerConnection(cfg)

    def forward(self, x, src_mask):
        x = self.sub1(x, lambda y: self.self_attn(y, y, src_mask))
        return self.sub2(x, self.ffn)


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg)
        self.cross_attn = MultiHeadAttention(cfg)
        self.ffn = PositionwiseFFN(cfg)
        self.sub1 = SublayerConnection(cfg)
        self.sub2 = SublayerConnection(cfg)
        self.sub3 = SublayerConnection(cfg)

    def forward(self, y, memory, src_mask, tgt_mask):
        y = self.sub1(y, lambda z: self.self_attn(z, z, tgt_mask))
        y = self.sub2(y, lambda z: self.cross_attn(z, memory, src_mask))
        return self.sub3(y, self.ffn)


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.N)])

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.N)])

    def forward(self, y, memory, src_mask, tgt_mask):
        for layer in self.layers:
            y = layer(y, memory, src_mask, tgt_mask)
        return y


class Transformer(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(vocab_size, cfg.d_model, padding_idx=PAD)
        self.pe = PositionalEncoding(cfg)
        self.emb_dropout = nn.Dropout(cfg.P_drop)
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.generator = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.generator.weight = self.embedding.weight  # tied
        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if name == "embedding.weight":
                nn.init.normal_(p, 0.0, self.cfg.d_model**-0.5)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)
        with torch.no_grad():
            self.embedding.weight[PAD].zero_()

    def embed(self, tokens):
        x = self.embedding(tokens) * math.sqrt(self.cfg.d_model)
        return self.emb_dropout(self.pe(x))

    def encode(self, src, src_mask):
        return self.encoder(self.embed(src), src_mask)

    def decode(self, tgt_in, memory, src_mask, tgt_mask):
        return self.decoder(
            self.embed(tgt_in), memory, src_mask, tgt_mask
        )  # (B, T_y, d_model)

    def forward(self, src, tgt_in):
        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_in)
        memory = self.encode(src, src_mask)
        out = self.decode(tgt_in, memory, src_mask, tgt_mask)
        return self.generator(out)  # logits (B, T_y, V); softmax lives in the loss


#######################
# Training
#######################


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, eps):
        super().__init__()
        self.V, self.eps = vocab_size, eps

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)  # (N, V)
        q = torch.full_like(log_probs, self.eps / (self.V - 2))
        q.scatter_(1, target.unsqueeze(1), 1.0 - self.eps)
        q[:, PAD] = 0.0
        keep = target != PAD
        smoothed = -(q * log_probs).sum(-1)[keep].sum()
        nll = F.nll_loss(log_probs, target, ignore_index=PAD, reduction="sum")
        return smoothed, nll


def lrate(step, cfg):
    step = max(step, 1)
    return cfg.d_model**-0.5 * min(step**-0.5, step * cfg.warmup_steps**-1.5)


def compute_loss(model, src, tgt, criterion):
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    logits = model(src, tgt_in)
    smoothed, nll = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
    n_tokens = (tgt_out != PAD).sum().item()
    return smoothed / n_tokens, nll.item(), n_tokens


def train(model, pairs, cfg, vocab, epochs, log_every=1, valid=None):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(cfg.beta1, cfg.beta2), eps=cfg.adam_eps
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lrate(s + 1, cfg)
    )
    criterion = LabelSmoothingLoss(len(vocab), cfg.eps_ls)
    data = [tensorize(p, vocab) for p in pairs]
    batches = make_batches(data, cfg)
    rng = random.Random(cfg.seed)
    step, start = 0, time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_nll, total_tokens = 0.0, 0
        for src, tgt in iterate_batches(batches, rng):
            optimizer.zero_grad()
            loss, nll, n_tokens = compute_loss(model, src, tgt, criterion)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            total_nll += nll
            total_tokens += n_tokens

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            per_token = total_nll / total_tokens
            msg = (
                f"  epoch {epoch:4d}  step {step:5d}  lr {scheduler.get_last_lr()[0]:.2e}  "
                f"nll/token {per_token:6.4f}  ppl {math.exp(per_token):7.3f}"
            )
            if valid is not None:
                msg += f"  valid greedy {exact_match(model, valid, cfg, vocab, greedy_decode):.1%}"
            print(msg + f"  ({time.time() - start:.0f}s)")
    return model


############################
# Decoding
############################


@torch.no_grad()
def greedy_decode(model, src, cfg):
    model.eval()
    src_mask = make_src_mask(src)
    memory = model.encode(src, src_mask)
    ys = torch.tensor([[SOS]])
    for _ in range(src.size(1) + cfg.max_extra_len):
        out = model.decode(ys, memory, src_mask, make_tgt_mask(ys))
        next_tok = model.generator(out[:, -1]).argmax(-1)
        if next_tok.item() == EOS:
            break
        ys = torch.cat([ys, next_tok.sequence(0)], dim=1)
    return ys[0, 1:].tolist()


def length_penalty(length, alpha):
    return ((5 + length) / 6) ** alpha


@torch.no_grad()
def beam_search(model, src, cfg):
    model.eval()
    src_mask = make_src_mask(src)
    memory = model.encode(src, src_mask)
    max_len = src.size(1) + cfg.max_extra_len

    alive = [([SOS], 0.0)]
    finished = []

    for _ in range(max_len):
        k = len(alive)
        ys = torch.tensor(
            [toks for toks, _ in alive]
        )  # every alive hypothesis has the same length
        out = model.decode(
            ys,
            memory.expand(k, -1, -1),
            src_mask.expand(k, -1, -1, -1),
            make_tgt_mask(ys),
        )
        log_probs = torch.log_softmax(model.generator(out[:, -1]), dim=-1)

        candidates = []
        for i, (toks, score) in enumerate(alive):
            top = torch.topk(log_probs[i], cfg.beam_width)
            for lp, idx in zip(top.values.tolist(), top.indices.tolist()):
                candidates.append((toks + [idx], score + lp))
        candidates.sort(key=lambda x: x[1], reverse=True)

        alive = []
        for toks, score in candidates:
            if toks[-1] == EOS:
                n_out = len(toks) - 1  # output length, counting <eos>
                finished.append((toks[1:-1], score / length_penalty(n_out, cfg.alpha)))
            else:
                alive.append((toks, score))
            if len(alive) == cfg.beam_width:
                break

        if not alive:
            break
        if finished:
            best_possible = alive[0][1] / length_penalty(max_len, cfg.alpha)
            if max(s for _, s in finished) >= best_possible:
                break

    if not finished:
        finished = [
            (t[1:], s / length_penalty(len(t) - 1, cfg.alpha)) for t, s in alive
        ]
    return max(finished, key=lambda x: x[1])[0]


######################
# Evalutation
######################


def exact_match(model, pairs, cfg, vocab, decode_fn):
    correct = 0
    for pair in pairs:
        src_ids, _ = tensorize(pair, vocab)
        tokens = decode_fn(model, torch.tensor([src_ids]), cfg)
        correct += vocab.decode(tokens) == pair[1]
    return correct / len(pairs)


@torch.no_grad()
def print_alignment(model, pair, cfg, vocab, layer=-1):
    model.eval()
    src_ids, _ = tensorize(pair, vocab)
    src = torch.tensor([src_ids])
    out_ids = greedy_decode(model, src, cfg)
    model(src, torch.tensor([[SOS] + out_ids]))  # one teacher-forced pass to fill .attn

    cols = vocab.decode(src_ids)
    rows = vocab.decode(out_ids) + [
        "<eos>"
    ]  # row i is the step that predicts this token
    weights = model.decoder.layers[layer].cross_attn.attn[0]  # (h, T_y, T_x)
    width = max(len(w) for w in cols + ["0.00"]) + 1
    label = max(len(w) for w in rows) + 2
    for head in range(weights.size(0)):
        print(f"  layer {layer % cfg.N}, head {head}")
        print(" " * label + "".join(w.rjust(width) for w in cols))
        for word, row in zip(rows, weights[head]):
            print(
                word.ljust(label)
                + "".join(f"{a:.2f}".rjust(width) for a in row.tolist())
            )
        print()