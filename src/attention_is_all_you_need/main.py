import random
from dataclasses import dataclass

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
