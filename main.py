from dataclasses import dataclass, replace

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
    task: str = "reverse" # "reverse" | "sort"
    num_symbols : int = 20
    min_len: int = 3
    max_len: int = 12
    train_pairs: int = 10000
    valid_pairs: int = 300
    max_positions: int = 512

    overfit_steps: int = 400
    seed: int = 0