# Implementing Vaswani et al. (2017): Attention Is All You Need

Paper: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

## Quick start

```bash
uv sync
cd src/attention_is_all_you_need
python3 transformer.py reverse
python3 transformer.py sort
```

The argument selects the task. With no argument the script runs `reverse`.

## Goal

Implement the Transformer from scratch in PyTorch: multi-head scaled dot-product attention, the encoder-decoder stack, sinusoidal positional encodings, the warmup learning rate schedule, label smoothing, and beam search with a length penalty. The goal was a correct small-scale implementation where every component is checked, not a reproduction of the paper's BLEU scores, which would require its data and compute.

## The synthetic tasks

Instead of WMT 2014 English to German, the model is trained on two synthetic sequence tasks.

- The source is a random sequence of 3 to 12 symbols drawn from 20 letters (`a` to `t`), so repeated symbols are common.
- **reverse**: the target is the source in reverse order, for example `a k s s i q` to `q i s s k a`.
- **sort**: the target is the source in alphabetical order, for example `a k s s i q` to `a i k q s s`.
- There is exactly one correct output per input.
- 20000 training pairs and 300 held-out pairs, generated from separately seeded random generators.
- Vocabulary: 20 symbols plus 4 special tokens (`<pad>`, `<sos>`, `<eos>`, `<unk>`), 24 in total, shared by source and target.

The two tasks test different things. Reverse is a fixed positional mapping, so it tests whether the model can use positional encodings. Sort cannot be solved by position at all, so it tests whether attention can find tokens by content and keep count of repeated symbols.

## What follows the paper

| Component | Paper | This implementation |
|---|---|---|
| Architecture | Encoder-decoder, each a stack of N identical layers | Same |
| Attention | Scaled dot-product, softmax(QK^T / sqrt(d_k)) V | Same |
| Multi-head attention | h heads, concatenated, projected by W_O | Same |
| Feed-forward | Two linear layers with ReLU between | Same |
| Residuals and normalization | LayerNorm(x + Dropout(Sublayer(x))) | Same (post-norm) |
| Positional encoding | Fixed sinusoids, sin on even dimensions, cos on odd | Same, not trainable |
| Embedding scale | Embeddings multiplied by sqrt(d_model) | Same |
| Weight tying | Embeddings and pre-softmax projection share weights | Same |
| Dropout | On every sublayer output and on embeddings plus encodings | Same, P_drop = 0.1 |
| Masking | Padding mask, and a causal mask in decoder self-attention | Same |
| Optimizer | Adam, beta1 = 0.9, beta2 = 0.98, eps = 1e-9 | Same |
| Learning rate | d_model^-0.5 * min(step^-0.5, step * warmup^-1.5) | Same formula |
| Label smoothing | eps_ls = 0.1 | Same |
| Batching | Sentences of similar length grouped by token count | Same idea |
| Decoding | Beam search, width 4, length penalty alpha = 0.6 | Same |
| Maximum output length | Input length + 50 | Same |

## Deviations

| | Paper (base model) | This implementation | Reason |
|---|---|---|---|
| Layers N | 6 | 3 | Compute; 3 was enough for both tasks |
| d_model | 512 | 128 | Compute |
| d_ff | 2048 | 512 | Kept the paper's ratio of 4 to d_model |
| Heads h | 8 | 4 | Compute |
| d_k, d_v | 64 | 32 | Kept d_k = d_model / h |
| Parameters | about 65M | 1,387,008 | Compute |
| Data | WMT 2014 EN-DE, about 4.5M sentence pairs | 20000 synthetic pairs | Compute and data |
| Vocabulary | Shared BPE, about 37000 tokens | 24 tokens | Synthetic task |
| Batch size | About 25000 source and 25000 target tokens | About 1500 tokens | Smaller dataset |
| Warmup steps | 4000 | 400 | Scaled to the shorter run |
| Training length | 100000 steps | 60 epochs, 7680 steps | Smaller dataset |
| Hardware | 8 P100 GPUs, 12 hours | CPU, about 16 minutes per task | Compute |
| Metric | BLEU | Exact match | One correct answer per input, so partial credit would hide whether the mapping was learned |
| Checkpoint averaging | Last 5 checkpoints averaged | Not used | Not needed at this scale |
| Weight initialization | Not specified | Xavier uniform for matrices, normal(0, d_model^-0.5) for embeddings | The paper leaves it open |

### Why these defaults

The values in `Config` (`N = 3`, `epochs = 60`, `train_pairs = 20000`) were set for sort, the harder of the two tasks. Reverse reaches 100% on the validation subset by about epoch 25, so 30 epochs is enough if you only run reverse.

## Code decisions

These are choices the paper does not dictate, made in this implementation.

- **Boolean masks broadcast over heads.** The source mask has shape `(B, 1, 1, T_x)` and the target mask `(B, 1, T_y, T_y)`. Masked positions are filled with negative infinity before the softmax, so they get exactly zero weight.
- **The target mask combines padding and causality.** It is the logical AND of the padding mask and a lower-triangular matrix.
- **Attention weights are stored on every attention module** (`self.attn`, detached). This is what makes the attention checks and the alignment printout possible without changing the forward pass.
- **Label smoothing spreads eps over V minus 2 tokens.** The correct token gets 1 - eps, `<pad>` gets 0, and the remaining eps is shared among the other tokens. Padding positions are excluded from the loss entirely.
- **The loss returns two numbers.** The smoothed loss, divided by the number of real target tokens, is what gets optimized. The plain negative log-likelihood is what gets reported, so the logged numbers describe the data, not the smoothed objective.
- **Batches are built by token budget.** Pairs are sorted by length and grouped until the longest sequence times the batch size would exceed `tokens_per_batch`. The batch order is reshuffled every epoch from a seeded generator, so runs are reproducible.
- **Validation uses its own seed,** so it does not change when the training set size changes.
- **Beam search stops early** when the best finished hypothesis already beats the best possible score of every live beam, and falls back to live beams if nothing has finished by the length cap.
- **The overfit check turns off regularization.** It trains on 20 pairs with dropout and label smoothing set to 0 and warmup set to 400, so it measures pure memorisation. With regularization on, the check cannot reach 100% and stops being a clean test.
- **Everything is seeded.** Running the same task twice gives identical losses, accuracies and attention weights on CPU.

## Pipeline overview

1. **Data generation.** Seeded generators produce training and validation pairs for the chosen task.
2. **Vocabulary.** One shared vocabulary maps symbols to integer ids, with reserved ids for the special tokens.
3. **Tensorization.** `<eos>` is appended to the source. The target is wrapped in `<sos>` and `<eos>`.
4. **Batching.** Pairs are sorted by length, grouped by token budget, and padded.
5. **Model.** The encoder turns the source into a memory of shape `(B, T_x, d_model)`. The decoder attends to its own previous outputs and to that memory, and a tied linear layer maps each position to scores over the vocabulary.
6. **Training.** Teacher forcing with label-smoothed loss, Adam, and the warmup schedule, stepped once per batch.
7. **Decoding.** Greedy decoding and beam search, both starting from `<sos>` and stopping at `<eos>` or the length cap.
8. **Evaluation.** Exact match on the 300 held-out pairs, a printout of every failure, and a printout of the last decoder layer's cross-attention for one short example.

## Checks

Before training, the script runs a set of checks. Each one guards against a different kind of silent failure.

| Check | What it verifies |
|---|---|
| Shapes | Masks, encoder memory, decoder output, logits and all three kinds of attention weights have the expected shapes |
| Attention weights | Every attention row sums to 1, padding gets zero weight, and future positions get zero weight |
| Causal mask | Changing target tokens from position k onwards does not change any logits before k, so no future information leaks through any path |
| Padding invariance | Appending extra `<pad>` tokens to the source does not change any output, including through LayerNorm and the feed-forward layers |
| Weight tying | The embedding and the pre-softmax projection are the same tensor |
| Positional encoding | Values match the formulas in section 3.5 of the paper, and the encoding is not trainable |
| Learning rate schedule | The rate peaks exactly at the warmup step, at (d_model * warmup)^-0.5 |
| Overfit | The model can memorise 20 training pairs |

All checks pass on both tasks.

### Parameter count

The 1,387,008 parameters can be derived by hand, which confirms there are no stray biases and that weight tying removed the second output matrix.

| Part | Calculation | Parameters |
|---|---|---|
| Embedding (shared with the output layer) | 24 * 128 | 3,072 |
| One attention block | 4 matrices of 128 * 128, no bias | 65,536 |
| One feed-forward block | 128 * 512 + 512 + 512 * 128 + 128 | 131,712 |
| One LayerNorm | 2 * 128 | 256 |
| One encoder layer | 1 attention + 1 feed-forward + 2 LayerNorms | 197,760 |
| One decoder layer | 2 attention + 1 feed-forward + 3 LayerNorms | 263,552 |
| Total | 3 * 197,760 + 3 * 263,552 + 3,072 | 1,387,008 |

## Bugs found during implementation

| Bug | Symptom | Fix |
|---|---|---|
| `make_pair` body was a copy of `build_dataset`, so it called itself | `RecursionError: maximum recursion depth exceeded` | Rewrote it to generate one `(src, tgt)` pair |
| `MultiHeadAttention.__init__` missing `super().__init__()` | `AttributeError: cannot assign module before Module.__init__() call` | Added the call |
| `B, T_q = q_in.shape` on a 3D tensor | Unpacking error | `B, T_q, _ = q_in.shape` |
| `greedy_decode` called `next_tok.sequence(0)` | `AttributeError` | `next_tok.unsqueeze(0)` |
| Adam betas set to `(0.1, 0.9)` | No error, but training would be much worse | The paper's `(0.9, 0.98)` |

The most dangerous of these was the Adam betas. It raised no error, and the model would still have trained, just badly, with nothing pointing to the cause.

## Results

All runs on CPU with seed 0.

| | reverse | sort |
|---|---|---|
| Overfit check (20 pairs, 400 steps) | 100% memorised, nll/token 0.0001 | 100% memorised, nll/token 0.0001 |
| Final training nll/token | 0.111 | 0.112 |
| Final training perplexity | 1.118 | 1.119 |
| Held-out exact match, greedy | 99.7% (299 / 300) | 99.3% (298 / 300) |
| Held-out exact match, beam | 99.7% | 99.3% |
| Training time | about 16 minutes | about 17 minutes |

### Training curves

Reverse reaches 100% on the 100-example validation subset at epoch 25 and stays there. Sort reaches 97% by epoch 15 and sits at 99% to 100% from epoch 30, with small swings (97% at epoch 50) that are noise: each percentage point on the subset is a single sequence. The full 300-example figures in the table are the reliable ones.

Sort starts faster than reverse (nll/token 1.52 against 2.47 after the first epoch) but takes longer to finish. Sort has an easy shortcut, early letters first and late letters last, which gives quick early progress. The counting of repeated symbols is what takes the remaining epochs.

Both tasks end with nll/token around 0.11. This is not an error left to fix. It is the floor set by label smoothing (see the glossary).

Greedy decoding and beam search give the same accuracy on both tasks. Each input has exactly one correct output and the model is confident at every step, so the best greedy path is almost always the best beam.

### The sort failures

```
src   l a b f t l j l o q l t   (len 12, repeats {'l': 4, 't': 2})
want  a b f j l l l l o q t t
got   a b f j l l l o q t t
-> wrong length (11 vs 12)

src   d q d s n s q q k   (len 9, repeats {'d': 2, 'q': 3, 's': 2})
want  d d k n q q q s s
got   d d k n q q s s
-> wrong length (8 vs 9)
```

Both failures have the same shape:

- The order is correct, and every other symbol has the right count, including the other repeated ones.
- The error is one missing copy of the most repeated symbol in the sequence (`l` appears 4 times, `q` appears 3 times).
- It is always an undercount. The model moves on to the next symbol one copy too early.

The model has learned to sort. What it occasionally gets wrong is keeping count of a symbol that appears three or more times. The attention analysis below explains why.

## Analysis

### Reverse: attention by position

The last decoder layer's cross-attention for `a k s s i q` to `q i s s k a` is a clean anti-diagonal in every head. The step that outputs `q` attends to the source `q` at the last position, the step that outputs `a` attends to the first position, and `<eos>` attends to the source `<eos>`. Most weights are between 0.92 and 1.00. The model has learned reversal as a pointer from each output step to the mirrored input position, and it can only do that through the positional encodings, since the symbols themselves say nothing about where they are.

The repeated `s s` is where position matters most, because the two tokens have the same content. Heads 1 and 3 point firmly at the correct `s` (0.92 and 0.90). Head 2 puts 0.89 on `k`, the neighbouring position, and head 0 splits 0.37 and 0.63. The output is still correct because the heads are combined through W_O, so the model tolerates individual heads being off by one. Repeated symbols are the soft spot for position-based pointing, and the single remaining reverse error most likely comes from one.

### Sort: attention by content

For `a k s s i q` to `a i k q s s` there is no diagonal at all. The model finds each symbol by what it is, not where it is.

- **Head 0 is a content pointer.** At the steps that output `a`, `i`, `k` and `q` it puts 0.82 to 0.89 on that symbol, wherever it sits in the source.
- **Repeated symbols get exactly equal weight.** The two `s` columns read 0.17 and 0.17, 0.29 and 0.29, 0.45 and 0.45, 0.33 and 0.33 across heads. In reverse, a split between duplicates meant uncertainty about position. In sort, the two `s` tokens are interchangeable, and the model treats them identically.
- **The last steps lean on `<eos>`.** Head 0 puts 0.37 on `<eos>` in the first `s` row and 0.49 in the `<eos>` row. With no single new symbol to point at, attention spreads out and part of it settles on `<eos>`. Knowing that a second `s` is needed, and when to stop, must come from decoder self-attention over what has already been output.
- **Head 3 looks one symbol ahead.** At the step that outputs `a`, it puts 0.71 on `i`, the next symbol to output. The other heads carry `a`, so the output is correct.
- **Heads 1 and 2 are mostly diffuse,** spreading 0.1 to 0.3 over many positions. They appear to gather information about the whole source rather than copy a single token.

### Why attention undercounts

When the decoder attends to the copies of a symbol, it gives each copy equal weight. Attention returns a weighted average of the values it attends to, and the average of four identical `l` vectors is the same as the average of three. So cross-attention can retrieve "this is an `l`" but cannot directly see how many copies there are.

To know when to stop emitting a symbol, the model has to count indirectly: by how much attention mass that symbol takes relative to the rest of the source, or by tracking through decoder self-attention how many copies it has already written. That indirect counting gets less precise as the count grows. Counts of two are reliable, and counts of three and four occasionally slip by one, which is exactly what the two failures show. This is a property of attention, not a bug in the implementation.

Beam search cannot recover these errors. The model is confident it has finished the symbol, so every beam moves on at the same point.

## Glossary

**Negative log-likelihood (nll).** For each target position, the model gives a probability p to the correct token. The nll for that position is -log(p). It is 0 when the model is certain and correct, and grows without bound as the model becomes confidently wrong. For reference, a uniform guess over 24 tokens gives -log(1/24), about 3.18.

**nll/token.** The total nll over every real (non-padding) target token in an epoch, divided by the number of those tokens. It is the average surprise per token. The value e^(-nll/token) is the geometric mean of the probabilities given to the correct tokens, so a final value of 0.111 corresponds to about 0.89.

**Perplexity (ppl).** e^(nll/token). It reads as the number of tokens the model is effectively choosing between at each step. A perplexity of 1 means certain and correct every time. A perplexity of 24 means guessing uniformly over the vocabulary.

**Exact match.** The fraction of held-out sequences where the whole output is identical to the target. One wrong or missing token makes the whole sequence wrong. It is used instead of BLEU because each input here has exactly one correct output, and partial credit would hide whether the mapping was actually learned.

**Label smoothing.** Instead of training toward a target of 1.0 on the correct token, the model trains toward 1 - eps (0.9 here), with the remaining eps spread over the other tokens. This discourages overconfidence. It also sets a floor on the reported loss: a model that fully learns the smoothed target puts about 0.9 on the correct token, so nll/token settles near -log(0.9), about 0.105, not 0. That is why both tasks stop around 0.11.

**Teacher forcing.** During training, the decoder input is the correct target shifted right by one (starting with `<sos>`), not the model's own previous predictions. All positions are trained in parallel, with the causal mask stopping each position from seeing the answer. At decoding time there is no target, so the model feeds back its own outputs one step at a time.

**Warmup learning rate schedule.** The learning rate rises linearly for the first `warmup_steps` steps, then decays in proportion to 1 / sqrt(step). The peak is (d_model * warmup_steps)^-0.5, about 4.42e-3 here. The warmup avoids large, unstable updates at the start of training, when Adam's estimates of the gradient statistics are still poor.

**Greedy decoding.** At each step, take the single most likely next token. Fast, but one early mistake cannot be undone.

**Beam search.** Keep the `beam_width` most likely partial outputs (4 here) at every step, extend each by every possible next token, and keep the best 4 again. When outputs end with `<eos>`, they are set aside, and the best finished output wins. It can recover from a choice that looked best locally but leads to a worse sequence overall.

**Length penalty (alpha).** Each extra token adds a negative log-probability, so raw beam search scores favour short outputs. Finished outputs are divided by ((5 + length) / 6)^alpha, with alpha = 0.6 as in the paper, which offsets that bias. With alpha = 0 there is no correction.

**Masking.** Setting some attention scores to negative infinity before the softmax so those positions get exactly zero weight. The padding mask stops any position from attending to `<pad>`. The causal mask stops each decoder position from attending to later positions, so the model cannot see the token it is meant to predict.

**Weight tying.** The input embedding matrix and the final projection to vocabulary scores are the same tensor. It saves parameters and means a token's input representation and its output score are learned together.

**Overfit check.** Training on a tiny set (20 pairs here) until the model memorises it. A correct model and training loop should reach 100%. If it cannot, something in the model, the loss or the optimizer is broken, and there is no point training on the full data.

## Lessons

- **Check each part directly.** The checks test the masks, causality, padding, tying, encodings and schedule in isolation. A model can train to a decent score with a subtle masking bug, so a good final score alone does not prove the implementation is right.
- **A sanity check must be configured to be clean.** With dropout and label smoothing on, the overfit check cannot reach 100%, and a shortfall says nothing about whether the model is correct. Turning regularization off makes it a real test.
- **Silent bugs are the expensive ones.** The crashes were fixed in minutes. Flipped Adam betas would have trained without complaint and simply produced worse results.
- **Know where the loss floor is.** The nll/token stopping at 0.11 looks like unfinished training until you account for label smoothing.
- **Look at the failures, not just the metric.** 99.3% says the model is nearly right. The two printed failures show exactly how it is wrong, and that pointed to a real limitation of attention.
- **Attention maps show the algorithm.** The same architecture learned a positional strategy for reverse and a content-based strategy for sort, and both are visible directly in the cross-attention weights.

## Citations

```bibtex
@misc{vaswani2017attention,
    title   = {Attention Is All You Need},
    author  = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
    year    = {2017},
    eprint  = {1706.03762},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url     = {https://arxiv.org/abs/1706.03762}
}
```