"""
Experiment 1 — Replicate induction behaviour in GPT-2 small and locate the heads
that implement it.

Setup
-----
A prompt is built as [BOS] + S + S, where S is a sequence of `SEQ_LEN` uniformly
random tokens. The first copy of S is unpredictable by construction. The second
copy is perfectly predictable *if* the model can look back to the earlier
occurrence of the current token and copy whatever followed it. That behaviour is
what "induction" means here.

Two measurements
----------------
1. Per-position loss. If induction is present, loss should be high across the
   first copy and drop sharply at the start of the second.

2. Per-head induction score. For a destination position d in the second copy,
   the informative source position is d - SEQ_LEN + 1: the token immediately
   after the previous occurrence of the current token. Averaging attention on
   that offset diagonal gives a score per head.

   Derivation of the offset: let d = 1 + SEQ_LEN + j, so the current token is
   S[j] and the next token will be S[j+1]. S[j+1] first appeared at position
   1 + (j+1) = j + 2, and d - SEQ_LEN + 1 = j + 2. Hence offset = 1 - SEQ_LEN.

Outputs land in results/.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from transformer_lens import HookedTransformer

# --- config -----------------------------------------------------------------
MODEL_NAME = "gpt2"
SEQ_LEN = 50        # length of the repeated block
BATCH = 32          # independent random sequences
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


def make_repeated_tokens(model, batch, seq_len, generator):
    """[BOS] + S + S, with S uniformly random over the vocabulary."""
    bos = torch.full((batch, 1), model.tokenizer.bos_token_id, dtype=torch.long)
    block = torch.randint(
        0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.long, generator=generator
    )
    return torch.cat([bos, block, block], dim=1).to(DEVICE)


def induction_score_per_head(cache, model, seq_len):
    """Mean attention on the induction offset diagonal, per (layer, head)."""
    scores = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    for layer in range(model.cfg.n_layers):
        # pattern: [batch, head, dest, src]
        pattern = cache["pattern", layer]
        diag = pattern.diagonal(offset=1 - seq_len, dim1=-2, dim2=-1)
        scores[layer] = diag.mean(dim=(0, -1)).cpu()
    return scores


def main():
    torch.set_grad_enabled(False)
    generator = torch.Generator().manual_seed(SEED)

    print(f"loading {MODEL_NAME} on {DEVICE}")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()

    tokens = make_repeated_tokens(model, BATCH, SEQ_LEN, generator)
    print(f"prompt shape {tuple(tokens.shape)}  (BOS + {SEQ_LEN} + {SEQ_LEN})")

    loss_per_token = model(tokens, return_type="loss", loss_per_token=True)
    loss = loss_per_token.mean(dim=0).cpu().numpy()  # [2*SEQ_LEN]

    # loss[i] is the loss predicting position i+1, so the second copy starts at
    # index SEQ_LEN of this array.
    first_copy = float(loss[:SEQ_LEN].mean())
    second_copy = float(loss[SEQ_LEN:].mean())
    print(f"\nmean loss, first copy  : {first_copy:.3f}")
    print(f"mean loss, second copy : {second_copy:.3f}")
    print(f"drop                   : {first_copy - second_copy:.3f} nats")

    _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
    scores = induction_score_per_head(cache, model, SEQ_LEN)

    flat = scores.flatten()
    top = torch.topk(flat, k=10)
    print("\ntop 10 heads by induction score")
    top_heads = []
    for rank, (val, idx) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
        layer, head = divmod(idx, model.cfg.n_heads)
        print(f"  {rank:2d}. L{layer}H{head}  {val:.3f}")
        top_heads.append({"layer": layer, "head": head, "score": val})

    # --- figures ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, 2 * SEQ_LEN + 1), loss, lw=1.4)
    ax.axvline(SEQ_LEN + 0.5, color="crimson", ls="--", lw=1,
               label="second copy begins")
    ax.set_xlabel("position predicted")
    ax.set_ylabel("cross-entropy loss (nats)")
    ax.set_title(f"{MODEL_NAME}: loss on a repeated random sequence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_per_position.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(scores.numpy(), cmap="viridis", aspect="auto")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title("induction score per head")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "induction_scores.png", dpi=150)

    payload = {
        "model": MODEL_NAME,
        "seq_len": SEQ_LEN,
        "batch": BATCH,
        "seed": SEED,
        "loss_first_copy": first_copy,
        "loss_second_copy": second_copy,
        "loss_per_position": loss.tolist(),
        "induction_scores": scores.tolist(),
        "top_heads": top_heads,
    }
    with open(RESULTS / "01_induction.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nwrote figures + 01_induction.json to {RESULTS}")


if __name__ == "__main__":
    main()
