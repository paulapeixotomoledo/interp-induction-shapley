"""
Experiment 2 — are induction-head contributions additive?

Ablate the induction heads found in experiment 1, individually and in every
combination, and decompose the resulting loss increase with Shapley values.

Why bother with Shapley
-----------------------
Single-head ablation measures a head's *marginal* effect: the damage done by
removing it while everything else stays intact. If heads compose, marginal
effects need not sum to the joint effect, and the shortfall is interaction that
single-ablation studies never see. Shapley averages each head's marginal
contribution over all orderings of the others, which is the standard way to
split a joint outcome across components when interactions exist.

With n = 5 heads there are 2^5 = 32 subsets, so the exact Shapley value is
computable — no sampling approximation needed.

The counterfactual matters
--------------------------
"Ablate a head" is underspecified. Three choices are implemented here:

  zero      — set the head's output to 0. Cheap, but takes the model
              off-distribution: no input ever produces an exactly zero head.
  mean      — replace with the head's mean output over the batch. Removes the
              head's variation while preserving its average contribution.
  resample  — replace with the head's output on a *different* sequence in the
              batch. Destroys the induction-relevant information while keeping
              activations on-distribution.

These are not interchangeable. In earlier work on a biophysical circuit model I
found that attributing an effect by replacing a parameter with the group median
and by shifting it by the group-level delta assigned the same effect to
opposite mechanisms — the first was measuring variance compression rather than
the effect of interest. Ablation faces the same hazard, so all three are run and
compared rather than one being picked in advance.

Value function
--------------
v(S) = mean cross-entropy on second-copy positions with the heads in S ablated.
The quantity being attributed is v(N) - v(empty): the total damage from removing
all induction heads. Loss is measured only on the second copy, since that is
where induction is the thing being used.

Outputs land in results/.
"""

import itertools
import json
from math import factorial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from transformer_lens import HookedTransformer

# --- config -----------------------------------------------------------------
MODEL_NAME = "gpt2"
SEQ_LEN = 50
BATCH = 64
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Top induction heads from experiment 1, in score order.
HEADS = [(5, 5), (6, 9), (5, 1), (7, 10), (7, 2)]

ABLATIONS = ["zero", "mean", "resample"]

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


def make_repeated_tokens(model, batch, seq_len, generator):
    """[BOS] + S + S, S uniformly random. Same construction as experiment 1."""
    bos = torch.full((batch, 1), model.tokenizer.bos_token_id, dtype=torch.long)
    block = torch.randint(
        0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.long, generator=generator
    )
    return torch.cat([bos, block, block], dim=1).to(DEVICE)


def second_copy_loss(model, tokens, hooks=None):
    """Mean cross-entropy over positions in the second copy."""
    if hooks:
        loss_per_token = model.run_with_hooks(
            tokens, return_type="loss", loss_per_token=True, fwd_hooks=hooks
        )
    else:
        loss_per_token = model(tokens, return_type="loss", loss_per_token=True)
    # loss_per_token[:, i] scores the prediction of position i+1, so the second
    # copy begins at column SEQ_LEN.
    return loss_per_token[:, SEQ_LEN:].mean().item()


def make_hook(head_idx, mode, mean_z=None, perm=None):
    """Hook that ablates one head's z output under the chosen counterfactual."""

    def hook(z, hook):
        # z: [batch, pos, head_index, d_head]
        if mode == "zero":
            z[:, :, head_idx, :] = 0.0
        elif mode == "mean":
            z[:, :, head_idx, :] = mean_z
        elif mode == "resample":
            z[:, :, head_idx, :] = z[perm, :, head_idx, :]
        else:
            raise ValueError(mode)
        return z

    return hook


def build_hooks(subset, mode, mean_cache, perm):
    """fwd_hooks list ablating every head in `subset`."""
    hooks = []
    for layer, head in subset:
        name = f"blocks.{layer}.attn.hook_z"
        mean_z = mean_cache[(layer, head)] if mode == "mean" else None
        hooks.append((name, make_hook(head, mode, mean_z=mean_z, perm=perm)))
    return hooks


def shapley_values(v, heads):
    """Exact Shapley values from the full subset->value table.

    phi_i = sum over S not containing i of
            |S|!(n-|S|-1)!/n! * [v(S + i) - v(S)]
    """
    n = len(heads)
    phi = {}
    for i, head in enumerate(heads):
        others = [h for h in heads if h != head]
        total = 0.0
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for subset in itertools.combinations(others, size):
                s = frozenset(subset)
                total += weight * (v[s | {head}] - v[s])
        phi[head] = total
    return phi


def main():
    torch.set_grad_enabled(False)
    generator = torch.Generator().manual_seed(SEED)

    print(f"loading {MODEL_NAME} on {DEVICE}")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()

    tokens = make_repeated_tokens(model, BATCH, SEQ_LEN, generator)

    # Clean baseline and the mean activations used by mean-ablation.
    _, cache = model.run_with_cache(tokens)
    mean_cache = {
        (layer, head): cache["z", layer][:, :, head, :].mean(dim=(0, 1))
        for layer, head in HEADS
    }
    del cache

    # Fixed derangement-ish permutation for resample ablation: roll by 1 so no
    # sequence receives its own activations.
    perm = torch.roll(torch.arange(BATCH, device=DEVICE), shifts=1)

    v_clean = second_copy_loss(model, tokens)
    print(f"\nclean second-copy loss: {v_clean:.4f} nats")

    all_subsets = [
        frozenset(c)
        for size in range(len(HEADS) + 1)
        for c in itertools.combinations(HEADS, size)
    ]

    payload = {
        "model": MODEL_NAME,
        "seq_len": SEQ_LEN,
        "batch": BATCH,
        "seed": SEED,
        "heads": [list(h) for h in HEADS],
        "clean_loss": v_clean,
        "methods": {},
    }

    for mode in ABLATIONS:
        print(f"\n=== ablation: {mode} ===")
        v = {}
        for subset in all_subsets:
            hooks = build_hooks(subset, mode, mean_cache, perm)
            v[subset] = second_copy_loss(model, tokens, hooks=hooks) if hooks else v_clean

        v_full = v[frozenset(HEADS)]
        total_effect = v_full - v_clean

        marginal = {h: v[frozenset([h])] - v_clean for h in HEADS}
        phi = shapley_values(v, HEADS)

        sum_marginal = sum(marginal.values())
        sum_phi = sum(phi.values())
        interaction = total_effect - sum_marginal

        print(f"  all-5 ablated loss : {v_full:.4f}  (total effect {total_effect:+.4f})")
        print(f"  sum of marginals   : {sum_marginal:+.4f}")
        print(f"  sum of Shapley     : {sum_phi:+.4f}   (must equal total effect)")
        print(f"  interaction        : {interaction:+.4f} "
              f"({100 * interaction / total_effect:+.1f}% of total)")
        print(f"  {'head':>7}  {'marginal':>10}  {'shapley':>10}")
        for h in HEADS:
            print(f"  L{h[0]}H{h[1]:<4}  {marginal[h]:>10.4f}  {phi[h]:>10.4f}")

        payload["methods"][mode] = {
            "v_full": v_full,
            "total_effect": total_effect,
            "sum_marginal": sum_marginal,
            "sum_shapley": sum_phi,
            "interaction": interaction,
            "interaction_frac": interaction / total_effect if total_effect else None,
            "marginal": {f"L{l}H{h}": marginal[(l, h)] for l, h in HEADS},
            "shapley": {f"L{l}H{h}": phi[(l, h)] for l, h in HEADS},
            "subset_values": {
                ",".join(sorted(f"L{l}H{h}" for l, h in s)) or "none": val
                for s, val in v.items()
            },
        }

    # --- figures ------------------------------------------------------------
    labels = [f"L{l}H{h}" for l, h in HEADS]
    x = np.arange(len(HEADS))
    width = 0.38

    fig, axes = plt.subplots(1, len(ABLATIONS), figsize=(4.4 * len(ABLATIONS), 4),
                             sharey=True)
    for ax, mode in zip(np.atleast_1d(axes), ABLATIONS):
        m = payload["methods"][mode]
        ax.bar(x - width / 2, [m["marginal"][k] for k in labels], width,
               label="marginal")
        ax.bar(x + width / 2, [m["shapley"][k] for k in labels], width,
               label="Shapley")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"{mode} ablation")
        ax.axhline(0, color="k", lw=0.6)
    np.atleast_1d(axes)[0].set_ylabel("Δ second-copy loss (nats)")
    np.atleast_1d(axes)[0].legend()
    fig.suptitle("Per-head attribution: marginal vs Shapley")
    fig.tight_layout()
    fig.savefig(RESULTS / "attribution_by_head.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(len(ABLATIONS))
    ax.bar(xs - width / 2,
           [payload["methods"][m]["sum_marginal"] for m in ABLATIONS], width,
           label="sum of marginals")
    ax.bar(xs + width / 2,
           [payload["methods"][m]["total_effect"] for m in ABLATIONS], width,
           label="joint effect (all 5)")
    ax.set_xticks(xs)
    ax.set_xticklabels(ABLATIONS)
    ax.set_ylabel("Δ second-copy loss (nats)")
    ax.set_title("Do marginal effects sum to the joint effect?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "additivity_check.png", dpi=150)

    with open(RESULTS / "02_ablation_shapley.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nwrote figures + 02_ablation_shapley.json to {RESULTS}")


if __name__ == "__main__":
    main()
