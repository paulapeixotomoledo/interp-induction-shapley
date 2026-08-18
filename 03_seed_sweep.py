"""
Experiment 4 — error bars.

Everything reported so far came from a single random draw of the prompt
(`SEED = 0`). That is fine for the large effects but not for the small ones, and
two of the claims in this repo rest on small numbers:

  * L7H10's marginal ablation effect came out at -0.011 nats, i.e. removing it
    slightly *improved* the loss. That could easily be noise.
  * L2H9 costs +0.065 nats ablated alone, which is the basis for concluding that
    its +2.5 nats at position seven was redundancy exhaustion rather than the
    head being special. If the solo effects are all within noise of each other,
    that conclusion is firm; if L2H9 stands out, it needs revisiting.

This re-runs the key measurements over several seeds. The head *set* is held
fixed — the question is whether the effects are stable, not whether head
selection is — but head selection is checked separately, since it has never been
tested across seeds either.

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

MODEL_NAME = "gpt2"
SEQ_LEN = 50
BATCH = 64
SEEDS = [0, 1, 2, 3, 4]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IND_HEADS = [(5, 5), (6, 9), (5, 1), (7, 10), (7, 2)]
PREV_HEADS = [(4, 11), (3, 7), (6, 8), (2, 2), (5, 6), (3, 2),
              (2, 9), (3, 8), (2, 5), (3, 6), (3, 3), (2, 3)]

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


def make_tokens(model, generator):
    bos = torch.full((BATCH, 1), model.tokenizer.bos_token_id, dtype=torch.long)
    block = torch.randint(0, model.cfg.d_vocab, (BATCH, SEQ_LEN),
                          dtype=torch.long, generator=generator)
    return torch.cat([bos, block, block], dim=1).to(DEVICE)


def zero_hook(head_idx):
    def hook(z, hook):
        z[:, :, head_idx, :] = 0.0
        return z
    return hook


def hooks_for(heads):
    return [(f"blocks.{l}.attn.hook_z", zero_hook(h)) for l, h in heads]


def metrics(model, tokens, heads):
    """loss, p(correct), centred logit -- all on second-copy positions."""
    if heads:
        with model.hooks(fwd_hooks=hooks_for(heads)):
            lg = model(tokens)
    else:
        lg = model(tokens)
    sl = lg[:, SEQ_LEN:2 * SEQ_LEN, :]
    tgt = tokens[:, SEQ_LEN + 1:2 * SEQ_LEN + 1]
    correct = sl.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    logprobs = sl.log_softmax(-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return {"loss": (-logprobs).mean().item(),
            "probability": logprobs.exp().mean().item(),
            "centred logit": (correct - sl.mean(-1)).mean().item()}


def shapley(v, players):
    n = len(players)
    out = {}
    for p in players:
        others = [q for q in players if q != p]
        tot = 0.0
        for sz in range(n):
            w = factorial(sz) * factorial(n - sz - 1) / factorial(n)
            for c in itertools.combinations(others, sz):
                s = frozenset(c)
                tot += w * (v[s | {p}] - v[s])
        out[p] = tot
    return out


def induction_scores(model, tokens):
    _, cache = model.run_with_cache(tokens)
    sc = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    for layer in range(model.cfg.n_layers):
        d = cache["pattern", layer].diagonal(offset=1 - SEQ_LEN, dim1=-2, dim2=-1)
        sc[layer] = d.mean(dim=(0, -1)).cpu()
    del cache
    return sc


def main():
    torch.set_grad_enabled(False)
    print(f"loading {MODEL_NAME} on {DEVICE}")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()

    subsets = [frozenset(c) for n in range(len(IND_HEADS) + 1)
               for c in itertools.combinations(IND_HEADS, n)]
    METRICS = ["loss", "probability", "centred logit"]

    per_seed = []
    for seed in SEEDS:
        print(f"\n--- seed {seed} ---")
        tokens = make_tokens(model, torch.Generator().manual_seed(seed))

        vals = {s: metrics(model, tokens, sorted(s)) for s in subsets}
        base = vals[frozenset()]

        rec = {"seed": seed, "clean": base, "marginal": {}, "shapley": {},
               "interaction_share": {}, "joint": {}, "solo_prev": {}}

        for m in METRICS:
            sign = 1.0 if m == "loss" else -1.0
            dmg = {s: sign * (vals[s][m] - base[m]) for s in subsets}
            total = dmg[frozenset(IND_HEADS)]
            marg = {h: dmg[frozenset([h])] for h in IND_HEADS}
            phi = shapley(dmg, IND_HEADS)
            rec["joint"][m] = total
            rec["marginal"][m] = {f"L{l}H{h}": marg[(l, h)] for l, h in IND_HEADS}
            rec["shapley"][m] = {f"L{l}H{h}": phi[(l, h)] for l, h in IND_HEADS}
            rec["interaction_share"][m] = (total - sum(marg.values())) / total

        for lh in PREV_HEADS:
            rec["solo_prev"][f"L{lh[0]}H{lh[1]}"] = (
                metrics(model, tokens, [lh])["loss"] - base["loss"])

        sc = induction_scores(model, tokens)
        top = torch.topk(sc.flatten(), k=5)
        rec["top5_induction"] = [
            {"head": f"L{i // model.cfg.n_heads}H{i % model.cfg.n_heads}",
             "score": v}
            for v, i in zip(top.values.tolist(), top.indices.tolist())]

        print("  top5:", [t["head"] for t in rec["top5_induction"]])
        print(f"  interaction share: " + "  ".join(
            f"{m} {100*rec['interaction_share'][m]:.1f}%" for m in METRICS))
        per_seed.append(rec)

    # ---------------------------------------------------------------- summary
    def agg(getter):
        a = np.array([getter(r) for r in per_seed], dtype=float)
        return a.mean(axis=0), a.std(axis=0, ddof=1)

    labels = [f"L{l}H{h}" for l, h in IND_HEADS]
    print("\n=== marginal ablation effect, cross-entropy (mean +/- sd over "
          f"{len(SEEDS)} seeds) ===")
    marg_mu, marg_sd = agg(lambda r: [r["marginal"]["loss"][k] for k in labels])
    for k, mu, sd in zip(labels, marg_mu, marg_sd):
        flag = "   <-- crosses zero" if abs(mu) < 2 * sd else ""
        print(f"  {k:<7}{mu:>+8.3f}  +/- {sd:.3f}{flag}")

    print("\n=== interaction share by metric ===")
    for m in METRICS:
        a = np.array([r["interaction_share"][m] for r in per_seed]) * 100
        print(f"  {m:<15}{a.mean():>6.1f}%  +/- {a.std(ddof=1):.1f}")

    print("\n=== previous-token heads ablated alone, cross-entropy ===")
    plabels = [f"L{l}H{h}" for l, h in PREV_HEADS]
    solo_mu, solo_sd = agg(lambda r: [r["solo_prev"][k] for k in plabels])
    for k, mu, sd in zip(plabels, solo_mu, solo_sd):
        flag = "   <-- crosses zero" if abs(mu) < 2 * sd else ""
        star = "  (the k=7 head)" if k == "L2H9" else ""
        print(f"  {k:<7}{mu:>+8.3f}  +/- {sd:.3f}{flag}{star}")

    stable = all(set(t["head"] for t in r["top5_induction"]) ==
                 set(t["head"] for t in per_seed[0]["top5_induction"])
                 for r in per_seed)
    print(f"\ntop-5 induction head set identical across all seeds: {stable}")

    # ---------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].bar(np.arange(len(labels)), marg_mu, yerr=marg_sd, capsize=4)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(np.arange(len(labels))); axes[0].set_xticklabels(labels, rotation=45)
    axes[0].set_ylabel("Δ loss (nats)")
    axes[0].set_title("Induction heads, ablated alone")

    shares = [np.array([r["interaction_share"][m] for r in per_seed]) * 100 for m in METRICS]
    axes[1].bar(np.arange(len(METRICS)), [s.mean() for s in shares],
                yerr=[s.std(ddof=1) for s in shares], capsize=4, color="purple")
    axes[1].set_xticks(np.arange(len(METRICS))); axes[1].set_xticklabels(METRICS, rotation=20)
    axes[1].set_ylabel("interaction share (%)"); axes[1].set_ylim(0, 100)
    axes[1].set_title("Interaction share by metric")

    axes[2].bar(np.arange(len(plabels)), solo_mu, yerr=solo_sd, capsize=3)
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_xticks(np.arange(len(plabels))); axes[2].set_xticklabels(plabels, rotation=90)
    axes[2].set_ylabel("Δ loss (nats)")
    axes[2].set_title("Previous-token heads, ablated alone")

    plt.suptitle(f"Mean ± sd over {len(SEEDS)} seeds, batch {BATCH}")
    plt.tight_layout()
    fig.savefig(RESULTS / "seed_sweep.png", dpi=150)

    with open(RESULTS / "03_seed_sweep.json", "w") as f:
        json.dump({"seeds": SEEDS, "batch": BATCH, "seq_len": SEQ_LEN,
                   "per_seed": per_seed}, f, indent=2)
    print(f"\nwrote seed_sweep.png + 03_seed_sweep.json to {RESULTS}")


if __name__ == "__main__":
    main()
