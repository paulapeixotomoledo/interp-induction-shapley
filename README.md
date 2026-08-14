# Are attention head contributions additive?

A Shapley analysis of induction-head ablations in GPT-2 small.

**Status:** experiments 1 and 2 are done. Headline: for these heads, **85–97% of
the joint effect is interaction** — single-head ablation sees almost none of it.

---

## The question

Single-head ablation is the standard way to ask what an attention head
contributes to a behaviour. It measures each head's *marginal* effect, holding
everything else fixed. But if heads compose — and induction heads sit in
different layers, so they can — the marginal effects need not sum to the joint
effect. Whatever is left over is interaction, and single-ablation studies
discard it by construction.

This repo measures that gap directly. Ablate the induction heads of GPT-2 small
individually and in every combination, then decompose the loss recovery with
Shapley values, which is the standard way to attribute a joint outcome across
components while accounting for interactions.

I came to this from computational neuroscience, where I use the same
counterfactual-swap-plus-Shapley approach to attribute physiological effects
across parameters of a circuit model. There, the joint nonlinearity turned out
to be near zero and the marginal and Shapley attributions coincided. I would be
surprised if the same held in a transformer. This is an attempt to find out.

## Experiment 1 — replication

Prompt the model with `[BOS] + S + S`, where `S` is 50 uniformly random tokens.
The first copy is unpredictable by construction; the second is perfectly
predictable given the ability to look back at the previous occurrence of the
current token and copy what followed it.

![loss per position](results/loss_per_position.png)

| | mean loss (nats) |
|---|---|
| first copy | 13.183 |
| second copy | 0.536 |

Scoring every head by how much attention it places on the induction offset
diagonal recovers the expected heads:

![induction scores](results/induction_scores.png)

**L5H5 · L6H9 · L5H1 · L7H10 · L7H2** — matching the induction heads reported
for GPT-2 small elsewhere, which is the point: it confirms the measurement is
right before anything is built on top of it.

Observations and open questions from this run, including the ones I have not
resolved, are in [NOTES.md](NOTES.md).

## Experiment 2 — ablation and Shapley

All five heads ablated individually and across all 32 subsets. `v(S)` = mean
cross-entropy on second-copy positions with the heads in `S` ablated; the
attributed quantity is `v(all) − v(none)`. Shapley computed exactly, not
sampled. Σφ equals the total effect in every condition, as the efficiency axiom
requires.

Three ablation counterfactuals, because "ablate a head" is underspecified:
**zero** (output → 0), **mean** (output → its batch mean), **resample** (output
→ that head's output on a different sequence).

### Marginal effects do not sum to the joint effect

![additivity](results/additivity_check.png)

| ablation | Σ marginals | joint effect | interaction share |
|---|---|---|---|
| zero | +0.72 | +4.80 | **85.0%** |
| mean | +0.76 | +5.16 | **85.2%** |
| resample | +0.29 | +11.59 | **97.5%** |

Removing any one induction head is close to harmless; removing all five is
catastrophic. Most of the effect lives in the interaction term, which
single-head ablation cannot observe by construction.

### Attention and causal importance disagree

![attribution](results/attribution_by_head.png)

L5H5 has the **highest** induction score in experiment 1 — it puts more
attention on the induction diagonal than any other head — and a marginal
ablation effect of **+0.05 nats**. L5H1, third by attention, carries most of the
marginal signal. Attending to the right place and being load-bearing are not the
same property.

### The counterfactual choice changes the answer

Resample ablation returns a joint effect 2.4× larger than zero ablation *and*
promotes a different head to the top of the Shapley ranking. There is no
counterfactual-independent answer to "how much does this head matter."

### What is and is not new here

Head redundancy is already documented in the literature — backup heads and
self-repair are known. The contribution here is the measurement: an exact
Shapley decomposition that quantifies how much of the total effect is
interaction rather than individual contribution. Caveats, including why the
resample number is probably an overestimate, are in [NOTES.md](NOTES.md).

## Running it

Environment notes are in `requirements.txt` — the install order matters, since
`transformer-lens` will otherwise pull a CPU-only torch.

```bash
python -m venv .venv
.venv/Scripts/activate                  # Windows
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformer-lens matplotlib
python 01_induction_replication.py
```

Runs on CPU too; GPT-2 small is only 124M parameters.

## References

The induction-head phenomenon and the replication setup follow Olsson et al.,
*In-context Learning and Induction Heads* (2022), and the transformer-circuits
framing of Elhage et al., *A Mathematical Framework for Transformer Circuits*
(2021). Tooling is [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).
