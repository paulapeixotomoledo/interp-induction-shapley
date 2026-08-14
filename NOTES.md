# Lab notes

Running log. Observations, surprises, and open questions — including the ones I
have not resolved.

---

## 2026-08-12 — Experiment 1: induction replication

**Setup.** GPT-2 small, `[BOS] + S + S` with `S` = 50 uniformly random tokens,
batch 32, seed 0. RTX A4500, torch 2.5.1+cu121, transformer-lens 3.7.1.

**Result.** Induction is unmistakable.

| | mean loss (nats) |
|---|---|
| first copy | 13.183 |
| second copy | 0.536 |
| **drop** | **12.647** |

Top heads by induction score:

| rank | head | score |
|---|---|---|
| 1 | L5H5 | 0.907 |
| 2 | L6H9 | 0.891 |
| 3 | L5H1 | 0.889 |
| 4 | L7H10 | 0.888 |
| 5 | L7H2 | 0.814 |
| 6 | L10H1 | 0.509 |
| 7 | L9H6 | 0.504 |
| 8 | L9H9 | 0.494 |
| 9 | L10H7 | 0.462 |
| 10 | L5H0 | 0.451 |

The top five match the induction heads reported for GPT-2 small in the
literature, which I take as evidence the measurement is implemented correctly
rather than as a new finding.

### Things I noticed that I did not expect

**1. The loss drop is not instantaneous.** It takes roughly 3–4 positions into
the second copy before loss reaches its floor, rather than dropping at the
first repeated token. Some of this has to be that position `SEQ_LEN + 1` is the
first point at which a previous occurrence exists to attend to — but that
accounts for one position, not four. Open question: is the residual ramp a
property of the heads, or of how much context the QK circuit needs before the
match is unambiguous?

**2. There are isolated spikes in the second copy** — clear ones near positions
70 and 88, reaching ~2 nats against a floor near 0. My guess is that these are
positions where the sampled token happens to occur more than once in the first
copy, so "the token after the previous occurrence" is ambiguous and the head has
to split attention. Testable: check whether spike positions coincide with
duplicated tokens in `S`. Not yet done.

**3. The head scores fall into two tiers**, not a smooth gradient: five heads at
0.81–0.91, then a gap, then a cluster at 0.45–0.51. The gap between rank 5
(0.814) and rank 6 (0.509) is much larger than any within-tier step. Are the
second-tier heads doing weaker induction, or something related but distinct that
this metric happens to score? The induction score is a fairly blunt instrument —
it only measures attention on one offset diagonal and says nothing about whether
that attention is used downstream.

That last point is what motivates experiment 2: attention on the right diagonal
is a correlational measure. Ablation is the causal one.

### Methodological note

`HookedTransformer.from_pretrained` emits a DeprecationWarning under
transformer-lens 3.x, which wants `TransformerBridge.boot_transformers(...)`
plus `enable_compatibility_mode()`. Kept the old path for now because it is what
most published code uses and the numerics are the reference ones. Worth
migrating before this repo is read by anyone.

---

## 2026-08-14 — Experiment 2: ablation and Shapley

**Setup.** The five heads from experiment 1, ablated individually and across all
32 subsets. Value function `v(S)` = mean cross-entropy on second-copy positions
with the heads in `S` ablated. Attributed quantity is `v(all) − v(none)`. Batch
64, seed 0. Exact Shapley, no sampling. Sanity check passed: Σφ equals the total
effect to floating point in all three conditions, as the efficiency axiom
requires.

Clean second-copy loss: **0.5215 nats**.

### The answer is no, and not by a little

| ablation | Σ marginals | joint effect | interaction | interaction share |
|---|---|---|---|---|
| zero | +0.720 | +4.800 | +4.080 | **85.0%** |
| mean | +0.763 | +5.156 | +4.392 | **85.2%** |
| resample | +0.291 | +11.586 | +11.295 | **97.5%** |

Removing any single induction head does almost nothing. Removing all five is
catastrophic. Between 85% and 97.5% of the damage lives in the interaction term,
which single-head ablation cannot see by construction.

I expected some interaction. I did not expect the marginal effects to be
*almost entirely uninformative*. In the biophysical circuit model the same
decomposition came out additive to within 0.001 — Shapley told me nothing that
marginal attribution had not. Here it is the reverse: marginal attribution is
close to worthless on its own.

### The correlational and causal measures disagree about which head matters

L5H5 had the **highest** induction score in experiment 1 (0.907 — it places more
attention on the induction diagonal than any other head). Its marginal ablation
effect is **+0.046 nats**. Essentially nothing.

Meanwhile L5H1, ranked third by attention, carries almost the entire marginal
signal (+0.640 under zero ablation).

Attending to the right place and being load-bearing are different properties.
This is the reason experiment 2 exists, and it is a sharper demonstration than I
expected to get.

### The counterfactual changes the answer — magnitude and ranking

| | top head by Shapley | joint effect |
|---|---|---|
| zero | L5H1 (1.57) | 4.80 |
| mean | L5H1 (2.14) | 5.16 |
| resample | **L6H9** (3.54) | **11.59** |

Resample ablation gives a joint effect 2.4× larger than zero ablation, and
promotes a different head to the top. So "how much does this head matter" has no
answer independent of how you choose to remove it.

This is the same failure mode I hit on the circuit model, where replacing a
parameter with the group median and shifting it by the group-level delta
attributed the effect to opposite mechanisms. I went looking for it here because
of that, and it is present.

### Things I am not claiming, and one caveat

**Redundancy among attention heads is not my discovery.** Backup heads and
self-repair are documented in the interpretability literature — ablate one head
and others compensate. What I have added is a measurement: the exact Shapley
decomposition puts a number on how much of the total effect is interaction
rather than individual contribution, and it is most of it.

**Resample ablation may overstate.** It does not merely delete the head's
contribution, it substitutes another sequence's — injecting actively wrong
information rather than none. Some of the 97.5% is likely the model being misled
rather than deprived. Zero and mean are the more conservative bounds. I have not
separated these.

**Untested:** whether the interaction is concentrated in specific pairs, or
spread evenly. The subset table in `results/02_ablation_shapley.json` has
everything needed to check, and I have not looked yet.

**L7H10 has a negative marginal effect under zero ablation** (−0.011 nats):
removing it very slightly improves second-copy loss. Small enough that it may be
noise. Not checked against a seed sweep.

---

## Next

- Is the interaction pairwise or higher-order? The 32-subset table already
  answers this.
- Seed sweep, to put error bars on the small marginals and settle L7H10.
- Separate "information removed" from "wrong information injected" in the
  resample condition.
- Migrate off the deprecated `from_pretrained` path.
