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

## Next

Experiment 2 — ablate the top heads individually and jointly, measure how much
of the loss drop each recovers, and compute Shapley values over the head set to
separate individual contributions from interactions.

The question I actually care about: **are head contributions additive?** In the
biological circuit model I work on, a comparable Shapley decomposition came out
almost perfectly additive — the joint nonlinearity was near zero. I do not
expect that here, because these heads sit in different layers and can compose,
but I want to measure it rather than assume it.
