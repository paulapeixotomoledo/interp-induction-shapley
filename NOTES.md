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
removing it very slightly improves second-copy loss. ~~Small enough that it may
be noise.~~ **Corrected 2026-08-18: the seed sweep says it is real** — −0.013 ±
0.002 over five seeds, consistently negative in every one. See the control-3
entry below.

---

## 2026-08-17 — Experiment 3: the previous-token stage

All of this is in `01_induction_annotated.ipynb`, sections 7–8e. `BATCH=32` here
against 64 in the script, so numbers shift slightly.

### Finding the other half of the circuit

Same offset-diagonal technique, offset `−1` instead of `1 − SEQ_LEN`. Strongest
previous-token head is **L4H11 at 0.989**, next is L3H7 at 0.541. Layer ordering
comes out right: previous-token heads in layers 2–4, induction heads in 5–7.

I want to be clear with myself that this part is **confirmatory**. I predicted
early-layer previous-token heads and found early-layer previous-token heads. The
prior was already high and the result could barely have surprised me. It is a
check on the instrument, not a finding.

The one thing in it I did not already know is the *shape* of the two
distributions: previous-token is a single isolated spike (0.989 vs 0.541 next),
induction is five heads bunched at 0.81–0.91. Different redundancy structure in
the two halves. That is what suggested the test below.

### The prediction, and its failure

If L4H11 is that dominant and has no real backup, ablating it alone should break
induction — even though ablating any single induction head barely does anything.
That can fail, which is why it was worth running.

It failed.

| ablation | L4H11 | control | best single induction | all 5 induction |
|---|---|---|---|---|
| zero | +0.250 | +0.009 | **+0.612** | +4.787 |
| mean | +0.250 | +0.003 | +0.632 | +5.068 |
| resample | **+0.736** | −0.001 | +0.116 | +11.525 |

L4H11 costs less than the best single induction head under zero ablation. Under
resample it costs more. Third time in this project that the counterfactual choice
reverses a ranking.

The causal link is real, though: with L4H11 zeroed the induction scores fall
(L5H1 0.889→0.537, L5H5 0.907→0.673, L7H2 0.814→0.672) and the matched control
moves nothing by more than 0.01. Specific and localised — just not load-bearing.

### No breakpoint exists

Cumulative removal, strongest first, matched control at every step, out to k=24.

| | k |
|---|---|
| induction score halved | 5 |
| below 0.2, and loss halfway to no-information | 11 |
| below 0.1 | 15 |
| largest role-specific gap (+5.2 nats) | 16 |

Erosion, not a cliff. ~11–16 heads carry the role.

**Validity stops at k ≈ 19.** At k=20 the control's own induction score drops to
0.365 and the gap collapses from 4.45 to 1.63 — past that I am breaking the model,
not measuring the role. Rows beyond k=19 are in the output but should not be read.

### A control error I made and had to fix

The first matched control picked heads by lowest previous-token score within
layer. That selected **L6H9 and L5H1 — induction heads**, which score low on
previous-token precisely because they do not attend to `i−1`. Ablating them put
the effect I was controlling for *into* the control, and inflated the apparent
role-specific loss effect roughly threefold.

Fixed by excluding anything with induction score > 0.3. Worth remembering: a
control defined by the absence of one property can accidentally select for
another.

---

## 2026-08-17 — Control 1: is importance a property of the head?

Adding L2H9 seventh cost +2.5 nats, the biggest single step anywhere in the
sweep, from a head ranked only seventh by score. Head, or position?

**Ablated alone, L2H9 costs +0.065 nats.** Nothing. And across eleven different
orderings the largest jump lands anywhere between k=6 and k=12, with L2H9
responsible in exactly one of them — the original. It is redundancy exhaustion.

But the same sweep produced the better result. **L4H11: +0.250 nats ablated
first, +5.23 nats ablated last.** A factor of twenty, same head, same model, same
metric. It is the head behind the largest jump in five of eleven orderings.

So the thing I should stop saying is "how much does this head matter". There is
no such quantity. Single-head ablation on an intact model measures one cell of a
table and reports it as the head's importance.

---

## 2026-08-17 — Control 2: how much of the interaction is the logarithm?

This is the one I should have run before writing up experiment 2.

Cross-entropy is logarithmic, and log-loss is superadditive even for independent
contributions — while enough probability survives on the right token the loss
hardly moves, and only when it collapses does it explode. The circuit-model
comparison I keep making measured firing rate, which is roughly linear, so the
two are not on the same footing.

Same 32 subsets, same Shapley, three measures off one forward pass:

| metric | joint | Σ marginals | interaction |
|---|---|---|---|
| cross-entropy | 4.787 | 0.696 | **85.5%** |
| probability of correct token | 0.613 | 0.333 | **45.7%** |
| centred logit | 9.004 | 6.918 | **23.2%** |

Efficiency axiom holds in all three, so each decomposition is internally fine.
What differs is what is being decomposed.

**Conclusion: the interaction is real but the 85% was inflated.** 23% under the
most linear measure is still far from the ~0% I got in the circuit model, so
there is genuine redundancy here. But "85% is interaction" is a statement about
cross-entropy, and I wrote it as though it were a statement about the circuit.
README and this file have been corrected.

Rule for myself: never quote an interaction share without the metric attached.

---

## 2026-08-18 — Control 3: error bars

`03_seed_sweep.py`. Five seeds, batch 64, head set held fixed. Ran it because two
claims in here rested on numbers small enough to be noise.

### The headline holds

| metric | interaction share |
|---|---|
| cross-entropy | 84.6% ± 0.4 |
| probability | 45.2% ± 0.7 |
| centred logit | 20.8% ± 1.9 |

Under two points of spread in every condition. The metric dependence is not a
seed artefact, which was the thing I most wanted to be sure of.

### I was wrong about L7H10

I had written that its negative marginal (−0.011 nats) was small enough to be
noise. It is not: **−0.013 ± 0.002 over five seeds, negative in all five**, about
six standard deviations from zero. Ablating L7H10 genuinely makes second-copy
prediction slightly *better*.

I have no explanation for that and am not going to invent one. Worth a proper
look — it is the kind of thing that usually means the head is doing something
that competes with induction rather than supporting it.

### L2H9 shrinks, and the conclusion gets firmer

Solo effect over five seeds: **+0.032 ± 0.017**, indistinguishable from zero, and
smaller than L2H3's +0.073 ± 0.005. The single-seed +0.065 was on the high side.

So "it was redundancy exhaustion, not the head" is on firmer ground than it was
when I wrote it, not shakier. Good, but I only know that because I checked.

### Head selection is stable

The top five induction heads are the same set in all five seeds. That had never
been tested, and it is the assumption every ablation in this repo rests on.

---

## Where this leaves the project

The question was whether contributions are additive. They are not, and that part
held up.

Three other things accumulated that I was not looking for:

| | swing |
|---|---|
| counterfactual (zero vs resample) | joint effect 2.4×, top head changes |
| metric (cross-entropy vs centred logit) | interaction 85% → 23% |
| context (first vs last) | same head, 0.25 → 5.23 nats |

I noticed these because I went looking for the replacement-versus-shift problem I
already knew about from the circuit model, and then kept running controls. They
were not the plan.

Whether any of it is news to people who work on this full time, I have no idea —
head redundancy certainly is not. What it has changed for me is that I no longer
report a number from an ablation without saying which counterfactual, which
metric, and in what order.

---

## Next

- Is the interaction pairwise or higher-order? The 32-subset table already
  answers this; I have not looked.
- Seed sweep for error bars on the small marginals, and to settle whether
  L7H10's negative marginal is real.
- Ablate across both stages at once. Experiment 2 touched only induction heads,
  experiment 3 only previous-token heads. Never both.
- Separate "information removed" from "wrong information injected" in the
  resample condition.
- Do the second-tier induction heads (L10H1, L9H6, L9H9, L10H7, L5H0) act as the
  backups?
- Migrate off the deprecated `from_pretrained` path.
- Fold experiment 3 and the controls into a script; they currently exist only in
  the notebook.
