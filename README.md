# Q-Spy - Quantum Machine Learning for Astrophysics

**Projected quantum kernels to classify and discover glitches in LIGO data.**

Q-Spy uses an IBM quantum computer as a feature extractor for LIGO glitch
classification, and checks  whether the
quantum geometry adds anything over well-tuned classical methods.

## The problem
LIGO measures length changes smaller than a proton, and at that sensitivity it
records thousands of non-Gaussian noise transients (glitches) that can hide
or mimic real signals. Gravity Spy classifies them into families. Two gaps
motivate Q-Spy: few-shot (a new family has only a handful of labels) and
novelty detection (a closed-set classifier cannot say "this is new", someone
had to discover Fast Scattering in O3 by hand).

## The idea
Every glitch = 7 Omicron features → encoded as rotation angles in an 8-qubit
circuit with entangling gates → 24 numbers per glitch (Bloch vectors) →
projected quantum kernel → classical SVM / One-Class SVM.

> The quantum computer doesn't learn anything: it changes the geometry in
> which glitches are compared. All the learning stays classical.

The projected kernel (Huang et al. 2021) compares per-qubit Bloch vectors
instead of full states: **3 circuits per glitch instead of N²/2** (360 vs 7140),
γ tunable after measurement, much weaker concentration.

## Headline results
| Result | What we measured |
|---|---|
| Concentration (E-01) | projected kernel avoids concentration; bandwidth matters most: KTA 0.534 at s=0.1778 vs 0.357 at s=1 |
| Noise prediction (Q-06) | glitches differ by 0.23 in Bloch space = 8× the measurement noise |
| Hardware pilot (H-01) | 13 s QPU vs 13.7 predicted; HW error 0.033 vs 0.030 shot-only |
| Hardware campaign (H-02) | 7 experiments, 780 circuits, 276 s; mitigated error 0.030 = shot noise; KTA 0.525 on chip vs 0.531 exact (H4 passed) |
| Few-shot (E-02 preview) | quantum = classical at every k (macro-F1 0.838 both at k=50) |
| Novelty (E-03) | every detector finds Fast Scattering (AUROC > 0.93); classical RBF beats projected by 0.026 [−0.044, −0.008] |

## The verdict
**Two positive results, one negative.** The quantum pipeline runs on real IBM
hardware at predicted cost and noise, and the projected kernel avoids
concentration but on 7 tabular features a well-tuned classical method matches
(few-shot) or slightly beats (novelty) it.

> No quantum advantage — a negative result measured with rules fixed in advance.

Where an advantage could still live: richer inputs (spectrograms, auxiliary
channels), non-simulable scales, and pre-screening with the geometric
difference (implemented in G-02).

## How we work
Five people (lead + 4 areas), ~70 reviewed PRs, ~290 tests, interface contracts
first, frozen hash-verified splits, every result regenerable from a script.

## Documentation
```bash
$ pip install -r requirements-docs.txt && python -m mkdocs serve
```
Quick figure regeneration:
```bash
$ make figures
```
Full test suite:
```bash
$ make test
```
## License
GPL-3.0
