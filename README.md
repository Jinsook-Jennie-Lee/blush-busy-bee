# blush-busy-bee

Side project: apply the AI-in-the-loop ideas from Price et al., *IEEE Computer* (2025)
to the MBE growth of wurtzite InScN reported by Lonergan et al., *J. Appl. Phys.* 139,
125301 (2026). Background and the candidate project list are in `docs/papers.md`.

Current focus: can in-situ RHEED pixels replace some of the ex-situ measurements
(AFM roughness, XRD lattice constant, phase identification)?

## Layout

- `papers/` the InScN paper (CC-BY).
- `docs/papers.md` plain-words summaries, the sample table, and the AI plan.
- `scripts/simulate_rheed.py` makes dummy RHEED frames with ground-truth labels.
- `scripts/read_rheed.py` reads frames back: lattice constant, roughness score, phase.
- `data/synthetic/` output of the simulator: `labels.csv`, `contact_sheet.png`,
  `readings.csv` from the reader, and one folder of PNG frames per run (frames are
  git-ignored; regenerate them).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pillow
```

## Make dummy RHEED data

One frame every 5 minutes over a 60-minute growth, for ten Sc fractions:

```bash
.venv/bin/python scripts/simulate_rheed.py --out data/synthetic --interval 5 --duration 60
```

Each row of `data/synthetic/labels.csv` is one frame with: Sc %, time, the in-plane
lattice constant the streak spacing encodes, the streak spacing in pixels, the final
AFM roughness from the paper, a 0–1 roughness proxy, the rock-salt fraction, and a
phase label (`wurtzite`, `rs_onset`, `mixed_rs`).

The images are synthetic and only mimic the paper's figures 1(g)–(k) and 4(b). They are
for building and testing the pipeline, not for drawing physical conclusions.

## Read the frames back

```bash
.venv/bin/python scripts/read_rheed.py data/synthetic --out data/synthetic/readings.csv
```

No training. Per frame the reader finds the shadow edge, measures the streak spacing
(autocorrelation, then a fit through the streak positions), converts it to the in-plane
lattice constant using the GaN streaks in each run's first frame as the ruler, scores
roughness from how much each streak breaks into spots along its length, and scores a
second phase from distinct peaks between the streaks. When `labels.csv` is present it
grades itself. Current scores on the 130 synthetic frames:

| quantity | score |
|---|---|
| lattice constant a | mean abs error 0.012 Å, worst 0.038 Å |
| roughness | Spearman 0.71 against the simulator's roughness proxy |
| rock-salt phase | 22 of 24 flagged, 0 false alarms; the 2 misses have < 4 % rock-salt |

These numbers say the reader works on the cartoon. They say nothing yet about real
RHEED, where the next step is to run the same script on recorded frames and compare
with the paper's XRD and AFM values.
