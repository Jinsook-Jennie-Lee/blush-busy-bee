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
- `data/synthetic/` output of the simulator: `labels.csv`, `contact_sheet.png`, and one
  folder of PNG frames per run (frames are git-ignored; regenerate them).

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
