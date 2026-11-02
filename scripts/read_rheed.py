#!/usr/bin/env python3
"""Read RHEED frames: lattice constant, roughness score, phase label.

Physics-only, no training. For each frame it

1. finds the shadow edge (the row where intensity falls off the sample horizon);
2. collapses the region above it into a column profile and measures the streak spacing
   by autocorrelation;
3. converts spacing to the in-plane lattice constant a. The scale is self-calibrated per
   run from the first frame, which is assumed to show the GaN buffer (a = 3.189 Å);
4. scores roughness as how strongly each streak is modulated along its length
   (a smooth streak = 2D growth, a string of spots = 3D growth);
5. flags the rock-salt phase from distinct intensity peaks that sit between the
   wurtzite streaks in the upper part of the pattern (3D spots of a second phase).

If a labels.csv is present (from simulate_rheed.py) it scores itself against it.

Usage:
    python scripts/read_rheed.py data/synthetic            # all runs, with scoring
    python scripts/read_rheed.py data/synthetic/run_sc015p1/frame_t030.png
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

A_GAN = 3.189  # Å


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_gray(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).astype(np.float32)
    return arr[..., 1]  # phosphor screen: green channel carries the signal


def smooth1d(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    k = np.ones(w, dtype=np.float32) / w
    return np.convolve(x, k, mode="same")


def find_shadow_edge(img: np.ndarray) -> int:
    """Row with the steepest drop in mean intensity, searched in the lower 70 % of the frame."""
    rows = smooth1d(img.mean(axis=1), 5)
    grad = np.diff(rows)
    lo = int(0.3 * len(grad))
    return int(np.argmin(grad[lo:]) + lo)


def local_maxima(x: np.ndarray, min_dist: int, min_height: float) -> list[int]:
    peaks = []
    for i in range(1, len(x) - 1):
        if x[i] >= x[i - 1] and x[i] > x[i + 1] and x[i] >= min_height:
            if peaks and i - peaks[-1] < min_dist:
                if x[i] > x[peaks[-1]]:
                    peaks[-1] = i
                continue
            peaks.append(i)
    return peaks


def streak_spacing(profile: np.ndarray, min_lag: int = 30, expected: float | None = None) -> float:
    """Coarse spacing in px from the autocorrelation of the column profile.

    With `expected` (the GaN spacing from the run's first frame) only lags in the
    physically allowed window are considered: the film's a can only grow from the GaN
    value up to ~4 Å, so the spacing can only shrink to ~0.8 of the GaN spacing. This
    keeps a rock-salt spot net, which has its own spacing, from hijacking the estimate.
    """
    p = profile - profile.mean()
    ac = np.correlate(p, p, mode="full")[len(p) - 1:]
    ac /= ac[0] + 1e-9
    peaks = local_maxima(ac, min_dist=10, min_height=0.02)
    peaks = [k for k in peaks if k >= min_lag]
    if expected:
        lo, hi = 0.78 * expected, 1.05 * expected
        peaks = [k for k in peaks if lo <= k <= hi]
    if not peaks:
        return float("nan")
    k = max(peaks, key=lambda i: ac[i]) if expected else peaks[0]
    if 1 <= k < len(ac) - 1:
        y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-9:
            k = k + 0.5 * (y0 - y2) / denom
    return float(k)


def refine_spacing(profile: np.ndarray, spacing: float) -> tuple[float, list[int]]:
    """Fit the spacing from the streak positions themselves.

    Locate the brightest column near each predicted streak position centre +- n*spacing,
    then least-squares fit position vs order. Returns (spacing, streak columns).
    """
    sm = smooth1d(profile, 3)
    centre = int(np.argmax(sm[int(0.3 * len(sm)):int(0.7 * len(sm))]) + int(0.3 * len(sm)))
    half = max(4, int(0.2 * spacing))
    orders, cols = [], []
    for n in range(-3, 4):
        x = centre + n * spacing
        lo, hi = int(round(x - half)), int(round(x + half))
        if lo < 0 or hi >= len(sm):
            continue
        c = int(np.argmax(sm[lo:hi + 1]) + lo)
        # weighted centroid for sub-pixel position
        w = np.clip(sm[lo:hi + 1] - sm[lo:hi + 1].min(), 0, None)
        xs = np.arange(lo, hi + 1)
        pos = float((w * xs).sum() / (w.sum() + 1e-9))
        orders.append(n)
        cols.append(pos)
        _ = c
    if len(orders) < 3:
        return spacing, [int(round(c)) for c in cols]
    slope = np.polyfit(orders, cols, 1)[0]
    return float(abs(slope)), [int(round(c)) for c in cols]


def roughness_score(img: np.ndarray, cols: list[int], top: int, shadow: int, half_w: int = 4) -> float:
    """Mean modulation depth along the streaks: std of the high-passed intensity / mean.

    0 for a perfectly smooth streak, larger as it breaks into spots.
    """
    scores = []
    for c in cols:
        strip = img[top:shadow, max(0, c - half_w):c + half_w + 1].mean(axis=1)
        if strip.mean() < 1e-3:
            continue
        trend = smooth1d(strip, 61)  # slow brightening toward the shadow edge
        hp = strip - trend
        hp = smooth1d(hp, 5)  # drop pixel noise
        scores.append(float(hp.std() / (trend.mean() + 1e-6)))
    return float(np.mean(scores)) if scores else float("nan")


def box_blur(img: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)x(2r+1) window via integral image, edge-padded."""
    pad = np.pad(img, r, mode="edge")
    ii = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    ii = np.pad(ii, ((1, 0), (1, 0)))
    k = 2 * r + 1
    out = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    return out / (k * k)


def rocksalt_score(img: np.ndarray, cols: list[int], spacing: float, top: int, shadow: int) -> float:
    """Strength of off-streak column peaks in the upper part of the pattern, relative to the streaks.

    Rock-salt (3D) spots sit between the wurtzite streaks and make their own peaks in the
    column profile. A Kikuchi arc spreads across all columns and makes no peak; rough
    streaks broaden but stay centred on the streak columns, which are masked out.
    """
    region = img[top:max(top + 10, shadow - 90), :].astype(np.float64)
    region = np.clip(region - box_blur(region, 20), 0, None)
    prof = smooth1d(region.sum(axis=0), 5)
    if not cols:
        return float("nan")
    streak_height = float(np.median([prof[c] for c in cols if 0 <= c < len(prof)]))
    mask = np.ones(len(prof), dtype=bool)
    band = max(6, int(0.25 * spacing))
    for c in cols:
        mask[max(0, c - band):c + band + 1] = False
    off = prof[mask]
    if off.size < 10:
        return float("nan")
    return float((off.max() - np.median(off)) / (streak_height + 1e-6))


# ---------------------------------------------------------------------------
# per-frame and per-run reading
# ---------------------------------------------------------------------------
def central_cols(cols: list[int], n: int = 5) -> list[int]:
    """The n streaks nearest the centre; outer orders are too faint to score roughness on."""
    if len(cols) <= n:
        return cols
    mid = float(np.median(cols))
    return sorted(cols, key=lambda c: abs(c - mid))[:n]


def read_frame(path: Path, expected_spacing: float | None = None) -> dict:
    img = load_gray(path)
    shadow = find_shadow_edge(img)
    top = int(0.08 * img.shape[0])
    # streaks are brightest just above the shadow edge; secondary-phase spots sit higher
    band = img[max(top, shadow - 90):shadow, :].astype(np.float64)
    band = np.clip(band - box_blur(band, 20), 0, None)  # drop the centred diffuse glow
    profile = band.sum(axis=0)
    spacing = streak_spacing(profile, expected=expected_spacing)
    if math.isnan(spacing) and expected_spacing:
        spacing = expected_spacing  # streaks too weak for autocorrelation; refine from the guess
    cols: list[int] = []
    if not math.isnan(spacing):
        spacing, cols = refine_spacing(profile, spacing)
    return {
        "file": str(path),
        "shadow_row": shadow,
        "spacing_px": round(spacing, 2),
        "n_streaks": len(cols),
        "rough_score": round(roughness_score(img, central_cols(cols), top, shadow), 4) if cols else float("nan"),
        "rs_score": round(rocksalt_score(img, cols, spacing, top, shadow), 4) if cols else float("nan"),
    }


def read_run(frames: list[Path], rs_thresh: float = 0.25, rs_strong: float = 0.6) -> list[dict]:
    """Read a run in time order. The first frame (GaN buffer, a = 3.189 Å) calibrates the scale."""
    first = read_frame(frames[0])
    out = [first] + [read_frame(p, expected_spacing=first["spacing_px"]) for p in frames[1:]]
    k_px = first["spacing_px"] * A_GAN  # spacing * a is constant for the setup
    for r in out:
        r["a_est_angstrom"] = round(k_px / r["spacing_px"], 4) if r["spacing_px"] else float("nan")
        sc = r["rs_score"]
        if math.isnan(sc) or sc < rs_thresh:
            r["phase_est"] = "wurtzite"
        elif sc < rs_strong:
            r["phase_est"] = "rs_onset"
        else:
            r["phase_est"] = "mixed_rs"
    return out


# ---------------------------------------------------------------------------
# scoring against labels.csv
# ---------------------------------------------------------------------------
def rank(x):
    order = np.argsort(x)
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    return r


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(rank(x[m]), rank(y[m]))[0, 1])


def score(results: list[dict], labels: dict[str, dict]) -> str:
    a_err, rough_x, rough_y, ph_true, ph_est = [], [], [], [], []
    for r in results:
        lab = labels.get(r["key"])
        if lab is None:
            continue
        a_err.append(abs(r["a_est_angstrom"] - float(lab["a_true_angstrom"])))
        rough_x.append(r["rough_score"])
        rough_y.append(float(lab["roughness_proxy"]))
        ph_true.append(lab["phase"])
        ph_est.append(r["phase_est"])
    ph_true, ph_est = np.array(ph_true), np.array(ph_est)
    bin_true = ph_true != "wurtzite"
    bin_est = ph_est != "wurtzite"
    tp = int((bin_true & bin_est).sum())
    fp = int((~bin_true & bin_est).sum())
    fn = int((bin_true & ~bin_est).sum())
    lines = [
        f"frames scored:                    {len(a_err)}",
        f"lattice a: mean abs error         {np.nanmean(a_err):.4f} Å  (max {np.nanmax(a_err):.4f} Å)",
        f"roughness: Spearman vs truth      {spearman(rough_x, rough_y):.3f}",
        f"phase (rock-salt vs wurtzite):    exact 3-class accuracy {np.mean(ph_true == ph_est):.3f}; "
        f"rock-salt detected {tp}/{tp + fn}, false alarms {fp}",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="a frame, a run directory, or a data directory holding run_* folders")
    ap.add_argument("--out", help="write per-frame results CSV here")
    args = ap.parse_args()
    target = Path(args.target)

    if target.is_file():
        r = read_frame(target)
        for k, v in r.items():
            print(f"{k:14s} {v}")
        return

    runs = sorted(target.glob("run_*")) if not (target / "labels.csv").exists() and not target.name.startswith("run_") else None
    if target.name.startswith("run_"):
        runs = [target]
        data_dir = target.parent
    else:
        runs = sorted(target.glob("run_*"))
        data_dir = target
    if not runs:
        sys.exit(f"no run_* folders under {target}")

    results = []
    for run in runs:
        frames = sorted(run.glob("frame_t*.png"))
        for r in read_run(frames):
            r["run"] = run.name
            r["key"] = str(Path(r["file"]).relative_to(data_dir))
            results.append(r)

    cols = ["run", "key", "spacing_px", "a_est_angstrom", "n_streaks", "rough_score", "rs_score", "phase_est"]
    print(",".join(cols))
    for r in results:
        print(",".join(str(r[c]) for c in cols))

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows({c: r[c] for c in cols} for r in results)

    labels_path = data_dir / "labels.csv"
    if labels_path.exists():
        with open(labels_path) as f:
            labels = {row["file"]: row for row in csv.DictReader(f)}
        print("\n== score vs labels.csv ==")
        print(score(results, labels))


if __name__ == "__main__":
    main()
