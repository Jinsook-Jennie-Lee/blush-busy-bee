#!/usr/bin/env python3
"""Generate dummy RHEED frames for InScN-on-GaN growth runs, one frame every N minutes.

The images are synthetic. They mimic the look of the [10-10] zone-axis patterns in
Lonergan et al., J. Appl. Phys. 139, 125301 (2026), figures 1(g)-(k) and 4(b):

* vertical streaks above a shadow edge, spacing inversely proportional to the in-plane
  lattice constant a;
* streaks that break into spots along their length as the surface roughens;
* for Sc > ~10.5 %, a second set of spots (twinned rock-salt InScN) that appears part way
  through the growth and strengthens with time.

Every frame is written with its ground truth to labels.csv so a model can be trained
and scored against known answers. Values for a, roughness and the phase boundary are
interpolated from the paper's six samples; everything in between is a guess.

Usage:
    python scripts/simulate_rheed.py --out data/synthetic --interval 5 --duration 60
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Material model, from the paper (Sc % -> property). Linear interpolation between points.
# ---------------------------------------------------------------------------
A_GAN = 3.189  # Å, in-plane lattice constant of the GaN substrate/buffer
_SC = np.array([0.0, 4.2, 6.8, 7.9, 10.4, 15.1])
_A = np.array([3.521, 3.530, 3.540, 3.520, 3.545, 3.545])  # Å, fig. 3(c); 15.1 % extrapolated
_RMS = np.array([0.324, 0.270, 2.686, 0.716, 3.628, 6.0])  # nm, AFM; 15.1 % is a guess (delaminated)
RS_ONSET_SC = 10.5  # Sc % above which the rock-salt phase shows up during growth


def a_of_sc(sc: float) -> float:
    return float(np.interp(sc, _SC, _A))


def rms_of_sc(sc: float) -> float:
    return float(np.interp(sc, _SC, _RMS))


def roughness_proxy(rms_nm: float) -> float:
    """Map RMS roughness (nm) to a 0..1 'spottiness' used to draw the streaks."""
    return float(1.0 - math.exp(-rms_nm / 1.5))


def rs_onset_minutes(sc: float) -> float | None:
    """When the rock-salt spots first appear. The paper saw them at 14 min for 15.1 %."""
    if sc < RS_ONSET_SC:
        return None
    # earlier onset at higher Sc: 15.1 % -> 14 min, 10.5 % -> ~45 min
    return float(np.interp(sc, [RS_ONSET_SC, 15.1, 20.0], [45.0, 14.0, 5.0]))


# ---------------------------------------------------------------------------
# Image synthesis
# ---------------------------------------------------------------------------
H, W = 480, 640
CX = W // 2
SHADOW_Y = 330  # row of the shadow edge (sample horizon)
K_PX = 110.0 * A_GAN  # so that GaN first-order streaks sit 110 px from the centre


def streak_spacing_px(a: float) -> float:
    return K_PX / a


def _gauss2d(yy, xx, y0, x0, sy, sx):
    return np.exp(-0.5 * (((yy - y0) / sy) ** 2 + ((xx - x0) / sx) ** 2))


def _draw_streaks(img, yy, xx, spacing, rough, amp, rng, orders=(0, 1, 2)):
    """Vertical streaks at +-order*spacing. Roughness breaks them into spots."""
    top = 40
    length = SHADOW_Y - top
    for n in orders:
        for sgn in ((1,) if n == 0 else (1, -1)):
            x0 = CX + sgn * n * spacing
            if x0 < 5 or x0 > W - 5:
                continue
            order_amp = amp * (1.0 if n == 0 else 0.75 / n)
            width = 2.2 + 0.6 * n
            # Smooth streak, brighter near the shadow edge.
            profile = np.clip((yy - top) / length, 0, 1) ** 0.6
            streak = order_amp * profile * np.exp(-0.5 * ((xx - x0) / width) ** 2)
            streak[(yy < top) | (yy > SHADOW_Y)] = 0
            # Roughness: modulate along the streak so it breaks into spots.
            if rough > 0.02:
                period = 38.0
                phase = rng.uniform(0, 2 * math.pi)
                mod = 0.5 * (1 + np.cos(2 * math.pi * yy / period + phase))
                streak = streak * ((1 - rough) + rough * (mod ** 2) * 2.0)
                # widen into blobs as it gets rough
                blob = np.exp(-0.5 * ((xx - x0) / (width * (1 + 2.5 * rough))) ** 2)
                streak = streak * (1 - rough) + order_amp * rough * (mod ** 2) * blob * profile
                streak[(yy < top) | (yy > SHADOW_Y)] = 0
            img += streak
    return img


def _draw_rocksalt_spots(img, yy, xx, spacing, strength, rng):
    """Two twinned rock-salt domains: a 3D spot net, like fig. 4(b) of the paper."""
    # Spot positions on a rough hexagon around the specular area.
    dx = spacing * 0.78
    dy = 70.0
    spots = [(CX, SHADOW_Y - 150)]
    for i, j in [(1, 1), (-1, 1), (1, -1), (-1, -1), (2, 0), (-2, 0), (0, 2), (1, 3), (-1, 3)]:
        spots.append((CX + i * dx * 0.9, SHADOW_Y - 150 - j * dy * 0.75))
    for x0, y0 in spots:
        if 5 < x0 < W - 5 and 20 < y0 < SHADOW_Y - 5:
            amp = strength * rng.uniform(0.6, 1.0)
            img += amp * _gauss2d(yy, xx, y0, x0, 9.0, 9.0)
    return img


def render_frame(sc: float, t_min: float, duration: float, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Render one frame and return (uint8 image HxW, label dict)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W), dtype=np.float32)

    # --- what is on the surface at time t ---------------------------------
    # 0-2 min: GaN buffer streaks; 2-8 min: InScN streaks take over (pseudomorphic ->
    # relaxed, spacing drifts from GaN to InScN); then InScN.
    a_final = a_of_sc(sc)
    if t_min <= 2:
        f = 0.0
    elif t_min >= 8:
        f = 1.0
    else:
        f = (t_min - 2) / 6.0
    a_now = A_GAN + f * (a_final - A_GAN)
    spacing = streak_spacing_px(a_now)

    # roughness ramps from the smooth buffer value to the film value, with a bump during
    # the first minutes of nucleation
    rough_film = roughness_proxy(rms_of_sc(sc))
    nucleation_bump = 0.35 * math.exp(-((t_min - 5) / 3.0) ** 2) if 0 < t_min < 12 else 0.0
    rough = min(1.0, f * rough_film + (1 - f) * roughness_proxy(0.3) + nucleation_bump)

    # rock-salt phase strength, 0..1
    onset = rs_onset_minutes(sc)
    if onset is None or t_min < onset:
        rs = 0.0
    else:
        rs = min(1.0, (t_min - onset) / max(5.0, 0.6 * (duration - onset)))
    phase = "wurtzite" if rs == 0 else ("rs_onset" if rs < 0.3 else "mixed_rs")

    # --- draw --------------------------------------------------------------
    amp = 0.95 * (1 - 0.5 * rs)
    img = _draw_streaks(img, yy, xx, spacing, rough, amp, rng)
    if rs > 0:
        img = _draw_rocksalt_spots(img, yy, xx, spacing, 0.9 * rs, rng)

    # specular spot near the shadow edge on the 00 streak
    img += 0.6 * (1 - 0.7 * rough) * _gauss2d(yy, xx, SHADOW_Y - 18, CX, 7, 5)

    # diffuse background: glow above the shadow edge + roughness haze
    glow = 0.10 * np.exp(-((yy - (SHADOW_Y - 60)) / 120.0) ** 2)
    glow *= np.exp(-((xx - CX) / 260.0) ** 2)
    img += glow * (1 + 1.5 * rough)
    # faint Kikuchi-like arcs
    r = np.sqrt((xx - CX) ** 2 + (yy - (SHADOW_Y + 120)) ** 2)
    img += 0.04 * (1 - rough) * np.exp(-((r - 300) / 8.0) ** 2)

    # below the shadow edge: dark with a soft edge
    img[yy > SHADOW_Y] *= 0.05
    img += 0.03

    # drift in overall brightness and slight camera jitter
    img *= rng.uniform(0.85, 1.05)
    shift = int(rng.integers(-2, 3))
    img = np.roll(img, shift, axis=1)

    # shot noise + read noise + vignette
    vign = 1 - 0.35 * (((xx - CX) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    img *= vign
    counts = rng.poisson(np.clip(img, 0, None) * 220.0).astype(np.float32)
    counts += rng.normal(0, 3.0, size=counts.shape)
    out = np.clip(counts / 255.0, 0, 1)
    out = (255 * out ** 0.9).astype(np.uint8)

    label = {
        "sc_pct": round(sc, 2),
        "t_min": t_min,
        "a_true_angstrom": round(a_now, 4),
        "streak_spacing_px": round(spacing, 2),
        "rms_roughness_nm_final": round(rms_of_sc(sc), 3),
        "roughness_proxy": round(rough, 3),
        "rocksalt_fraction": round(rs, 3),
        "phase": phase,
    }
    return out, label


def to_green(gray: np.ndarray) -> Image.Image:
    """Phosphor-screen look: green channel carries the signal."""
    g = gray.astype(np.float32)
    rgb = np.stack([0.25 * g, g, 0.35 * g], axis=-1)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def contact_sheet(frames: dict[tuple[float, float], Path], scs: list[float], times: list[float],
                  thumb=(160, 120), out: Path | None = None) -> Image.Image:
    from PIL import ImageDraw

    pad, lab = 4, 18
    sheet = Image.new("RGB", (lab + len(times) * (thumb[0] + pad), lab + len(scs) * (thumb[1] + pad)), (12, 12, 12))
    d = ImageDraw.Draw(sheet)
    for j, t in enumerate(times):
        d.text((lab + j * (thumb[0] + pad) + 4, 3), f"{int(t)} min", fill=(200, 200, 200))
    for i, sc in enumerate(scs):
        d.text((2, lab + i * (thumb[1] + pad) + 4), f"{sc:g}%", fill=(200, 200, 200))
        for j, t in enumerate(times):
            p = frames.get((sc, t))
            if p is None:
                continue
            im = Image.open(p).resize(thumb)
            sheet.paste(im, (lab + j * (thumb[0] + pad), lab + i * (thumb[1] + pad)))
    if out:
        sheet.save(out)
    return sheet


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic", help="output directory")
    ap.add_argument("--interval", type=float, default=5.0, help="minutes between frames")
    ap.add_argument("--duration", type=float, default=60.0, help="growth length in minutes")
    ap.add_argument("--sc", type=float, nargs="*", default=[0, 4.2, 6.8, 7.9, 10.4, 15.1, 2.0, 9.0, 12.5, 13.5],
                    help="Sc mole fractions (%%) to simulate, one run each")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    times = [round(t, 3) for t in np.arange(0, args.duration + 1e-9, args.interval)]

    rows, paths = [], {}
    for sc in args.sc:
        run = out / f"run_sc{sc:05.1f}".replace(".", "p")
        run.mkdir(exist_ok=True)
        for t in times:
            gray, label = render_frame(sc, t, args.duration, rng)
            p = run / f"frame_t{int(t):03d}.png"
            to_green(gray).save(p)
            label = {"run": run.name, "file": str(p.relative_to(out)), **label}
            rows.append(label)
            paths[(sc, t)] = p
        print(f"{run.name}: {len(times)} frames")

    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    contact_sheet(paths, sorted(args.sc), times, out=out / "contact_sheet.png")
    print(f"wrote {len(rows)} frames, labels.csv and contact_sheet.png to {out}")


if __name__ == "__main__":
    main()
