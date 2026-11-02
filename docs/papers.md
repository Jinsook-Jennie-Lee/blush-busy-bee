# Papers behind this project

## 1. Method paper: AI-Accelerated Electronic Materials Discovery and Development

Price, Takeuchi, Rondinelli, Chen, Hinkle. *IEEE Computer* 58(2), Feb 2025, pp. 98–104.
DOI 10.1109/MC.2024.3515392. Paywalled; abstract, figure captions and references are public.

**In plain words.** Making new electronic materials is slow because a person picks a
recipe, grows a sample, measures it, and repeats. The paper argues AI can run that loop:
a model picks the next recipe, a tool grows it, instruments measure it, AI reads the
measurements, the model learns, repeat. Every sample makes the next guess better
(the "flywheel").

Three places AI goes in:

1. **Choosing the next sample.** Bayesian optimization (BO). Their flavour, LVGP
   (latent-variable Gaussian process), handles mixed inputs: numbers such as
   temperature and flux, and categories such as which element or substrate.
   It also reports uncertainty, so it can explore where it knows least.
2. **Reading the instruments.** Neural networks read RHEED patterns during growth,
   electron diffraction, XPS/XAS spectra and spectroscopic ellipsometry faster and more
   consistently than a person.
3. **Simulating first.** DFT and graph neural networks rank candidates before any growth.

Figure 3 of the paper shows a preliminary closed loop: deposit → characterize → model
picks the next trial. Fewer trials reached a good material than a human-led search.

## 2. Material paper: Molecular beam epitaxy and characterization of wurtzite InScN

Lonergan, Savant, Nguyen, Panagiotopoulos, Griffin, Singh, Xing, Jena (Cornell).
*J. Appl. Phys.* 139, 125301, 23 Mar 2026. DOI 10.1063/5.0315898. CC-BY.
PDF: `papers/lonergan-2026-inscn-mbe-jap.pdf`.

**In plain words.** They mixed a little scandium into indium nitride and grew thin films
of it, one atomic layer at a time, on gallium nitride wafers (plasma-assisted MBE,
Veeco Gen10). Nobody had grown this alloy by MBE before.

What they found:

- **Grow it metal-rich.** Extra indium on the surface acts as a surfactant. Films come
  out smooth (roughness under 1 nm) with atomic steps, i.e. a 2D step-flow growth mode.
  This is the opposite of AlScN, which needs nitrogen-rich conditions.
- **Scandium limit is about 10 %.** Up to 10.4 % Sc the crystal stays wurtzite.
  At 15.1 % Sc a second phase (twinned rock-salt InScN) appears. Theory had predicted
  the switch at about 40 %, so the real limit is much lower.
- **Lattice.** The in-plane constant *a* barely moves (3.519–3.545 Å) and between
  about 5 % and 8 % Sc it matches InN. The out-of-plane constant *c* shrinks
  (5.716 → 5.652 Å), so *c/a* drops from 1.623 to 1.594. A falling *c/a* hints at
  larger spontaneous polarization.
- **Crystal quality.** XRD rocking-curve widths stay flat up to 6.8 % Sc, then rise,
  meaning more dislocations at higher Sc.
- **Optics.** The absorption edge rises from 0.8 to 1.05 eV as Sc goes 0 → 10.4 %.
  Refractive index peak stays around 2.95–3.01 but shifts to shorter wavelength.
- **Raman.** The E2(high) peak does not move; the A1(LO) peak climbs from 586 to 597 cm⁻¹.

Why it matters: InN has a tiny band gap and very fast electrons, but its usual alloys
(InGaN, InAlN) separate into patches at high In content. InScN may be a stable new
alloy for InN-based transistors and infrared photonics.

### The sample set (what an AI loop would have to learn from)

| Sc % | RMS roughness (nm) | *a* (Å) | *c* (Å) | *c/a* | abs. edge (eV) | A1(LO) (cm⁻¹) | phase |
|-----:|-------------------:|--------:|--------:|------:|---------------:|--------------:|-------|
| 0    | 0.324 | 3.521 | 5.716 | 1.623 | 0.80 | 586 | wurtzite |
| 4.2  | 0.270 | ~3.53 | ~5.70 | ~1.615 | 0.81 | ~589 | wurtzite |
| 6.8  | 2.686 | ~3.54 | ~5.69 | ~1.61 | 0.83 | ~591 | wurtzite |
| 7.9  | 0.716 | ~3.52 | ~5.66 | ~1.605 | 0.92 | ~594 | wurtzite |
| 10.4 | 3.628 | 3.545 | 5.652 | 1.594 | 1.05 | 597 | wurtzite (faint RS onset in RHEED) |
| 15.1 | delaminated | – | – | – | – | – | wurtzite + twinned rock-salt |

Values marked ~ are read off figures 3 and 5, not stated in the text.

Fixed growth settings in the paper: substrate 400 °C, 160 nm film, N₂ plasma
1.05 sccm at 200 W, In-rich flux, 40 nm n+GaN buffer at 700 °C. The only knob they
swept was the Sc cell temperature. Six samples total.

## 3. How the two fit together

The Computer paper is the method; the InScN paper is the material. The InScN study is a
manual loop with one knob and six samples. The method paper says every step of that
loop can be automated and that the search can have more knobs.

### Candidate AI projects, easiest first

1. **Replay the search with Bayesian optimization (no lab needed).**
   Treat the six published samples as ground truth. Fit a Gaussian process to
   (Sc %) → (roughness, c/a, absorption edge, phase). Ask: if BO had chosen the Sc
   fractions, how many samples would it have needed to (a) locate the wurtzite /
   rock-salt boundary and (b) find the smoothest film with the highest band edge?
   Deliverable: a notebook and a plot of "samples needed, BO vs the paper's sweep".
   Risk: six points is thin; the answer is illustrative, not statistical.

2. **RHEED phase classifier.**
   Figures 1(g)–(k) and 4(b) show streaky wurtzite patterns versus the spotted twinned
   rock-salt pattern. Train a small CNN or use a pretrained vision backbone to classify
   RHEED frames as "2D wurtzite", "3D / spotty", "rock-salt twin onset". With live
   RHEED video this gives an early stop signal 14 min into growth, before the 15 % Sc
   sample delaminates. Needs RHEED frames from the group; the paper has only six.

3. **Ellipsometry and Raman to composition.**
   Fit a model from (n(λ), k(λ), Raman A1(LO)) → Sc % and phase. Cheap, non-destructive
   stand-in for RBS/XPS. Needs the raw spectra, which the supplement partly provides.

4. **Multi-knob closed loop (needs the MBE).**
   Inputs: Sc cell temperature, substrate temperature, In/N ratio, growth rate.
   Targets: roughness, rocking-curve FWHM, c/a, absorption edge, no second phase.
   Run LVGP + BO with uncertainty, as in the method paper's figure 1, to extend the
   wurtzite window past 10.4 % Sc or to hit a target band edge. This is the real
   "update" of the InScN paper.

### Open questions for the owner of this project

- Do we have raw data beyond the PDF: growth logs, RHEED video, XRD/RSM files,
  ellipsometry and Raman spectra?
- Are we working with the Cornell group (new growths possible) or independently
  (analysis of published data only)?
- What is the target: lower roughness, higher Sc without phase change, or a specific
  band edge?
