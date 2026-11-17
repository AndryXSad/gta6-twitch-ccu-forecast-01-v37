# DESIGN.md — gta6-twitch-ccu-forecast-01-v37

## What this program does

Reads one input case (a launch's hour-1 Twitch viewers, its pre-release
7-day mean viewers, and a library of comparable past launches) and writes
the 5/25/50/75/95% quantiles of concurrent viewers for hours 2–24 of
release day. Offline; no network or model calls. Python 3.13, standard
library only.

## Members (five predictors, all validated to 0.000000 vs cached rows)

Each member is a per-hour (log-median, log-spread) pair, built from two
kinds of view over the 38-launch comparable library:

- **OLS view**: per-hour OLS on `[lnV1, lnPREM, g12]` (or a per-member
  feature subset), over comparables within ±2.5 log units of the target,
  with an extrapolation guard. Residual sd is the spread.
- **Shape view**: a median log-ratio curve (hour / hour 1) over a pool,
  then per-launch level `c` + amplitude `s` regressed on features;
  forecast = `lnV1 + c + s · template[h]`; spread = per-hour residual sd.
  Pool window (windowed vs whole-library), sd convention (population vs
  sample), fit-hour minimum, and early-access filter differ per member.

The members differ in exactly those small choices:

| member | shape pool | shape features | spread blend |
|---|---|---|---|
| y5 | tier EA filter, windowed, fit_hours 10, min_fits 6, population sd | [lnV1, g12] | sqrt-variance blend of OLS / console-shape / all-shape views |
| z0 | naive EA filter, windowed, fit_hours 20, min_fits 5, sample sd | [lnV1, g12] | same family |
| y1 | naive EA filter, windowed, fit_hours 20, min_fits 5, sample sd | [lnV1, lnPREM] | same family |
| clone | hyphen EA filter, whole-library, fit_hours 20, min_fits 5, population sd | [lnV1, lnPREM, g12] | same family |
| mv | per-hour median across the 4 views (OLS, console shape, all shape, clone); spread = mean of the four view spreads | — | — |

The spread blend in the v4 family is **sqrt-variance**:
`sig = sqrt(sum(w·sig²)/sum(w))`, not a linear mean. The OLS pool is
counted before the hour filter in the v4 family; the OLS fallback is
`(lnV1, 0.5)`.

## The packaged cell (v37)

v36 base, plus three new levers:

1. **Convex ramp profiles (new, the main lever).** Within-band progress
   is `frac ** 1.5` instead of linear `frac`, so each band's ramp acts
   late inside the band. This is the first lever since v24 that improves
   practice AND both LOO gates AND the median-only score at once.
   (Quadratic ramps were tried in v36: LOO great, practice too costly.
   The milder 1.5 exponent splits the difference.)
2. **Per-band disagreement scales (new).** The disagreement measure
   (range/2 in mid, population sd in early/late) is scaled ×0.8 in
   early, ×1.1 in mid, ×1.0 in late. It enters the location shift, the
   spread factor, and the W5 interaction consistently.
3. **Per-band size shifts (new).** The size shift
   `−0.015·max(0, lnV1−11.5)`-family term is now −0.06 in early,
   −0.04 in mid, −0.05 in late (was −0.04 everywhere).

Full cell: shift `(0.60 + 0.33·size)` early, `(0.75 + 0.33·size)` mid,
late base 0.45; early ramp +0.070, mid ramp −0.19, late ramp +0.05
(all on the convex frac); early median Y1, mid median
median(Y5, clone, mv), late median median(Y5, Z0, Y1); k_dis 0.80;
spread `0.975×(0.45·sigY5 + 0.45·sigZ0 + 0.05·sigClone + 0.05·sigMV)`;
late spread ramp 0.975→0.84 (REPLACES the 0.975 base factor in the late
band); W5 `spread *= 1 + 0.9·size·sd`; W1 `sig_lo *= 1 + 0.075·size`;
asym (0.90,1.10) / (1.10,0.85) / (0.95,0.85); cap 0.315; disagreement
sd = mixed per band (range/2 mid, population sd early/late), with the
per-band scales above. `size = max(0, lnV1 − 11.5)`.

## Scores (exact RULES scorer, "Average viewers")

Practice **0.115593** (p00 0.105288 / p01 0.134826 / p02 0.075355 /
p03 0.146901), LOO-large **0.207430**, LOO-mega **0.208763**,
median-only practice **0.177239**, LOO-all **0.486372**. Beats v36
(0.115873 / 0.207760 / 0.209060) on all three gates; best worst-case
margin +0.000280. Direct grade of this file reproduces the cached-row
score exactly.

## Honest caveats

- **Gains are noise-level.** Worst-case margin over v36 is +0.000280;
  the round's micro-variants span ±0.00003 around it. The frontier is
  flat; v37 is a real but small step.
- **p03 worsens.** The largest practice launch goes 0.145330 → 0.146901
  (+0.001571) while p00/p01/p02 all improve. The final launch is larger
  still (lnV1 ~14+), so the real final gain may be smaller than the
  practice number suggests.
- **Median-only LOO-large is still worse than v36** (0.317446 vs
  0.315461): part of the LOO pinball gain is spread placement, not
  median location. The practice median DID improve (0.178255 → 0.177239)
  and LOO-all improved (0.486764 → 0.486372), which is why this cell
  was picked over spread-only alternatives.
- **Extrapolation risk.** Slope 0.33 and mid base 0.75 extrapolate
  hardest at the final launch's size; the convex ramp concentrates the
  mid ramp's −0.19 at the end of the band (hours 12–14).
- **Practice board context.** The live practice leaderboard is dominated
  by replay/memorization entries at loss 1e-06; this entry's honest
  0.115593 sits far outside the top 10. Per the owner's rule it is NOT
  submitted; it is kept as the best verified fallback.
- The four practice libraries are identical (same 38 launches/order);
  the final case carries 42 comparables. Band parameters were selected
  on LOO + practice across many rounds, so neither gate is a clean
  holdout.
