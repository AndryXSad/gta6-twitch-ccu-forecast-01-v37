"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

Design (v37, built on the verified v36 cell — see DESIGN.md): five
member predictors are computed from the case's comparable launches,
then combined per hour into a log-median and asymmetric log-spreads,
which become the five quantiles.

Members (each returns {hour: (log-median, log-spread)} for hours
2..24):
  y5    — v4-family blend (OLS view + console/all shape views, tier
          EA filter, shape features [lnV1, g12])
  z0    — v4-family blend (naive EA filter, shape features
          [lnV1, g12])
  y1    — v4-family blend (naive EA filter, shape features
          [lnV1, lnPREM])
  clone — v9-clone blend (whole-library shape pools, sample sd,
          hyphen EA filter, shape features [lnV1, lnPREM, g12])
  mv    — per-hour median across four views (OLS, console shape,
          all shape, clone forecast); spread = mean of the four
          views' spreads. Added in v33 as the mid-band median
          source (median of Y5, clone, mv) and the fifth member in
          the disagreement range.

The v4-family spread blend is a sqrt-variance blend, not a linear
mean (recovered in v32 from the harness that generated the cached
rows); the shape views differ per member in pool window, fit hours,
sd convention, OLS pool order and EA filter — see DESIGN.md.

Combine (v36 cell): per-band median source (early = Y1, mid =
median(Y5, clone, mv), late = median(Y5, Z0, Y1)), per-band ramps,
size shift, size-indexed disagreement shift where the disagreement
measure is MIXED per band — range (max - min) / 2 in the mid band,
population sd in the early and late bands — 3-way spread blend with
disagreement factor, late spread ramp, asymmetric scales, spread
cap 0.315, W5 size*disagreement and W1 lower-side factors.

Honest caveat: every combine parameter was selected on the same
four practice launches and the same 37-launch LOO as every round
since v22, so neither gate is a clean holdout; the final launch
(lnV1 ~14+) sits beyond the library max (13.68), so the
size-indexed terms extrapolate hardest exactly there. This is the
best verified fallback, not a guarantee. See DESIGN.md for the
full caveat.

Program contract (RULES.txt): reads $MAKEFUN_INPUT_DIR (else
/input), one *.in file per case; writes $MAKEFUN_OUTPUT_DIR (else
/output), one *.out file per case: {"hours": [{"hour": h, "q":
[q05, q25, q50, q75, q95]}]} for hours 2..24. Stdlib only, no
network, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_VALUES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)
HOURS = tuple(range(2, 25))
KINDS = ("y5", "z0", "y1", "clone", "mv")


# ------------------------------------------------------------------ numerics

def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _sd_pop(values):
    """Population sd (ddof=0) — the v4 family's convention."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _sd_sample(values):
    """Sample sd (ddof=1) — the clone's convention."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _solve_linear(mat, rhs):
    """Gauss-Jordan elimination with partial pivoting; None if singular.

    Every other row is eliminated at each pivot step, so on return the
    system is diagonal and each unknown is just aug[i][n]/aug[i][i].
    """
    n = len(rhs)
    aug = [list(mat[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-11:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col] / aug[col][col]
            if factor == 0.0:
                continue
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def _least_squares(design, ys, ridge=1e-9):
    """Least squares via normal equations; None if singular."""
    p = len(design[0])
    gram = [[sum(r[i] * r[j] for r in design)
             + (ridge if i == j else 0.0) for j in range(p)]
            for i in range(p)]
    vec = [sum(r[i] * y for r, y in zip(design, ys)) for i in range(p)]
    return _solve_linear(gram, vec)


def _dot(beta, row):
    return sum(b * x for b, x in zip(beta, row))


# ---------------------------------------------------------------------- data

def _parse_time(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def _hourly_maps(points):
    """({datetime: viewers}, {datetime: channels}) for positive points."""
    viewers, channels = {}, {}
    for pt in points:
        val = pt.get("viewers")
        if val is None or val <= 0:
            continue
        t = _parse_time(pt["from"])
        viewers[t] = float(val)
        ch = pt.get("channels")
        if ch is not None and ch > 0:
            channels[t] = float(ch)
    return viewers, channels


def _roughness(vmap, day_start):
    """Sd of hour-to-hour log-diffs over the 7 days before release."""
    start = _parse_time(day_start)
    pts = sorted((t, v) for t, v in vmap.items()
                 if start - timedelta(days=7) <= t < start)
    diffs = []
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if (t1 - t0).total_seconds() == 3600:
            diffs.append(math.log(v1) - math.log(v0))
    return _sd_pop(diffs) if len(diffs) >= 10 else None


def _launch_features(vmap, day_start, hour1):
    """lnV1, lnPREM, g12, pre_rough for a launch, or None."""
    start = _parse_time(day_start)
    prior = [v for t, v in vmap.items()
             if start - timedelta(days=7) <= t < start]
    if not prior:
        return None
    prem = math.log(_mean(prior))
    want = start - timedelta(hours=12)
    best = None
    for t, v in vmap.items():
        gap = abs((t - want).total_seconds())
        if gap <= 3 * 3600 and (best is None or gap < best[0]):
            best = (gap, v)
    g12 = None if best is None else math.log(hour1) - math.log(best[1])
    return {"lnV1": math.log(hour1), "lnPREM": prem, "g12": g12,
            "pre_rough": _roughness(vmap, day_start)}


def _tier_of(note):
    lowered = (note or "").lower()
    if ("early-access-contaminated" in lowered
            or "early-access contaminated" in lowered):
        return "ea"
    if "tier: weak" in lowered:
        return "weak"
    return "comparable"


def _parse_comparable(comp):
    """Feature dict + release-day curves for one comparable, or None."""
    vmap, _cmap = _hourly_maps(comp["hourly"])
    start = _parse_time(comp["day_start"])
    v1 = vmap.get(start)
    if v1 is None:
        return None
    curve = {}
    for h in range(1, 25):
        t = start + timedelta(hours=h - 1)
        if t in vmap:
            curve[h] = vmap[t]
    if sum(1 for h in curve if h >= 2) < 20:
        return None
    feats = _launch_features(vmap, comp["day_start"], v1)
    if feats is None:
        return None
    feats["curve"] = curve
    feats["name"] = comp["name"]
    feats["console"] = "PC" not in comp.get("platforms", "")
    note = comp.get("unlock", "") or ""
    feats["tier"] = _tier_of(note)
    feats["space_ea"] = "early access" in note.lower()
    feats["hyphen_ea"] = "early-access" in note.lower()
    return feats


def _parse_target(case):
    """Target features for the case's own launch."""
    vmap, _cmap = _hourly_maps(case["history_hourly"])
    v1 = max(1.0, float(next(k for k in case["known"]
                             if k["hour"] == 1)["viewers"]))
    feats = _launch_features(vmap, case["day_start"], v1)
    if feats is None:
        feats = {"lnV1": math.log(v1), "lnPREM": math.log(v1),
                 "g12": 0.0, "pre_rough": None}
    feats["curve"] = {}
    return feats


def _load_case(case):
    """(target feats, usable library) for one case dict."""
    tgt = _parse_target(case)
    lib = []
    for comp in case["comparables"]:
        if "before release" in comp["name"]:
            continue
        parsed = _parse_comparable(comp)
        if parsed is not None:
            lib.append(parsed)
    for key in ("g12", "pre_rough"):
        vals = sorted(f[key] for f in lib if f[key] is not None)
        if not vals:
            continue
        fallback = vals[len(vals) // 2]
        if tgt[key] is None:
            tgt[key] = fallback
        for f in lib:
            if f[key] is None:
                f[key] = fallback
    return tgt, lib


# ---------------------------------------------------------------------- views

def _span(lib):
    return min(f["lnV1"] for f in lib), max(f["lnV1"] for f in lib)


def _design_row(feats, lo, hi, cols):
    """Design row; size clamped to the library range + 0.5 headroom."""
    size = _clamp(feats["lnV1"], lo, hi + 0.5)
    vals = {"lnV1": size, "lnPREM": feats["lnPREM"],
            "g12": feats["g12"]}
    return [1.0] + [vals[c] for c in cols]


def _nearby(lib, lnv1, half=2.5):
    """Launches within +/-half of the target size; whole library when
    fewer than 10 qualify."""
    close = [f for f in lib if abs(f["lnV1"] - lnv1) <= half]
    return close if len(close) >= 10 else list(lib)


def _ols_view(lib, tgt, hour, window=2.5,
              cols=("lnV1", "lnPREM", "g12"), mult=0.8,
              population=True, count_after=False):
    """Per-hour OLS view -> (log-median, log-spread) or None.

    Pool = window-filtered launches with this hour in their curve;
    centre recentered on the median residual; spread = residual sd x
    mult. population=True uses population sd (v4 family); False uses
    sample sd (clone). count_after=True counts the pool AFTER the
    hour filter (the clone's rule); False counts the window first
    (v4's).
    """
    if count_after:
        pool = [f for f in lib if hour in f["curve"]
                and abs(f["lnV1"] - tgt["lnV1"]) <= window]
        if len(pool) < 10:
            pool = [f for f in lib if hour in f["curve"]]
    else:
        pool = [f for f in _nearby(lib, tgt["lnV1"], window)
                if hour in f["curve"]]
    if len(pool) < 8:
        return None
    lo, hi = _span(lib)
    design = [_design_row(f, lo, hi, cols) for f in pool]
    ys = [math.log(f["curve"][hour]) for f in pool]
    beta = _least_squares(design, ys)
    if beta is None:
        return None
    resid = [y - _dot(beta, r) for r, y in zip(design, ys)]
    sd = _sd_pop(resid) if population else _sd_sample(resid)
    mu = _dot(beta, _design_row(tgt, lo, hi, cols)) + _median(resid)
    return mu, sd * mult


def _ea_filter(base, kind):
    """EA filter on a pool: 'tier' (tier != ea), 'naive' (no SPACE
    'early access' in the raw text), 'hyphen' (no 'early-access')."""
    if kind == "tier":
        return [f for f in base if f["tier"] != "ea"]
    if kind == "naive":
        return [f for f in base if not f["space_ea"]]
    return [f for f in base if not f["hyphen_ea"]]


def _shape_view(lib, tgt, console_only, feats=("lnV1", "lnPREM", "g12"),
                mult=1.0, population=True, windowed=False,
                ea_kind="hyphen", fit_hours=20, min_fits=5):
    """One-factor shape view -> {hour: (log-median, log-spread)}.

    Median log-ratio template over the pool; per-launch level c +
    amplitude s regressed on `feats`; amplitude clamped [0, 2];
    spread = per-hour residual sd x mult.
    """
    base = _nearby(lib, tgt["lnV1"]) if windowed else list(lib)
    pool = _ea_filter(base, ea_kind)
    if console_only:
        pool = [f for f in pool if f["console"]]
    if len(pool) < 6:
        pool = _ea_filter(base, ea_kind)
    if len(pool) < 6:
        return None
    template = {}
    for h in HOURS:
        vals = [math.log(f["curve"][h] / f["curve"][1]) for f in pool
                if h in f["curve"] and 1 in f["curve"]]
        template[h] = _median(vals) if vals else 0.0
    fits = []
    for f in pool:
        xs = [template[h] for h in HOURS if h in f["curve"]]
        ys = [math.log(f["curve"][h] / f["curve"][1]) for h in HOURS
              if h in f["curve"]]
        if len(xs) < fit_hours:
            continue
        mx, my = _mean(xs), _mean(ys)
        den = sum((x - mx) ** 2 for x in xs)
        slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
                 if den else 0.0)
        fits.append((f, my - slope * mx, slope))
    if len(fits) < min_fits:
        return None
    lo, hi = _span(lib)
    design = [_design_row(f, lo, hi, feats) for f, _, _ in fits]
    beta_c = _least_squares(design, [c for _, c, _ in fits])
    beta_s = _least_squares(design, [s for _, _, s in fits])
    if beta_c is None or beta_s is None:
        return None
    xt = _design_row(tgt, lo, hi, feats)
    c_star = _dot(beta_c, xt)
    s_star = _clamp(_dot(beta_s, xt), 0.0, 2.0)
    out = {}
    for h in HOURS:
        res = [math.log(f["curve"][h] / f["curve"][1])
               - (c + s * template[h])
               for f, c, s in fits if h in f["curve"]]
        sd = _sd_pop(res) if population else _sd_sample(res)
        out[h] = (tgt["lnV1"] + c_star + s_star * template[h], sd * mult)
    return out


# ------------------------------------------------------------------ predictors

def _v4_member(lib, tgt, ea_kind, feats, mix=(0.5, 0.2, 0.3)):
    """Per-hour (log-median, log-spread) for one v4-family member,
    using the sqrt-variance spread blend."""
    con = _shape_view(lib, tgt, True, feats=feats, mult=1.0,
                      population=True, windowed=True,
                      ea_kind=ea_kind, fit_hours=10, min_fits=6)
    all_ = _shape_view(lib, tgt, False, feats=feats, mult=1.0,
                       population=True, windowed=True,
                       ea_kind=ea_kind, fit_hours=10, min_fits=6)
    out = {}
    for h in HOURS:
        views = []
        ov = _ols_view(lib, tgt, h, window=2.5, mult=0.8,
                       population=True)
        if ov is not None:
            views.append((mix[0], ov))
        else:
            views.append((mix[0], (tgt["lnV1"], 0.5)))
        if con is not None and h in con:
            views.append((mix[1], con[h]))
        if all_ is not None and h in all_:
            views.append((mix[2], all_[h]))
        tot = sum(w for w, _ in views)
        mu = sum(w * v[0] for w, v in views) / tot
        sig = math.sqrt(max(0.02, sum(w * v[1] ** 2 for w, v in views)
                            / tot))
        out[h] = (mu, sig)
    return out


def _clone_member(lib, tgt, weights=(0.6, 0.2, 0.2), spread_scale=0.81):
    """Per-hour (log-median, log-spread) for the v9-clone baseline."""
    con = _shape_view(lib, tgt, True,
                      feats=("lnV1", "lnPREM", "g12"),
                      population=False, ea_kind="hyphen",
                      fit_hours=20, min_fits=5)
    all_ = _shape_view(lib, tgt, False,
                       feats=("lnV1", "lnPREM", "g12"),
                       population=False, ea_kind="hyphen",
                       fit_hours=20, min_fits=5)
    out = {}
    for h in HOURS:
        views = []
        ov = _ols_view(lib, tgt, h, window=2.5, population=False,
                       count_after=True)
        if ov is not None:
            views.append((weights[0], ov))
        if con is not None and h in con:
            views.append((weights[1], con[h]))
        if all_ is not None and h in all_:
            views.append((weights[2], all_[h]))
        if not views:
            out[h] = (tgt["lnV1"], 0.5)
            continue
        tot = sum(w for w, _ in views)
        mu = sum(w * v[0] for w, v in views) / tot
        sig = spread_scale * sum(w * v[1] for w, v in views) / tot
        out[h] = (mu, sig)
    return out


def _mv_member(lib, tgt):
    """Per-hour median across four views (the 'views' member):
    per-hour OLS (v4-family), console shape (v4, 3 feats), all shape
    (v4, 3 feats), and the clone forecast. Spread = mean of the four
    views' spreads. Validated to 0.000000 against the cached rows
    from the round that first built this member."""
    con = _shape_view(lib, tgt, True,
                      feats=("lnV1", "lnPREM", "g12"),
                      population=True, windowed=True,
                      ea_kind="hyphen", fit_hours=10, min_fits=6)
    all_ = _shape_view(lib, tgt, False,
                       feats=("lnV1", "lnPREM", "g12"),
                       population=True, windowed=True,
                       ea_kind="hyphen", fit_hours=10, min_fits=6)
    cl = _clone_member(lib, tgt)
    out = {}
    for h in HOURS:
        mus, sigs = [], []
        ov = _ols_view(lib, tgt, h, window=2.5, mult=0.8,
                       population=True)
        if ov is not None:
            mus.append(ov[0])
            sigs.append(ov[1])
        if con is not None and h in con:
            mus.append(con[h][0])
            sigs.append(con[h][1])
        if all_ is not None and h in all_:
            mus.append(all_[h][0])
            sigs.append(all_[h][1])
        mus.append(cl[h][0])
        sigs.append(cl[h][1])
        out[h] = (_median(mus), _mean(sigs))
    return out


def _member(kind, lib, tgt):
    """{hour: (log-median, log-spread)} for one member predictor."""
    if kind == "y5":
        return _v4_member(lib, tgt, "tier", ("lnV1", "g12"))
    if kind == "z0":
        return _v4_member(lib, tgt, "naive", ("lnV1", "g12"))
    if kind == "y1":
        return _v4_member(lib, tgt, "naive", ("lnV1", "lnPREM"))
    if kind == "clone":
        return _clone_member(lib, tgt)
    if kind == "mv":
        return _mv_member(lib, tgt)
    raise ValueError(kind)


# ------------------------------------------------------------------- combine

def _band_of(hour):
    return "early" if hour <= 6 else "mid" if hour <= 14 else "late"


def _band_frac(hour):
    """Within-band progress, convex: frac ** 1.5 (v37 lever). Ramps
    act late inside each band, which buys LOO and practice at once
    (v36's linear frac was flat on this axis)."""
    bounds = {"early": (2, 6), "mid": (7, 14), "late": (15, 24)}
    lo, hi = bounds[_band_of(hour)]
    frac = (hour - lo) / (hi - lo) if hi > lo else 1.0
    return frac ** 1.5


def _disagreement(members, band):
    """Mixed per-band disagreement over the five member medians:
    range (max - min) / 2 in the mid band, population sd in the
    early and late bands."""
    mus = [members[k][0] for k in KINDS]
    if band == "mid":
        # range (max - min) / 2, scaled x1.1 (v37 lever)
        return (max(mus) - min(mus)) / 2.0 * 1.1
    if band == "early":
        # population sd, scaled x0.8 (v37 lever)
        return _sd_pop(mus) * 0.8
    return _sd_pop(mus)


def _combine(members, ln_v1, hour):
    """(log-median, lower log-spread, upper log-spread) for one hour.

    members: {kind: (mu, sig)} for this hour. v37 cell (v36 base +
    convex ramps frac**1.5, per-band disagreement scales 0.8/1.1/1.0,
    per-band size shifts -0.06/-0.04/-0.05, slope 0.33, k_dis 0.80):
    shift (0.60 + 0.33*size) in early, (0.75 + 0.33*size) in mid,
    late base 0.45; ramps early +0.070 / mid -0.19 / late +0.05;
    early median Y1, mid median median(Y5, clone, mv), late
    median median(Y5, Z0, Y1); spread
    0.975*(0.45*sigY5 + 0.45*sigZ0 + 0.05*sigClone + 0.05*sigMV);
    late spread ramp 0.975 -> 0.84 (replaces the 0.975 base factor
    in the late band); W5 spread *= 1 + 0.9*size*sd; W1 sig_lo *=
    1 + 0.075*size; asym (0.90,1.10)/(1.10,0.85)/(0.95,0.85);
    cap 0.315; disagreement is the mixed per-band measure above.
    """
    mus = {k: members[k][0] for k in KINDS}
    sigs = {k: members[k][1] for k in KINDS}
    band = _band_of(hour)
    sdv = _disagreement(members, band)
    size = max(0.0, ln_v1 - 11.5)
    frac = _band_frac(hour)
    if band == "early":
        mu = mus["y1"]
        ramp = 0.070 * frac
        base = 0.60
    elif band == "mid":
        mu = _median([mus["y5"], mus["clone"], mus["mv"]])
        ramp = -0.19 * frac
        base = 0.75
    else:
        mu = _median([mus["y5"], mus["z0"], mus["y1"]])
        ramp = 0.05 * frac
        base = 0.45
    size_shift = -0.06 if band == "early" else -0.05 if band == "late" else -0.04
    mu = mu + ramp + size_shift * size
    mu = mu + (base + 0.33 * size) * sdv
    spread = 0.975 * (0.45 * sigs["y5"] + 0.45 * sigs["z0"]
                      + 0.05 * sigs["clone"] + 0.05 * sigs["mv"])
    spread = spread * (1.0 + 0.80 * sdv)
    if band == "late":
        late_factor = 0.975 + (0.84 - 0.975) * frac
        spread = spread / 0.975 * late_factor
    asym = {"early": (0.90, 1.10), "mid": (1.10, 0.85),
            "late": (0.95, 0.85)}[band]
    slo = min(spread * asym[0], 0.315)
    shi = min(spread * asym[1], 0.315)
    slo = min(slo * (1.0 + 0.075 * size), 0.315)
    factor = 1.0 + 0.9 * size * sdv
    slo = min(slo * factor, 0.315)
    shi = min(shi * factor, 0.315)
    return (mu, slo, shi)


# --------------------------------------------------------------------- output

def _quantiles(mu_sig):
    """Five quantiles (viewers) from (log-median, lower, upper spread)."""
    mu, slo, shi = mu_sig
    out = []
    for z in Z_VALUES:
        sig = slo if z < 0 else shi if z > 0 else 0.0
        out.append(math.exp(mu + z * sig))
    return out


def _valid_output(doc):
    """True iff doc is a valid forecast file per RULES.txt."""
    if not isinstance(doc, dict) or set(doc.keys()) != {"hours"}:
        return False
    rows = doc["hours"]
    if not isinstance(rows, list) or len(rows) != 23:
        return False
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row.keys()) != {"hour", "q"}:
            return False
        h = row["hour"]
        if not isinstance(h, int) or h < 2 or h > 24 or h in seen:
            return False
        seen.add(h)
        q = row["q"]
        if not isinstance(q, list) or len(q) != 5:
            return False
        prev = 0.0
        for val in q:
            if (not isinstance(val, (int, float))
                    or not math.isfinite(val)
                    or val <= 0.0 or val > 1e9 or val < prev):
                return False
            prev = val
    return seen == set(range(2, 25))


def _house_fallback(hour1):
    """House baseline forecast (RULES.txt) — used only if the main
    forecast ever fails the output contract."""
    p = max(1.0, float(hour1))
    rows = []
    for h in HOURS:
        q = [p * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z_VALUES]
        rows.append({"hour": h, "q": q})
    return {"hours": rows}


def _forecast_case(case):
    """{"hours": [...]} for one case dict."""
    tgt, lib = _load_case(case)
    members = {k: _member(k, lib, tgt) for k in KINDS}
    ln_v1 = tgt["lnV1"]
    rows = []
    for h in HOURS:
        cell = {k: members[k][h] for k in KINDS}
        q = _quantiles(_combine(cell, ln_v1, h))
        q = [min(max(v, 1.0), 1e9) for v in q]
        for i in range(1, 5):
            if q[i] < q[i - 1]:
                q[i] = q[i - 1]
        rows.append({"hour": h, "q": q})
    doc = {"hours": rows}
    if _valid_output(doc):
        return doc
    hour1 = max(1.0, math.exp(ln_v1))
    return _house_fallback(hour1)


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        doc = _forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w",
                  encoding="utf-8") as fh:
            json.dump(doc, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
