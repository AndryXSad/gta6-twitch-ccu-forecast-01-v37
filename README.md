# gta6-twitch-ccu-forecast-01-v37

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/AndryXSad/gta6-twitch-ccu-forecast-01-v37/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/AndryXSad/gta6-twitch-ccu-forecast-01-v37/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A make.fun `gta6-twitch-ccu-forecast-01` entry. It forecasts, for each of
hours 2–24 of GTA VI's launch day, the 5/25/50/75/95% quantiles of Twitch
concurrent viewers, from one input case (the launch's hour-1 viewers, its
pre-release 7-day mean, and comparable past launches). Offline, no network
or model calls. See `DESIGN.md` for the method.

## Scores (exact RULES scorer, "Average viewers")

| metric | v37 | v36 |
|---|---|---|
| practice pinball loss | **0.115593** | 0.115873 |
| LOO-large (lnV1 ≥ 11.0) | **0.207430** | 0.207760 |
| LOO-mega (lnV1 ≥ 11.5) | **0.208763** | 0.209060 |
| median-only practice | **0.177239** | 0.178255 |
| LOO-all | **0.486372** | 0.486764 |

Per-case practice: p00 0.105288 / p01 0.134826 / p02 0.075355 /
p03 0.146901. All five member predictors reproduce the cached
validation rows to 0.000000 on practice and LOO.

## Method in one paragraph

Five member predictors (y5, z0, y1, clone, mv) are built from per-hour
OLS views and one-factor shape views over 38 comparable launches, then
combined per hour-band with per-band median sources, per-band
asymmetric spread scales, a spread cap, and disagreement-driven
location/spread shifts. New in v37: **convex within-band ramp
profiles** (`frac ** 1.5` — ramps act late inside each band), per-band
disagreement scales (0.8 early / 1.1 mid / 1.0 late), and per-band size
shifts (−0.06 / −0.04 / −0.05). See `DESIGN.md` for the full cell and
the honest caveats.

## Run

```sh
python3 main.py <case.json> <output.json>
```

`main.py` reads one case, writes quantiles for hours 2–24.

## License

MIT — see `LICENSE`.
