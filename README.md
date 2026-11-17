[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Jhoni1098/gta6-twitch-ccu-forecast-01-v02/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Jhoni1098/gta6-twitch-ccu-forecast-01-v02/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# gta6-twitch-ccu-forecast-01-v02

Forecast GTA VI's launch-day Twitch viewers hour by hour, for the [make.fun](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) gta6-twitch-ccu-forecast-01 competition. One entrypoint, `main.py` (Python 3.13, standard library only, deterministic, no network): it reads a case's hour-1 print plus the past launches in `comparables` and writes five quantiles of each hour 2–24.

## Model

A launch day is one smooth shape whose amplitude is set by how much of the world is already awake at hour 1: viewers dip in the small hours, jump at the US midnight unlocks, climb through the day, peak in the US evening, then fall overnight. Big launches start high and stay flat; small launches start low and climb several-fold. So the forecast is a per-hour regression in log space, fitted fresh on each case's own comparables:

- Each comparable contributes its release day as log ratios to its own hour-1 print, `ln(v_h / v_1)`.
- The training family is the launches that shape like a rolling midnight console launch: already live at hour 1 (hour-1 print at least 5% of their own peak) and peaking in the evening (peak hour 12 or later). Simultaneous-global launches (near-zero at hour 1, ten-fold explosions), flat launches (unlocked the evening before), early-access launches and the GTA VI category's own pre-release history are all excluded.
- For each hour h = 2..24, one straight line is fitted through the family's log-ratios against `ln(p1)`: `r_h ~ a_h + b_h * ln(p1)`. The slope runs from ~0 in the small hours to about -0.45 at the evening peak, and is steepened 10%.
- The target's median is that line read at its own log hour-1 print, clipped to the training range (GTA VI is bigger than every past launch, so the slope is not extrapolated past the largest observed one). The spread is the cross-launch residual scatter around the line at that hour, clamped to [0.25, 0.85] and scaled by 0.7.
- `q_tau(h) = p1 * exp(mu_h + z_tau * s_h)`, with `z` the normal quantiles at 0.05, 0.25, 0.5, 0.75, 0.95.

Too few training launches falls back to the house baseline's flat persistence forecast, so the output stays valid either way.

## Local practice score (SullyGnome hourly averages as answers)

| Case | Loss | Baseline |
|------|------|----------|
| p00 Marvel's Wolverine | 0.231 | 0.744 |
| p01 Red Dead Redemption 2 | 0.125 | 0.287 |
| p02 God of War Ragnarök | 0.129 | 0.215 |
| p03 Marvel's Spider-Man 2 | 0.192 | 0.303 |
| **Mean** | **0.169** | **0.387** |

Mean 0.169 clearly beats variant 1's neighbour-median mean of 0.229 on the same practice cases.

Run it locally:

```
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Two runs on the same case write byte-identical output (~0.17 s per case).
