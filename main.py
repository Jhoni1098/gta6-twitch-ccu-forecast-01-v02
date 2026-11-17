"""gta6-twitch-ccu-forecast-01, variant 2: launch-shape regression forecast.

Model family: a parametric launch-day shape model, fitted fresh on every
case from that case's own comparables.  For each past launch we take its
release-day hourly viewers, divide by its hour-1 print, and work in logs:
r_h = ln(viewers_h / viewers_1).  The launches that shape like a rolling
midnight console launch -- already live at hour 1, peaking in the evening --
form the training family (see `in_shape_family`).  For each hour h = 2..24
we fit one straight line through that family's log-ratios against the
log of the launch's hour-1 print:

    r_h  ~  a_h + b_h * ln(p1)

Big launches start high and stay flat; small launches start low and climb
several-fold into the US evening, so b_h is near zero in the small hours
and about -0.45 at the evening peak.  The target launch's median is that
line evaluated at its own (clipped) ln(p1), and its spread is the
cross-launch residual scatter around the line at that hour, scaled down
(the practice launches sit closer to the line than the average comparable
does, so the raw residual spread over-covers).  Quantiles are the normal
quantiles of that lognormal predictive distribution, anchored on hour 1:

    q_tau(h) = p1 * exp(mu_h + z_tau * s_h)

with z the standard normal quantiles at 0.05 / 0.25 / 0.5 / 0.75 / 0.95.

Why this family: a launch day is an unlock wave (New Zealand first, then
Europe, the UK, the US east coast, the US west coast) riding on the
ordinary diurnal rhythm, so the whole day's curve is one smooth shape
whose amplitude is set by how much of the world is already awake at
hour 1 -- which is exactly what the hour-1 print measures.  A
nearest-neighbour median (variant 1) can only repeat past curves; this
regression interpolates the shape continuously with launch size, which is
what a record-size launch like GTA VI needs, since no past launch is as
big as it will be.

Python 3.13, standard library only, deterministic: no randomness, no
network, inputs read in name order, outputs written in hour order.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Hours of the release day we must forecast.
FORECAST_HOURS = list(range(2, 25))

# The fitted slope is steepened a little: the practice launches sit a touch
# above the regression line at the small end and a touch below it at the
# big end, and the target launch is bigger than every training launch.
SLOPE_BOOST = 1.1
# The cross-launch residual scatter overstates the target's own scatter
# (practice launches hug the line more closely than the average comparable),
# so the spread is scaled down.  Calibrated on the four practice cases.
SPREAD_SCALE = 0.7
SPREAD_MIN = 0.25
SPREAD_MAX = 0.85
# Minimum training launches for the regression to be trusted at all.
MIN_FAMILY = 5
# Minimum points at one hour for that hour's line to be fitted; fewer and
# the hour falls back to the family's mean log-ratio at that hour.
MIN_POINTS_PER_HOUR = 5


def parse_ts(text):
    """An ISO-8601 UTC timestamp like 2026-09-15T01:00:00Z, as a datetime."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def round2(x):
    """x rounded to 2 decimals, half up (the way the house baseline rounds).

    A whole number comes back as an int, so json.dumps writes 1240 where a
    float would write 1240.0.
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def release_day(comp):
    """A comparable's release-day hourly viewers, hours 1..24, None if absent.

    The comparable's `hourly` points cover 14 days before the release day
    to 3 days after it; only the 24 hours starting at `day_start` count.
    """
    day_start = parse_ts(comp["day_start"])
    by_hour = {}
    for point in comp.get("hourly") or []:
        when = parse_ts(point["from"])
        hour = int((when - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24 and point.get("viewers") is not None:
            by_hour[hour] = point["viewers"]
    return [by_hour.get(h) for h in range(1, 25)]


def tier_of(comp):
    """The research tier written into the comparable's unlock note.

    Returns 'comparable', 'early-access-contaminated', 'weak' or
    'target-history'; anything unrecognised counts as 'comparable' and is
    then judged on its shape alone.
    """
    note = comp.get("unlock") or ""
    for tier in ("early-access-contaminated", "target-history", "weak"):
        if "Tier: " + tier in note:
            return tier
    return "comparable"


def in_shape_family(comp):
    """True when a comparable's release day shapes like a rolling launch.

    A rolling midnight launch is already live at hour 1 (New Zealand and
    Australia unlocked hours earlier) and keeps climbing into the US
    evening, so its hour-1 print is a real share of its own peak and its
    peak lands late in the day.  Launches that unlocked all at once during
    the day instead start near zero at hour 1 and explode ten-fold or
    more; launches whose main unlock happened the evening before sit flat
    or fall all day.  Neither shape belongs in this family, and neither
    does an early-access launch (its release day is already saturated) or
    the target game's own pre-release history.
    """
    if tier_of(comp) != "comparable":
        return False
    curve = release_day(comp)
    first = curve[0]
    if first is None or first <= 0:
        return False
    present = [v for v in curve if v is not None]
    if len(present) < 20:
        return False
    peak = max(present)
    peak_hour = 1 + max(range(24), key=lambda i: curve[i] or 0)
    return first >= 0.05 * peak and peak_hour >= 12


def fit_hour(points, x_target):
    """One hour's line through (ln p1, log-ratio) points, read at x_target.

    `points` is a list of (x, y) with x = ln(hour-1 print) and y = the
    log-ratio ln(viewers_h / viewers_1) of one training launch.  Returns
    (mu, spread): the predicted log-ratio at x_target and the predictive
    log spread for that hour.  With too few points the line is replaced by
    the plain mean log-ratio, so the hour never fails outright.
    """
    n = len(points)
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mean_x) ** 2 for p in points)
    sxy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    if n < MIN_POINTS_PER_HOUR or sxx <= 1e-9:
        slope = 0.0
    else:
        slope = SLOPE_BOOST * sxy / sxx
    intercept = mean_y - slope * mean_x
    mu = intercept + slope * x_target
    residuals = [p[1] - (intercept + slope * p[0]) for p in points]
    dof = max(n - 2, 1)
    raw_spread = math.sqrt(sum(r * r for r in residuals) / dof)
    spread = min(max(raw_spread, SPREAD_MIN), SPREAD_MAX) * SPREAD_SCALE
    return mu, spread


def persistence_quantiles(p1, hour):
    """The house baseline's row for one hour: flat at p1, widening spread.

    Used only as a fallback when a case has too few training launches to
    fit the regression at all, so the output stays valid either way.
    """
    return [
        round2(p1 * math.exp(z * 0.15 * math.sqrt(hour - 1))) for z in Z
    ]


def forecast_case(case):
    """Forecast one case: {hours: [{hour, q: [q05..q95]}]} for hours 2..24."""
    if case.get("levels") != LEVELS:
        raise ValueError("unexpected quantile levels")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1, first["viewers"])

    # The training family: this case's own comparables, filtered to the
    # rolling-midnight shape.  Each launch contributes its log hour-1
    # print and its 23 log-ratios to hour 1.
    training = []
    for comp in case.get("comparables") or []:
        if not in_shape_family(comp):
            continue
        curve = release_day(comp)
        v1 = curve[0]
        ratios = []
        for hour in FORECAST_HOURS:
            viewers = curve[hour - 1]
            ratios.append(
                math.log(viewers / v1) if viewers and viewers > 0 else None
            )
        training.append((math.log(v1), ratios))

    if len(training) < MIN_FAMILY:
        return {
            "hours": [
                {"hour": h, "q": persistence_quantiles(p1, h)}
                for h in FORECAST_HOURS
            ]
        }

    # Read the line at the target's own log hour-1 print, clipped to the
    # training range: the target launch is bigger than every past launch,
    # and extrapolating the slope past the largest observed launch would
    # flatten the day more than any evidence supports.
    xs = [t[0] for t in training]
    x_target = min(max(math.log(p1), min(xs)), max(xs))

    rows = []
    for index, hour in enumerate(FORECAST_HOURS):
        points = [(x, r[index]) for x, r in training if r[index] is not None]
        mu, spread = fit_hour(points, x_target)
        quantiles = []
        for z in Z:
            value = p1 * math.exp(mu + z * spread)
            value = min(max(value, 1.0), 1e9)
            quantiles.append(round2(value))
        # Rounding can in principle tie or (never) invert neighbours; keep
        # the row non-decreasing so the output is always valid.
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main():
    """Read every case in the input directory, write one forecast per case."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
