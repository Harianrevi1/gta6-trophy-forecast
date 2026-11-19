"""GTA VI trophy rarity forecaster for make.fun (gta6-trophy-rarity-forecast-01).

The program reads one case file per run (schema "makefun/gta6-trophies-case/v1")
from $MAKEFUN_INPUT_DIR (else /input) and writes /output/<id>.out per case:
five quantiles of every trophy's earn rate at the case's target.

The model works in counts, not in raw rates, because a trophy's rate is a
ratio and the two sides of the ratio move for different reasons:

    rate_i(t) = earners_i(t) / players_tracked(t)

Exophase only refreshes a game's page every few hours. Between refreshes the
page is frozen: players_tracked is flat while earners creep up as the tracked
players earn trophies. At a refresh, players_tracked jumps (new profiles
sync) and every rate drops, because the newcomers hold far fewer trophies
than the players tracked so far. On launch night the page sat unchanged at
638 tracked players for hours, then refreshed to 3,868 tracked while total
trophies earned went 943 -> 4,276: the newcomers had earned almost nothing
yet, so every rate but the one at 100% collapsed to about a fifth of its
launch value. A forecaster that only extends the rate curve misses that.

So, per trophy, the program:

  1. collapses the history to its distinct states (consecutive reads with
     identical players_tracked and rates carry no new information);
  2. measures, over every jump between distinct states, the newcomers'
     earn fraction: new earners of the trophy per newly tracked player;
  3. forecasts players_tracked at the target with a log-log trend
     (log N against log(1 + hours since launch day)) fitted on the distinct
     states, and forecasts earners as last earners + newcomers' earn
     fraction x new players + residual hourly earn growth;
  4. takes the median rate as forecast earners / forecast players.

When the history holds only one distinct state (a page that has not
refreshed at all, as on launch night), there is no jump to learn from, so
the program falls back to the documented first-refresh pattern: one
refresh is due, tracked players multiply about sixfold, and newcomers
earn each trophy at only a small fraction of the rate the tracked players
show. The 100% trophy is left alone: it survived the first refresh at
100%, so every tracked player holds it.

The spread is honest about what is unknown: a base logit spread that
widens with the horizon, plus the scatter of the trend fit, plus a wide
scenario spread in the single-state case, where the page may or may not
have refreshed by the target. Quantiles are the median shifted on the
logit scale by the standard normal's quantiles times that spread.

Python 3.13, standard library only, no network, deterministic: the same
case always yields the same bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at those levels.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Launch day begins here; log-time is measured in hours from this instant.
LAUNCH_START = datetime(2026, 11, 19, tzinfo=timezone.utc)

# First-refresh pattern, taken from the competition's own account of launch
# night: the page refreshed from 638 to 3,868 tracked players while total
# trophies earned went 943 -> 4,276. Newcomers earned, per trophy, only
# about 0.067% of the launch rate in absolute terms (103 new earners over
# 54 non-100% trophies from 3,230 newcomers), and the median rate fell to
# about 22% of its launch value. Used only when the history shows a single
# distinct state, i.e. no refresh has been observed yet.
FIRST_REFRESH_TRACKED_MULTIPLE = 3868.0 / 638.0
FIRST_REFRESH_NEWCOMER_FRACTION = 0.000667
FIRST_REFRESH_MEDIAN_SHRINK = 0.2206

# Spread (in logit units) of the forecast quantiles.
BASE_SIGMA = 0.30
HORIZON_SIGMA_PER_SQRT_DAY = 0.25
# Single-state (stale page) spread, tuned on the practice backtest: 0.85
# keeps the 90% band near its honest 0.90 coverage on a refresh whose
# per-trophy shrink varies, while its 95% quantile still reaches most of
# the way back to the stale rate if no refresh lands before the target.
# Wider spreads (1.05+) only cost pinball loss on both simulations.
SINGLE_STATE_SIGMA = 0.85


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_launch(moment):
    """Hours from launch day's start; never negative."""
    return max(0.0, (moment - LAUNCH_START).total_seconds() / 3600.0)


def clamp_pct(value):
    return min(99.99, max(0.01, value))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(value):
    return 100.0 / (1.0 + math.exp(-value))


def round4(value):
    """Round to 4 decimals, halves up, as a plain number in JSON."""
    scaled = math.floor(value * 10000.0 + 0.5) / 10000.0
    return int(scaled) if scaled == int(scaled) else scaled


def distinct_states(case):
    """Collapse the history to reads where something actually changed.

    Returns a list of (moment, players_tracked, rates) with consecutive
    duplicates removed. A read is a duplicate when its players_tracked and
    its whole rates list equal the previous read's.
    """
    states = []
    previous_key = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != previous_key:
            states.append((parse_time(read["at"]), read["players_tracked"],
                           list(read["rates"])))
            previous_key = key
    return states


def last_known_rate(states, index):
    """The trophy's rate in the latest state that lists it."""
    for _, _, rates in reversed(states):
        if rates[index] is not None:
            return rates[index]
    return 0.0


def earners_at(state, index):
    """Earners implied by a state, or None when the rate is missing."""
    _, tracked, rates = state
    if rates[index] is None:
        return None
    return rates[index] / 100.0 * tracked


def fit_line(xs, ys):
    """Least-squares y = a + b*x. Returns (a, b, residual_std)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        return mean_y, 0.0, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = mean_y - b * mean_x
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(r * r for r in resid) / n
    return a, b, math.sqrt(var)


def forecast_trophy(case, states, index, target_moment, horizon_days):
    """Median rate and logit spread for one trophy at the target."""
    last_moment, last_tracked, _ = states[-1]
    last_rate = last_known_rate(states, index)
    last_earners = last_rate / 100.0 * last_tracked
    target_hours = hours_since_launch(target_moment)
    last_hours = hours_since_launch(last_moment)

    if len(states) == 1:
        # No refresh has been observed: the page is stale and one refresh
        # is due before the target. Apply the first-refresh pattern.
        if last_rate >= 99.99:
            median = 99.99
        else:
            new_tracked = last_tracked * FIRST_REFRESH_TRACKED_MULTIPLE
            new_earners = (last_earners + FIRST_REFRESH_NEWCOMER_FRACTION
                           * last_rate * (new_tracked - last_tracked))
            median = clamp_pct(100.0 * new_earners / new_tracked)
        sigma = SINGLE_STATE_SIGMA
        return median, sigma

    # --- Multi-state history: learn the jumps. ---
    # Newcomers' earn fraction for this trophy, over all jumps and over
    # the most recent jump, blended half and half (recent jumps describe
    # the newcomers of the coming days better; all jumps are steadier).
    total_new_earners = 0.0
    total_new_players = 0.0
    recent_fraction = None
    jumps = 0
    for prev, cur in zip(states, states[1:]):
        prev_e = earners_at(prev, index)
        cur_e = earners_at(cur, index)
        dn = cur[1] - prev[1]
        if prev_e is None or cur_e is None or dn <= 0:
            continue
        de = cur_e - prev_e
        total_new_earners += max(0.0, de)
        total_new_players += dn
        recent_fraction = max(0.0, de) / dn
        jumps += 1
    if total_new_players > 0:
        overall_fraction = total_new_earners / total_new_players
    else:
        overall_fraction = None
    if overall_fraction is not None and recent_fraction is not None:
        newcomer_fraction = 0.5 * overall_fraction + 0.5 * recent_fraction
    elif overall_fraction is not None:
        newcomer_fraction = overall_fraction
    elif recent_fraction is not None:
        newcomer_fraction = recent_fraction
    else:
        newcomer_fraction = last_rate / 100.0

    # Residual hourly earn growth not explained by newcomers (existing
    # players earning trophies between refreshes).
    residual_growth = 0.0
    residual_hours = 0.0
    for prev, cur in zip(states, states[1:]):
        prev_e = earners_at(prev, index)
        cur_e = earners_at(cur, index)
        dt = (cur[0] - prev[0]).total_seconds() / 3600.0
        if prev_e is None or cur_e is None or dt <= 0:
            continue
        dn = cur[1] - prev[1]
        explained = newcomer_fraction * max(0, dn)
        residual_growth += max(0.0, (cur_e - prev_e) - explained)
        residual_hours += dt
    hourly_growth = (residual_growth / residual_hours
                     if residual_hours > 0 else 0.0)

    # Players tracked at the target: log-log trend over distinct states.
    xs = [math.log1p(hours_since_launch(m)) for m, _, _ in states]
    ys = [math.log(max(1, n)) for _, n, _ in states]
    _, slope, _ = fit_line(xs, ys)
    slope = max(0.0, slope)
    target_tracked = last_tracked * math.exp(
        slope * (math.log1p(target_hours) - math.log1p(last_hours)))
    target_tracked = min(max(target_tracked, last_tracked),
                         last_tracked * 100.0)

    # Earners at the target, then the rate.
    future_earners = (last_earners
                      + newcomer_fraction * (target_tracked - last_tracked)
                      + hourly_growth * max(0.0, target_hours - last_hours))
    future_earners = max(future_earners, last_earners)
    median = clamp_pct(100.0 * future_earners / target_tracked)

    # Spread: base + horizon widening + scatter of the rate's own trend.
    trend_xs = []
    trend_ys = []
    for moment, _, rates in states:
        if rates[index] is not None:
            trend_xs.append(math.log1p(hours_since_launch(moment)))
            trend_ys.append(logit(rates[index]))
    _, _, trend_resid = fit_line(trend_xs, trend_ys) if len(trend_xs) >= 2 \
        else (0.0, 0.0, 0.0)
    sigma = math.sqrt(BASE_SIGMA ** 2
                      + (HORIZON_SIGMA_PER_SQRT_DAY
                         * math.sqrt(horizon_days)) ** 2
                      + min(trend_resid, 1.0) ** 2)
    return median, sigma


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not the contest's five levels")
    if not case["history"]:
        raise ValueError("case has no history")
    states = distinct_states(case)
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    rows = []
    for index, trophy in enumerate(case["trophies"]):
        median, sigma = forecast_trophy(case, states, index, target,
                                         horizon_days)
        centre = logit(median)
        quantiles = []
        for z in Z:
            quantiles.append(round4(clamp_pct(inv_logit(centre + z * sigma))))
        # Keep the row non-decreasing after rounding and clamping.
        for k in range(1, 5):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
