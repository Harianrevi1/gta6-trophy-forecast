# GTA VI Trophy Rarity Forecaster

[![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Harianrevi1/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Harianrevi1/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) on make.fun: a program that reads a case of Exophase trophy-page reads and forecasts, for each of the game's 55 PlayStation 5 trophies, five quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate at the case's target, scored by mean pinball loss on the logit scale.

## The idea: forecast counts, not rates

A trophy's earn rate is a ratio, and its two sides move for different reasons:

```
rate = earners / players_tracked
```

Exophase refreshes a game's page only every few hours. Between refreshes the page is frozen: `players_tracked` stays flat while earners creep up as tracked players earn trophies. At a refresh, `players_tracked` jumps (new profiles sync) and every rate drops, because the newcomers hold far fewer trophies than the players tracked so far. On launch night the page sat unchanged at 638 tracked players for hours, then refreshed to 3,868 tracked while total trophies earned went 943 → 4,276 — so every rate but the one at 100% collapsed to about a fifth of its launch value. A forecaster that only extends the rate curve misses that entirely.

So `main.py`, per trophy:

1. Collapses the history to its **distinct states** (consecutive identical reads carry no new information).
2. Measures, over every jump between distinct states, the **newcomers' earn fraction**: new earners of the trophy per newly tracked player.
3. Forecasts `players_tracked` at the target with a log-log trend (log N against log(1 + hours since launch day)) fitted on the distinct states, and forecasts earners as last earners + newcomers' earn fraction × new players + residual hourly earn growth from existing players.
4. Takes the median rate as forecast earners / forecast players.

When the history holds only one distinct state — a page that has not refreshed at all, as on launch night — there is no jump to learn from, so the program falls back to the documented first-refresh pattern: one refresh is due, tracked players multiply about sixfold, and newcomers earn each trophy at only a small fraction of the rate the tracked players show. The 100% trophy is left alone: it survived the first refresh at 100%.

The spread is honest about what is unknown: a base logit spread that widens with the horizon, plus the scatter of the rate's own trend fit, plus a wider scenario spread in the single-state case, where the page may or may not have refreshed by the target. Quantiles are the median shifted on the logit scale by the standard normal's quantiles times that spread.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic: the same case always yields the same bytes. Runtime is ~0.05 s for the two practice cases; outputs are ~2.8 KB per case (limit 65,536).

## Backtest

The kit ships no practice answers, so the practice backtest uses simulated answers calibrated from the competition's own documented refresh aggregates (04:30 read: 3,868 tracked, 4,276 earned). Two variants: a uniform shrink of every non-100% rate to 22.08% of its launch value, and the same aggregate with per-trophy lognormal shrink variation. The starter (house persistence baseline) scores 1.40 / 1.57 on those two variants — close to the house's stated ~1.51, which calibrates the simulation. This forecaster scores **0.17 / 0.29**.

The final regime was tested separately on synthetic week-long cases built from a different data-generating process than the model's own assumptions (per-trophy logistic adoption in log-time, power-law growth of tracked players, page refreshing every few hours): across 7 parameter settings the forecaster scored **0.13–0.15** against **0.22–0.24** for both house baselines (persistence and the fitted-curve baseline) — roughly 0.6× their loss in every setting.

## Files

- `main.py` — the forecaster (the whole entry)
- `README.md` — this file
- `LICENSE` — MIT
