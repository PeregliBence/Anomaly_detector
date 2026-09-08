# Model Selection Policy

This policy defines how the final thesis model is selected.

## Goal
Choose a model stack that is stable, reproducible, and acceptable in runtime.

## Candidate Lanes
- Baseline lane: thesis_profile.json
- ML lane: thesis_profile_ml.json
- LSTM lane: thesis_profile_lstm.json

## Mandatory Constraints
A candidate is considered valid only if all are true:
- Directional accuracy is at least 0.50.
- Sharpe ratio is at least 0.20.
- Max drawdown absolute value is at most 35%.
- Trade count is at least 15.
- Historical paper return is strictly better than benchmark_always_long for the same ticker.
- Runtime is acceptable for weekly use on available hardware.

These constraints are now enforced automatically by `thesis_runner.py` and written to:
- `selected_models.csv`
- `deployment_decisions.md`

## Ranking Criteria
Among valid candidates, rank by:
1. Higher directional accuracy
2. Higher Sharpe ratio
3. Lower absolute max drawdown
4. Higher historical_paper_return_pct
5. Lower runtime

## Tie-Break Rule
If two candidates are close (difference in directional accuracy < 1 percentage point),
prefer the simpler candidate with lower runtime and fewer dependencies.

## Rejection Rules
Reject a candidate if:
- Results improve only on one ticker but degrade materially on others.
- Improvement depends on unstable or hard-to-reproduce environment conditions.
- Weekly reruns produce highly inconsistent outcomes without explanation.

## Weekly Decision Protocol
- Run baseline, ML, and LSTM lanes on the same date window.
- Compare using compare_forecast_runs.py outputs.
- Record decision in WEEKLY_CHECKLIST.md.
- Apply only one model/feature change per week.

## Final Thesis Freeze
By the final phase:
- Freeze chosen profile and parameters.
- Freeze dependency versions.
- Produce one final run and one final comparison report.
