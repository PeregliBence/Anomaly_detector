# Weekly Thesis Checklist

Use this every week after running the benchmark scripts.

## 1) Run Information
- Week number:
- Date:
- Run command used: run_all_thesis_workflows.bat / custom
- Notes about environment (package updates, machine changes):

## 2) Artifacts Check
- Baseline run folder exists: results/thesis_runs/YYYY-MM-DD
- ML run folder exists: results/thesis_ml_runs/YYYY-MM-DD
- LSTM run folder exists: results/thesis_lstm_runs/YYYY-MM-DD
- Comparison reports exist:
  - results/comparisons/YYYY-MM-DD/baseline_vs_ml/comparison_report.md
  - results/comparisons/YYYY-MM-DD/baseline_vs_lstm/comparison_report.md

## 3) Key Metrics Snapshot
Fill from the generated forecast_benchmark.csv files.

- Baseline best directional_accuracy:
- Baseline best sharpe_ratio:
- Baseline best max_drawdown_pct:
- Baseline best historical_paper_return_pct:

- ML best directional_accuracy:
- ML best sharpe_ratio:
- ML best max_drawdown_pct:
- ML best historical_paper_return_pct:

- LSTM best directional_accuracy:
- LSTM best sharpe_ratio:
- LSTM best max_drawdown_pct:
- LSTM best historical_paper_return_pct:

## 4) Decision Log (One Change Rule)
- What single change was tested this week?
- Why this change?
- Expected effect:
- Observed effect:
- Keep or revert next week?

## 5) Risks / Issues
- Missing dependencies or runtime issues:
- Data issues (ticker anomalies, missing data):
- Evaluation caveats:

## 6) Thesis Writing Progress
- Figures/tables updated this week:
- Which section can be written now? (methods/results/discussion)
- Open questions for supervisor:

## 7) Next Week Plan
- One change to test:
- Primary success metric:
- Fallback plan if metric worsens:
