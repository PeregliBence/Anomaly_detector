# Thesis Usage Guide (Semester Workflow)

This project is prepared for two parallel goals:
1. robust anomaly detection on financial time series
2. directional forecasting + paper-trading style evaluation

## Stable Scope Model (Recommended)

The thesis baseline is split into clear scopes so results stay explainable and reproducible.

### Scope A: Anomaly Core
- Rolling Z-score
- Isolation Forest ensemble
- Ensemble anomaly score + severity

### Scope B: Forecast Core
- XGBoost directional forecast
- ARIMA + XGBoost combined benchmark

### Scope C: Evaluation Core
- Walk-forward validation
- Metrics: directional accuracy, Sharpe, max drawdown, profit factor
- Anomaly metrics: precision, recall, F1

### Scope D: Semester Projection
- Weekly rerun with same profile
- Paper-trading style return projection to semester end

LSTM is treated as experimental extension, not as baseline.

## What The App Can Do

### 1. Data Pipeline
- Download OHLCV market data from Yahoo Finance.
- Build technical and statistical features automatically.
- Validate data quality and report warnings.

### 2. Anomaly Detection
- Run multiple anomaly methods (statistical + ML).
- Merge model outputs into an ensemble anomaly score.
- Classify anomaly severity and summarize anomaly behavior.

### 3. Forecasting
- Evaluate ARIMA, XGBoost, LSTM, and model combinations.
- Provide directional prediction quality over rolling windows.
- Fallback to a deterministic heuristic when optional model dependencies are missing.

### 4. Backtesting And Paper-Trading Proxy
- Walk-forward out-of-sample validation.
- Metrics: directional accuracy, MAE, RMSE, Sharpe, profit factor, max drawdown.
- Semester projection: estimated return until semester end based on historical signal behavior.

### 5. Research Artifacts
Each run creates reproducible artifacts:
- forecast_benchmark.csv
- anomaly_benchmark.csv
- run_details.json
- research_summary.md

These are directly usable as figures/tables in your thesis writing.

## Ready-To-Use Commands

### Daily or Interactive Analysis
- Dashboard:
  - Windows: run_app.bat
  - Manual: streamlit run streamlit_app.py

### Main Research Run
- Default profile run:
  - python thesis_runner.py --profile thesis_profile.json
- ML-track profile run:
  - python thesis_runner.py --profile thesis_profile_ml.json
- One-click Windows:
  - run_thesis_research.bat --profile thesis_profile.json

### Weekly Tracked Run (Recommended)
- run_thesis_weekly.bat
- Output is stored in dated folders under results/thesis_runs/YYYY-MM-DD

### Weekly ML Tracked Run
- run_thesis_ml_weekly.bat
- Output is stored in dated folders under results/thesis_ml_runs/YYYY-MM-DD
- Use this lane for supervisor-facing ML comparisons

### Weekly LSTM Tracked Run (Experimental)
- run_thesis_lstm_weekly.bat
- Output is stored in dated folders under results/thesis_lstm_runs/YYYY-MM-DD
- Use this lane when you specifically analyze LSTM added value vs runtime cost

### Baseline vs ML Auto-Comparison
- run_compare_baseline_ml.bat
- Or run compare_forecast_runs.py manually for custom folders
- Generated comparison report helps produce supervisor-facing summary tables quickly

## Semester Plan (Practical)

### Week 1-2: Baseline Setup
- Run thesis_runner with default profile.
- Verify all output files are generated.
- Freeze ticker universe and model-set list.

### Week 3-6: Iterative Model Improvement
- Run weekly benchmark with run_thesis_weekly.bat.
- Keep one change per week (feature or model change) to preserve interpretability.
- Track directional accuracy and Sharpe as primary forecast KPIs.
- Keep Stable Scope unchanged; put LSTM attempts in separate experimental runs.

### Week 7-10: Comparative Evaluation
- Compare weekly run folders and identify the most stable model set.
- Document where models fail (high volatility periods, drawdown spikes).
- Select final candidate configuration for thesis conclusions.

### Week 11-13: Thesis Material Assembly
- Build tables from forecast_benchmark.csv and anomaly_benchmark.csv.
- Use research_summary.md for narrative anchors.
- Add plots from Streamlit screenshots for visual evidence.

### Week 14: Final Validation
- Run one final benchmark with unchanged settings.
- Include run_details.json in appendix for reproducibility evidence.

## Suggested Thesis Evaluation Structure

### A) Anomaly Section
- Precision, recall, F1 of anomaly signals.
- Post-anomaly return magnitude as economic relevance signal.

### B) Forecast Section
- Directional accuracy by model set and ticker.
- Risk-adjusted quality: Sharpe, max drawdown, profit factor.

### C) Paper-Trading Section
- Historical paper return from walk-forward evaluation.
- Projected return to semester end (clearly marked as estimate, not promise).

### D) Baseline vs ML Comparison
- Compare Stable profile vs ML profile on the same ticker universe and date range.
- Report trade-off: runtime vs directional accuracy vs risk metrics.
- Keep conclusions on consistency, not single-run outperformance.

## Notes On Reproducibility
- Keep the same profile file across the semester.
- Keep the same ticker universe for comparable results.
- Record any dependency changes in a short changelog.
- Avoid mixing experimental one-off runs into final weekly folders.

## Important Disclaimer
This project is for academic research and educational use. Forecast outputs and paper-trading projections are uncertain estimates and not investment advice.
