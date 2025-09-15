StockPulse — unsupervised anomaly flags for daily stocks (AAPL 1d)

What it does : Finds “WIERD” trading days using three simple detectors — Isolation Forest (distributional odd days), STL→S-H-ESD (spikes), and PELT (regime changes) — and fuses them, with a clean CSV for Tableau.

Canonical anomaly score: weirdness = -score_samples (so bigger = more anomalous), then z-normalize using train mean/std. In sklearn: decision_function = score_samples − offset_


Spike detection: STL residuals with robust=True (down-weights outliers while fitting), then Generalized ESD for up to k outliers. 

Change-points: PELT with tuned pen, and guardrails min_size + jump for speed and clean segments so like things wount be too slow and we make like clean segments.

What i used to run the code cus like i couldnt run it normally(paste each in order in git bas):
python src/data/fetch_ohlcv.py
python src/features/build_features.py
python src/data/split_time_series.py
python src/models/iof_pipeline.py
python src/detectors/STL_residuals.py
python src/detectors/SHESD.py
python src/detectors/cp_penalty_selection.py
python src/detectors/change_points.py
python src/decisions/fusion.py
python src/decisions/make_tableau_signals.py
python src/labels/build_earnings_labels.py
python src/metrics/compute_metrics_from_labels.py

below is basically how the project plan goes:

M2 — Features & IOF (primary detector)

Features: close_daily_return, rolling mean/std of returns (20), close_zscore20.

Split: chronological 60/20/20 (train/val/test).

IsolationForest: n_estimators=600, max_samples='auto', random_state=42.

Score: weirdness = -score_samples → z-normalize using train stats.

Cutoff (validation): sweep percentiles; pick the numeric cutoff that hits target recall (and, if possible, meets a min precision) using labels_event_windows.csv.

M3 — Spikes (robust STL → S-H-ESD)

Decompose Close with STL (robust=True, period=5), keep residuals.

Run Generalized ESD on residuals; alpha=0.05, max_outliers = 5%, window = 260 to avoid masking on long series.

M4 — Change-points & Fusion

PELT on returns; sweep pen, pick point of diminishing returns before CPs explode; set min_size & jump for stability/speed.

Fusion rule (K = ±1 day near a CP):

Tier A (AUTO-critical): (S-H-ESD spike near CP) or very-strong IOF (≥ p99.5 on val).

Tier B (AUTO-high): IOF flag and (spike or near CP).

Tier C (REVIEW): isolated spike far from CP or weak IOF only.

M5 — Tableau & Metrics (grading only; training stays unsupervised)

Write tableau/signals.csv (schema below) for the main visualisation.

Metrics outputs:

tableau/metrics_summary.csv — test precision/recall at the chosen cutoff.

tableau/pr_points.csv — points for Precision-Recall curve; we also report Average Precision (PR-AUC).

tableau/roc_points.csv — ROC points; we also report ROC-AUC.

LINKS:

Blog post: https://stockpulse-pandas.blogspot.com/2025/09/stockpulse-unsupervised-anomaly-flags.html

Tableau link: https://public.tableau.com/views/Book1_17579167957070/WeirdnessIOFcutoff?:language=en-

Git-hub: https://github.com/Kunmi-Player1/StockPulse


How the evaluation process goes:

Cutoff selection (validation): choose a numeric cutoff that meets target recall; if multiples qualify, pick higher precision.

Report on test: precision & recall at that cutoff, PR-AUC (Average Precision) and ROC-AUC for a threshold-free view. 
Why PR too? For imbalanced anomalies, PR curves and AP tell the story better than accuracy or even ROC alone. 

Assumptions (short)

Date range: 2018-01-01 → 2025-01-01.

Interval: daily (1d).

Training: fully unsupervised; labels only used for validation cutoff & grading.

Limits: offline CSVs, no live trading/execution.

Data:

Source: Yahoo Finance AAPL (1d CSV) — for demo/research; follow Yahoo TOS.

Counts  features.csv: around 1740 rows; train/val/test.csv present & sorted.