# Knowa — Explainable Business Decision Intelligence Platform

**Knowa** is an enterprise-grade, closed-loop business decision intelligence system that connects **prediction**, **explainability**, **actionable counterfactual recommendations**, **constraint-aware budget optimization**, **empirical evaluation**, and **feedback learning** into one end-to-end workflow.

---

## 🎯 Architecture & Upgraded Capabilities

### 1. Actionable Counterfactual Explanations
Conventional explainability (SHAP/LIME) explains *why* a model made a prediction, but does not provide actionable recourse. Knowa extends this by computing feasible counterfactuals:
- **Controllable vs Immutable Features**: Preserves immutable attributes (customer age, tenure, demographic identifiers) while perturbing controllable levers (contract types, support tiers, service add-ons).
- **Feasibility & Bounds Enforcement**: Bounds perturbations strictly within valid business ranges.
- **Sparsity & Cost Minimization**: Finds minimal-effort interventions to shift risk across decision thresholds.
- **Explicit Causal Disclaimer**: Explicitly labels recommendations as statistical model associations rather than proven causal guarantees, preventing overconfidence.

### 2. Constraint-Aware Business Decision Optimizer (Decision Studio)
A dedicated decision optimization engine that determines optimal action allocations subject to real-world operational constraints:
- **Configurable Constraints**: Enterprise budget limits (e.g. ₹50,000), team capacity limits (e.g. max 25 accounts), and unit retention valuations.
- **Multi-Strategy Comparison**:
  1. **ILP Optimal Knapsack**: Solved via mathematical integer linear programming (`scipy.optimize.linprog` with the HiGHS solver) to maximize net business benefit.
  2. **Transparent Greedy ROI**: Ranks interventions by benefit-to-cost ratio.
  3. **Risk-First Prioritization**: Allocates budget to the highest initial churn risk cases.
  4. **No-Action Baseline**: Zero spend, zero risk reduction reference.
- **Constraint Attribution**: Explains *why* actions were selected and which constraints bound the solution (e.g., budget exhaustion vs capacity saturation).
- **Direct Ledger Commitment**: One-click commitment of prioritized actions to the durable Decision Ledger with automated 30-day scheduled rechecks.

### 3. Model Trust & Evaluation Lab
A rigorous empirical benchmarking suite evaluated on held-out test data from a real-world enterprise dataset:
- **Real-World Benchmark Dataset**: IBM Telco Customer Churn (7,043 samples, 21 columns). Preprocessors and encoders fit strictly on training splits to prevent data leakage.
- **Multi-Model Benchmark**:
  - *Simple Prior Heuristic Baseline* (Accuracy: 73.5%, AUC-ROC: 0.500, Brier: 0.195)
  - *Logistic Regression (L2 Regularized)* (Accuracy: 79.9%, AUC-ROC: 0.850, PR-AUC: 0.642, Brier: 0.136, ECE: 0.031)
  - *Knowa Stacked Ensemble (XGBoost + LightGBM + Random Forest)* (Accuracy: 80.3%, AUC-ROC: 0.852, PR-AUC: 0.642, Brier: 0.134, ECE: 0.012)
- **Probability Calibration & Reliability**: Brier Score and Expected Calibration Error (ECE) with 10-bin calibration curves comparing predicted confidence vs empirical positive rates.
- **Conformal Prediction Uncertainty**: Calibrated conformal prediction intervals guaranteeing **91.3% empirical coverage** against a 90% confidence target.
- **Operational Latency**: Sub-20ms inference latency (p50: 18.6ms, p95: 40.5ms).

### 4. Drift Monitoring & Safe Model Updates
Production safety engine protecting model health and governing model updates:
- **Statistical Drift Detection**:
  - Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) two-sample tests for numerical features.
  - Total Variation Distance and new category emergence for categorical features.
  - Missingness spike anomaly detection (alerts on sudden null-rate increases).
- **Evidence-Based Retraining Triggers**: Triggers retraining recommendations based on statistical evidence rather than arbitrary sample counters.
- **Isolated Candidate Training**: Trains candidate models in an isolated staging namespace without touching the active production model.
- **Gated Acceptance Promotion**: Enforces rigorous acceptance criteria (AUC-ROC and accuracy degradation thresholds) before candidate models can be promoted.
- **Atomic Rollback**: One-click atomic rollback to the previous active production model version with full version history and audit trails.

### 5. End-to-End Decision Feedback Loop & Traceability
Complete auditability for every recommended and committed action:
- **Lineage Traceability**: Maps `decision_id` ↔ `prediction_id` ↔ `model_version` ↔ `dataset_version` ↔ `features` ↔ `action_code` ↔ `actual_outcome` ↔ `lift_discrepancy`.
- **Outcome Attribution & Empirical Lift**: Analyzes whether observed real-world outcomes match model predictions and flags high-performing vs underperforming actions.
- **Causal Attribution Disclaimer**: Distinguishes observational correlation from true counterfactual causality.
- **Verified Feedback Dataset Export**: Exports validated ground-truth outcome records for subsequent model retraining.

### 6. Phase 1: Semantic Data Understanding
- Deterministic profiling, semantic type inference, business-role detection (revenue, profit, churn, date, entity).
- Automated target candidate detection, data hygiene scoring, and human-in-the-loop overrides.

### 7. Phase 2: Multi-Dataset Intelligence
- Dataset grain detection, entity and foreign key discovery across datasets.
- Relationship discovery (1:1, 1:N, N:M), fan-out ratio estimation, join simulation, and double-counting safeguards.

### 8. Phase 3: Dynamic Business Intelligence
- **Analysis Opportunity Engine**: Derives meaningful business analyses dynamically from semantic roles and grain.
- **Grain-Aware Analytics**: Safely executes descriptive, trend, comparative, correlation, anomaly, segmentation, and cross-dataset analyses with pre-aggregation safeguards.
- **Non-Causal Association Intelligence**: Formulates bivariate correlations with p-values while strictly enforcing non-causal boundaries.
- **Statistically Defensible Anomaly Detection**: Uncovers extreme outliers (Z > 2.5) with baseline context, magnitude, and affected segments.
- **Proactive Insight Generation**: Deduplicates redundant observations and ranks high-value business insights by severity (`critical`, `warning`, `positive`, `info`), magnitude, and confidence.
- **Complete Explainability Trace**: Mathematical formula, data source, relevant dimensions, and limitations available for every insight.

### 9. Modern Frontend Integration
- **Decision Studio (`/decision-studio`)**: Interactive budget sliders, capacity limits, strategy comparison cards, constraint attribution panel, and ledger commitment.
- **Evaluation Lab (`/evaluation-lab`)**: Model comparison matrix, Recharts 10-bin calibration curves, conformal coverage monitors, and educational metric glossary.
- **Counterfactual Card**: Embedded in Case reviews (`/cases`), showing original vs proposed values, controllable feature adjustments, cost tiers, and causal caveats.
- **Upgraded Monitoring (`/monitoring`)**: Feature drift statistics table, health scorecards, candidate training controls, acceptance verification cards, and version rollback buttons.
- **Data & Intelligence Views**: Dynamic business intelligence overview, multi-dataset relationship modeling, and semantic entity mapping.

---

## 🛠️ API Reference

### Optimization
- `POST /api/optimizer/demo-optimize` — Run multi-strategy optimization on synthetic demo data
- `POST /api/optimizer/projects/{id}/optimize` — Run optimization on project prediction pool
- `POST /api/optimizer/projects/{id}/optimize/commit` — Commit optimal strategy to Decision Ledger

### Evaluation Lab
- `GET /api/evaluation/latest` — Fetch latest benchmark evaluation report
- `GET /api/evaluation/history` — List historical benchmark runs
- `POST /api/evaluation/run` — Execute real-world benchmark on held-out test split
- `POST /api/evaluation/projects/{id}/evaluate` — Evaluate specific project models

### Drift & Lifecycle Monitoring
- `GET /api/monitoring/projects/{id}/drift` — Compute PSI, KS-test, and missingness drift report
- `POST /api/monitoring/projects/{id}/candidate/train` — Train candidate model in isolated staging
- `GET /api/monitoring/projects/{id}/candidate/{candidate_id}/evaluate` — Test candidate against acceptance criteria
- `POST /api/monitoring/projects/{id}/candidate/{candidate_id}/promote` — Promote candidate to production
- `POST /api/monitoring/projects/{id}/rollback` — Roll back active model to previous version
- `GET /api/monitoring/projects/{id}/history` — Retrieve model version history audit trail

### Counterfactuals
- `GET /api/explain/{prediction_id}/counterfactual` — Generate counterfactual for prediction
- `POST /api/projects/{id}/counterfactual` — Compute feasible counterfactual scenario for feature set

### Feedback & Lineage Analytics
- `GET /api/feedback-analytics/decisions/{decision_id}/lineage` — Full audit trail and discrepancy analysis
- `GET /api/feedback-analytics/effectiveness` — Aggregated empirical action performance report
- `GET /api/feedback-analytics/projects/{id}/export-dataset` — Export verified feedback dataset for retraining

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt --prefer-binary

# Start backend server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation available at: `http://127.0.0.1:8000/docs`

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Web Application available at: `http://localhost:3000`

### Running the Automated Test Suite
```bash
cd backend
./venv/bin/pytest
```
The test suite validates:
- Counterfactual explainer feasibility & immutable invariant preservation (`test_counterfactual_explainer.py`)
- Highs ILP Knapsack & Greedy optimizer budget constraints (`test_decision_optimizer.py`)
- Model Trust & Evaluation Lab metric bounds and zero-leakage splits (`test_evaluation_lab.py`)
- PSI, KS tests, missingness spike alerts, candidate gating, and rollback (`test_drift_monitoring.py`)
- Decision audit lineage, empirical lift attribution, and feedback export (`test_feedback_analytics.py`)
- Full baseline platform tests (`test_blindspots.py`, `test_dataset_profiler.py`, `test_decision_portfolio.py`, etc.)

---

## 📊 Benchmark Dataset Details
- **Dataset**: IBM Telco Customer Churn
- **Source**: IBM Cognos Community / Kaggle Open Data
- **Location**: `backend/data/benchmarks/telco_churn.csv`
- **Documentation**: `backend/data/benchmarks/README.md`
- **License**: CC0 Public Domain
