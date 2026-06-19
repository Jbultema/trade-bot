# ML and Bayesian Research Framework

This project should use classical ML as targeted evidence machinery, not as an unconstrained trade generator. The operating system remains long-only, human-reviewed, and benchmarked against simple references. ML models must compete against the current rules in the same backtest, walk-forward, regime, operability, and paper-monitoring framework.

## First-Class Modeling Seams

1. **Future-state probabilities**
   - Target: probability of `risk_off`, `transition`, `risk_on_fragile`, and `risk_on` over 1-week, 1-month, and 3-month horizons.
   - Use: resizes risk exposure and defensive allocation; does not predict exact prices.
   - Current models: base rate, transition tables, analogs, centroid, naive Bayes, ridge-logit, tail-specialist, ensemble, and Bayesian posterior variants.

2. **Re-risking and dip repair**
   - Target: whether a drawdown setup repairs over the next 1-3 months or continues as a falling-knife regime.
   - Use: controls metered re-entry after defensive signals.
   - Candidate methods: regularized classifiers, Bayesian transition priors, survival-style hazard scoring, and calibrated repair probabilities.

3. **Left-tail off-ramp**
   - Target: elevated probability of future drawdown or stress regime.
   - Use: caps risk budget, AI beta, and sector concentration before the full drawdown appears.
   - Candidate methods: asymmetric classifiers, tail-specialists, anomaly/change-point features, and Bayesian model averaging.

4. **Feature selection and signal inclusion**
   - Target: which signal families add stable out-of-sample value by horizon and regime.
   - Use: governs which macro, market, credit, breadth, positioning, valuation, and news-derived features are allowed into state models.
   - Candidate methods: stability selection, elastic-net paths, permutation importance, mutual information, and family-level Bayesian shrinkage.

5. **Strategy-family routing**
   - Target: which family is favored in the current state: AI beta, broad momentum, sector rotation, defensive credit/rates, dip re-entry, or low-churn balanced.
   - Use: champion/challenger monitoring and candidate operating-system construction.
   - Guardrail: router output changes candidate confidence and sizing bands; it does not silently replace portfolio constraints.

6. **News/event impact scoring**
   - Target: whether an event should be ignored, watched, sized down, sector-tagged, or treated as a broad risk catalyst.
   - Use: event-risk pressure and explanation layer.
   - Candidate methods: lightweight supervised models on event metadata, source/category/phase tags, sector tags, recency, urgency, and simple text features.

7. **Transaction and churn filter**
   - Target: whether a proposed trade delta is likely durable enough to execute.
   - Use: suppresses tiny or reversible daily moves.
   - Candidate methods: persistence classifiers and Bayesian posterior confidence thresholds.

8. **Monitoring drift**
   - Target: whether a paper/live strategy is behaving outside its tested distribution.
   - Use: strategy-under-review alerts before scaling capital.
   - Candidate methods: feature drift tests, calibration drift, return-distribution drift, and allocation-behavior drift.

## Bayesian Baseline

Bayesian models enter the system as posterior probability estimators. The first implemented family uses:

- Dirichlet-smoothed class priors.
- Recency-weighted class evidence.
- Dirichlet-smoothed state-transition tables.
- Bayesian Gaussian naive Bayes with mean and variance shrinkage.
- Bayesian ensembles that blend priors, transition evidence, feature likelihoods, and tail-specialist probabilities.

These are intentionally cheap: they use pandas/numpy, run inside the existing batch backtests, and produce the same state-probability vector as other future-state models.

## Cadence

- **Daily inference:** use the latest cached prices and current signal state. No hyperparameter search.
- **Weekly batch refresh:** rerun selected research iterations, update scorecards, and refresh paper-monitoring candidates.
- **Monthly model review:** evaluate calibration, feature stability, live/paper drift, and whether any ML overlays should be promoted or demoted.
- **Quarterly retuning:** add or adjust model families only after reviewing walk-forward, regime holdout, and monitoring evidence.

## Validation Gates

Every ML candidate needs:

- A defined target label and horizon.
- Lag-safe features and no future leakage.
- Walk-forward and regime split evaluation.
- Probability calibration checks when outputs are probabilities.
- Economic utility tests after sizing, churn, execution cost, and risk constraints.
- Direct comparison against the existing heuristic or no-ML control.
- Dashboard explanation of what changed, why it helped, and where it failed.

No ML model should be promoted because it has a good in-sample score, attractive full-history CAGR, or a plausible narrative. It must improve the operating system under the same evidence standards as the non-ML strategies.
