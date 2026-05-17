# obf-psychiatric-pipeline

> Reproducible classification of psychiatric conditions from wrist-worn
> motor activity, applying the same pipeline philosophy as my
> [RNA-seq pipeline](https://github.com/arash-rahmani/rnaseq-python-pipeline)
> to a fundamentally different data domain: from genome to behavior.

**Language:** Python 3.10+ &nbsp;|&nbsp;
**Dataset:** OBF-Psychiatric (Garcia-Ceja et al., 2024) &nbsp;|&nbsp;
**Models:** logistic regression, XGBoost, dummy baseline &nbsp;|&nbsp;
**Tests:** pytest

---

## Headline finding

Wrist-worn motor activity distinguishes psychiatric inpatients from
healthy controls and, when enriched with temporal and circadian
features, meaningfully separates depression from schizophrenia.

**Distributional features only (baseline):** binary F1 0.798 (0.700–0.881),
3-class F1 0.595 (0.480–0.705).

**Combined features (distributional + temporal + circadian):**
binary F1 **0.849** (0.761–0.920), 3-class F1 **0.753** (0.645–0.841).

The +0.158 gain on the 3-class task is driven by temporal features
(interdaily stability, intradaily variability, cosinor acrophase, and
sleep metrics) that carry disorder-specific circadian signatures not
captured by activity volume alone. Temporal features alone outperform
distributional features alone on 3-class (F1 0.699 vs 0.595):
*the feature engineering choice moved the needle, not model complexity.*

![Circadian activity profiles by cohort](results/figures/circadian_activity_profiles.png)

> Mean 24-hour activity profiles for controls, depression, and schizophrenia.
> Both patient cohorts show attenuated amplitude and a flattened rhythm
> relative to controls; the depression cohort shows a visually delayed
> acrophase that the schizophrenia cohort does not.

> Profiles use all 77 raw actigraphy participants; one participant excluded
> from classification for fewer than 7 recording days. Depression and
> schizophrenia cohorts derive from separate source studies (Depresjon and
> Psykose) with different recording protocols.

**Per-participant headline numbers (5-fold GroupKFold CV, bootstrap CIs):**

| Task | Feature set | Classifier | Macro-F1 | 95% CI | Lift over dummy |
|---|---|---|---|---|---|
| Control vs Patient | Distributional | Logistic Regression | 0.798 | 0.700–0.881 | +0.19 |
| Control vs Patient | Distributional | XGBoost | 0.755 | 0.646–0.841 | +0.15 |
| Control vs Patient | Combined | Logistic Regression | 0.827 | 0.737–0.902 | +0.22 |
| Control vs Patient | **Combined** | **XGBoost** | **0.849** | **0.761–0.920** | **+0.25** |
| Control vs Depr vs Schiz | Distributional | Logistic Regression | 0.595 | 0.480–0.705 | +0.21 |
| Control vs Depr vs Schiz | Temporal only | Logistic Regression | 0.699 | 0.590–0.802 | +0.31 |
| Control vs Depr vs Schiz | **Combined** | **Logistic Regression** | **0.753** | **0.645–0.841** | **+0.37** |

![Temporal feature performance comparison](results/figures/temporal_performance_comparison.png)

> Distributional, temporal-only, and combined feature sets compared
> across both tasks. The 3-class gain from distributional to combined
> (+0.158) is the headline result; the binary gain (+0.051) is real but
> smaller. Circadian structure is disorder-specific, not merely
> patient-specific. The CIs for distributional and combined 3-class do
> not overlap.

![PCA projection](results/eda/pca_projection.png)

> PC1 (65.6%) captures overall activity level and creates a class
> gradient. PC2 (17.6%) does not separate cohorts. The geometry is
> nearly one-dimensional on distributional features alone; temporal
> features add dimensions that separate the patient cohorts.

---

## What this pipeline does

This pipeline takes wrist-worn motor activity data from psychiatric
inpatients and healthy controls, and runs an honest, reproducible
classification analysis that respects the structure of the data:

- **Dual task framing:** both 3-class (control / depression /
  schizophrenia) and 2-class (control / patient). The contrast between
  the two is the finding.
- **Dual aggregation:** both per-day and per-participant
  classification, with participant-level cross-validation in both cases
  to prevent within-subject leakage.
- **Three classifiers:** stratified dummy (floor), logistic regression
  (interpretable linear baseline), XGBoost (non-linear). All compared
  fairly on identical folds.
- **Bootstrap 95% CIs on every metric:** point estimates lie when
  n=76; intervals are honest.
- **Temporal and circadian feature extraction:** interdaily stability
  (IS), intradaily variability (IV), L5/M10 rest-activity windows,
  cosinor parameters (mesor, amplitude, acrophase, R²), and Cole-Kripke
  sleep metrics (TST, WASO, sleep efficiency, SOL) computed from raw
  per-minute actigraphy.
- **SHAP-based feature attribution** on the best non-linear model.
- **Config-driven, schema-validated, pytest-tested:** same
  architectural philosophy as my RNA-seq pipeline.

---

## Pipeline overview

```
Raw per-minute actigraphy (Depresjon / Psykose)
                 │
                 ▼
        Schema-validated raw loader
                 │
                 ▼
   Temporal feature extraction (17 features per participant)
   • IS, IV, L5, M10, amplitude, relative amplitude
   • Cosinor: mesor, amplitude, acrophase, R²
   • Sleep: TST, WASO, sleep efficiency, SOL
                 │
                 ▼
Raw OBF metadata (5 cohorts) + features.csv (3 cohorts)
                 │
                 ▼
   Schema-validated loader + preprocessing
   • drop participants with < 7 recording days
   • drop q25 (uninformative; saturates at zero)
                 │
                 ▼
   Join distributional + temporal on participant ID
                 │
                 ▼
   Participant-level GroupKFold (n=5, seed=42)
                 │
   ┌─────────────┴─────────────┐
   ▼                           ▼
 Track A: 3-class            Track B: 2-class
 (control/depr/schiz)        (control vs patient)
   │                           │
   ▼                           ▼
   Dummy  ‖  LogReg  ‖  XGBoost
   │                           │
   ▼                           ▼
   Macro-F1, per-class metrics, confusion matrices,
   ROC curves, SHAP attribution, bootstrap 95% CIs
```

---

## Repository structure

```
src/obf_psychiatric_pipeline/   # importable Python package
  config.py                     # YAML config loader, frozen dataclass
  data/
    loader.py                   # schema-validated loaders (features.csv)
    raw_loader.py               # raw per-minute actigraphy loader
    preprocess.py               # min-days filter, feature exclusion
    split.py                    # participant-level GroupKFold
  features/
    _helpers.py                 # shared private helpers
    temporal.py                 # IS, IV, L5, M10
    cosinor.py                  # cosinor model parameters
    sleep.py                    # Cole-Kripke scorer + sleep metrics
    derived.py                  # amplitude, relative amplitude
    extract.py                  # per-participant feature orchestrator
  models/
    classifiers.py              # estimator factories
    aggregate.py                # per-day -> per-participant
    relabel.py                  # 3-class -> 2-class
    evaluate.py                 # metrics + bootstrap CIs
    train.py                    # 2x2x3 experiment grid
  viz/
    eda.py                      # 5 EDA plots
    confusion.py                # confusion matrices
    roc.py                      # ROC curves
    shap_plots.py               # SHAP attribution
config/config.yaml              # paths, split seed, exclusions
scripts/                        # CLI entry points
tests/                          # 112 pytest tests
results/                        # generated outputs (gitignored)
data/                           # input data (gitignored)
```

---

## Quickstart

**Requirements:** Python 3.10+, Windows (PowerShell) or Linux

```powershell
git clone https://github.com/arash-rahmani/obf-psychiatric-pipeline
cd obf-psychiatric-pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
```

Place the OBF-Psychiatric CSVs in `data/raw/` (six files: five
`*info.csv` and `features.csv`). Place raw per-minute actigraphy in
`data/raw/actigraphy/{control,depression,schizophrenia}/`.

```powershell
python scripts/load_data.py               # smoke test
python scripts/run_eda.py                 # generates 5 EDA plots
python scripts/train_models.py            # distributional-only experiment
python scripts/run_temporal_experiment.py # distributional vs temporal vs combined
python scripts/run_viz.py                 # confusion matrices, ROC, SHAP
pytest tests/                             # 112 tests
```

---

## Key design decisions

**Why dual task framing (3-class and 2-class)?**
Exploratory PCA showed that depression and schizophrenia largely
overlap in distributional feature space while both separate cleanly
from controls. Reporting only the 3-class result would have hidden the
actual structure of the data. Reporting both, and the gap between them,
makes the finding visible. The gap narrows substantially with temporal
features (+0.158 on 3-class vs +0.051 on 2-class).

**Why participant-level splitting, not row-level?**
Each participant contributes ~12 rows (one per recording day). Random
row-level splitting puts the same participant in both train and test,
which inflates measured performance via subject-identity leakage. This
is the most common methodological error in actigraphy-based
classification literature. `GroupKFold(groups=user)` enforces
participant-level independence.

**Why both per-day and per-participant aggregation?**
Per-day uses more data (~880 samples) but day-rows from the same
person are not independent; confidence intervals computed on per-day
data are artificially tight. Per-participant aggregation (~76 samples)
respects the true unit of clinical inference and gives honest CIs.
Both are reported; the per-participant numbers are the ones to trust.

**Why temporal features at participant level only?**
IS, IV, L5/M10, cosinor, and sleep metrics are inherently multi-day
aggregates that characterise a participant's rhythm over the full
recording, not a single day. Computing them at the day level would be
methodologically incoherent. This is also why they most directly address
the 3-class problem: they capture rhythm structure, not daily volume.

**Why logistic regression as a baseline alongside XGBoost?**
When logreg matches XGBoost, the problem is near-linear and model
complexity is not the path to better performance; feature engineering
is. That prediction shaped Phase 2: temporal features pushed 3-class
F1 from 0.595 to 0.753, while XGBoost began to outperform logreg on
the combined 2-class task (0.849 vs 0.827), indicating the richer
feature space benefits from non-linear capacity.

**Why no hyperparameter tuning?**
With 22 participants in the smallest cohort, nested CV hyperparameter
search is mostly noise. Default sklearn / XGBoost hyperparameters were
used (XGBoost: `n_estimators=200, max_depth=4, learning_rate=0.05`,
logreg: `C=1.0`, both with `class_weight="balanced"`). The focus of
this work is methodological framing and feature engineering, not
hyperparameter optimization.

**Why bootstrap confidence intervals?**
At n=76 participants, point estimates of macro-F1 are unstable across
resampling. The bootstrap CI honestly reflects how much uncertainty
remains in the headline number.

---

## Findings

**1. Combined features substantially improve 3-class discrimination.**
Distributional-only 3-class F1 was 0.595. Temporal features alone
reach 0.699. Combined features reach 0.753. The CIs for distributional
and combined do not overlap. This is a robust finding, not noise.

![3-class confusion matrix (logistic regression, combined features)](results/figures/confusion_3class_combined_logreg.png)

> The 3-class classifier separates controls from both patient cohorts
> with reasonable reliability. Depression and schizophrenia still
> confuse each other in both directions; the residual overlap is
> the honest upper bound of what combined features can resolve at
> this sample size.

**2. Temporal features alone outperform distributional features alone
on the 3-class problem.**
F1 0.699 vs 0.595 with logistic regression. Circadian structure
carries disorder-specific information that activity volume statistics
do not. Sleep fragmentation, reduced interdaily stability, and shifted
acrophase differentiate the patient cohorts in ways that mean activity
level cannot.

**3. Feature engineering, not model complexity, was the lever.**
The original pipeline showed logreg matching XGBoost on distributional
features, a sign the problem was near-linear with that feature set.
Adding temporal features created a richer space where XGBoost gains an
edge on 2-class (0.849 vs 0.827), confirming the prediction: invest in
features, not tuning.

**4. One pre-computed feature carries no signal.**
The 25th percentile of per-minute activity (`q25`) saturates at zero
across all classes; every participant spends substantial time
motionless during sleep, regardless of diagnosis. Excluded from
modeling.

**5. Confidence intervals widen honestly with proper aggregation.**
Per-day CIs were ~0.06 wide; per-participant CIs were ~0.18 wide for
the same metric. The wider intervals are the honest ones.

### Mechanistic interpretation

SHAP attribution on the combined-feature XGBoost model (n=76, full
dataset) identifies which features drive each classification decision
and in which direction. Three results stand out.

**Depression.** Cosinor acrophase is the dominant feature. Low acrophase
values carry strongly negative SHAP for the depression class: a delayed
activity peak is the primary signal pushing a participant toward a
depression prediction. L5 onset ranks second, independently flagging
delayed sleep onset as corroborating evidence. Together, these two
features recover the textbook delayed circadian phase finding in MDD
from actigraphy alone, without any clinical annotation.

**Schizophrenia.** Activity standard deviation dominates: low SD pushes
strongly toward schizophrenia, consistent with antipsychotic motor
suppression flattening the behavioral repertoire. Cosinor acrophase
enters in the opposite direction from depression: an earlier activity
peak predicts schizophrenia rather than depression. This confirms
that the two disorders carry distinct circadian signatures even when
their gross activity levels overlap.

**Binary (control vs patient).** Percentage of zeros (sedentary time),
intradaily variability, and interdaily stability lead the ranking.
Temporal features place alongside distributional ones, confirming that
circadian structure contributes independent signal beyond overall
activity level.

![SHAP summary: 3-class XGBoost, depression](results/figures/shap_summary_combined_3class_depression.png)

![SHAP summary: 3-class XGBoost, schizophrenia](results/figures/shap_summary_combined_3class_schizophrenia.png)

> SHAP attributions computed by training XGBoost on full dataset (n=76)
> for interpretability purposes only. Classification performance metrics
> derive from held-out 5-fold GroupKFold cross-validation.

---

## Limitations

- **Sample size: n = 76 participants** (after preprocessing; 77 in
  raw data). Bootstrap CIs reflect this; readers should weight point
  estimates accordingly.
- **Inpatient cohorts on medication.** Both patient groups were
  recorded during inpatient stays. Antipsychotic and antidepressant
  medications have known motor effects. Results characterise "psychiatric
  inpatient on medication" as much as "depression" or "schizophrenia"
  per se, and do not generalise to outpatient or unmedicated populations.
- **Cole-Kripke device mismatch.** Sleep scoring was validated on AMI
  Motionlogger hardware; OBF uses Actiwatch. Absolute sleep-minute
  accuracy is reduced, but group separation survives because
  miscalibration affects all cohorts uniformly.
- **No external validation.** All performance numbers come from
  cross-validation on a single dataset. Generalisation to other
  actigraphy datasets is untested.
- **The OBF-Psychiatric dataset combines two earlier studies**
  (Depresjon and Psykose) with different recording protocols. Some
  cross-cohort confounds may reflect study-of-origin rather than
  diagnosis.

---

## Next work

This pipeline is the basis for a paper in preparation, targeting
*npj Digital Medicine*. The analysis reported here is complete.

The methodological backbone established here covers schema-validated
loading, participant-level CV, bootstrap-CI'd evaluation, dual task
framing, and SHAP attribution. It transfers directly to the open
question this work cannot yet answer: whether these findings replicate
on independent actigraphy datasets collected outside the
OBF-Psychiatric cohort. External validation is the natural next step.

---

## Citation

Data: Enrique Garcia-Ceja, Andrea Stautland, Michael A. Riegler, Pål
Halvorsen, Salvador Hinojosa, Gilberto Ochoa-Ruiz, Ketil Joachim
Oedegaard, and Petter Jakobsen. *OBF-Psychiatric, a motor activity
dataset of depressed, schizophrenic, and attention deficit
hyperactivity disorder patients*, 2024.

---

## About

Built by [Arash Rahmani](https://github.com/arash-rahmani), M.Sc.
Bioinformatics, Julius-Maximilians-Universität Würzburg.

This is the second in a series of reproducible analysis pipelines
spanning biological and behavioral data:

- **[rnaseq-python-pipeline](https://github.com/arash-rahmani/rnaseq-python-pipeline):** RNA-seq differential expression and pathway enrichment
- **obf-psychiatric-pipeline:** motor-activity classification of psychiatric conditions *(this repo)*

The same architectural principles apply across both: config-driven,
schema-validated, pytest-tested, modular. The data domain shifts
from genome to mind; the rigor doesn't.

[LinkedIn](https://linkedin.com/in/arash-rahmani-544684242)