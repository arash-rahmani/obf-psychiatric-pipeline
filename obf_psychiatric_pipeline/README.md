# obf-psychiatric-pipeline

> Reproducible classification of psychiatric conditions from wrist-worn
> motor activity — applying the same pipeline philosophy as my
> [RNA-seq pipeline](https://github.com/arash-rahmani/rnaseq-python-pipeline)
> to a fundamentally different data domain: from genome to behavior.

**Language:** Python 3.10+ &nbsp;|&nbsp;
**Dataset:** OBF-Psychiatric (Garcia-Ceja et al., 2024) &nbsp;|&nbsp;
**Models:** logistic regression, XGBoost, dummy baseline &nbsp;|&nbsp;
**Tests:** pytest

---

## Headline finding

Wrist-worn motor activity distinguishes psychiatric inpatients from
healthy controls (**macro-F1 0.80, 95% CI 0.70–0.88**) but does not
reliably distinguish depression from schizophrenia
(**macro-F1 0.60, 95% CI 0.48–0.71**). The two patient cohorts share
a common low-activity, high-sedentary motor signature — suggesting that
distributional features of activity capture *inpatient status and
medication effects* rather than *disorder-specific behavior*.

This is the project's central result: motor activity alone, summarized
distributionally, is sufficient for screening but insufficient for
differential diagnosis between major psychiatric conditions.

**Per-participant headline numbers (5-fold GroupKFold CV, bootstrap CIs):**

| Task | Classifier | Macro-F1 | 95% CI | Lift over dummy |
|---|---|---|---|---|
| Control vs Patient | Logistic Regression | **0.798** | 0.700 – 0.881 | +0.19 |
| Control vs Patient | XGBoost | 0.755 | 0.646 – 0.841 | +0.15 |
| Control vs Depression vs Schizophrenia | Logistic Regression | **0.595** | 0.480 – 0.705 | +0.21 |
| Control vs Depression vs Schizophrenia | XGBoost | 0.499 | 0.386 – 0.602 | +0.11 |

![Binary confusion matrix](results/figures/confusion_2class_per_participant_logreg.png)

> The binary classifier confuses control with patient cases primarily
> at the activity-level boundary — low-activity controls and
> higher-functioning patients overlap.

![PCA projection](results/eda/pca_projection.png)

> PC1 (65.6%) captures overall activity level and creates a class
> gradient. PC2 (17.6%) does not separate cohorts. The geometry is
> nearly one-dimensional, which is why a linear classifier matches
> XGBoost on this feature set.

---

## What this pipeline does

This pipeline takes wrist-worn motor activity data from psychiatric
inpatients and healthy controls, and runs an honest, reproducible
classification analysis that respects the structure of the data:

- **Dual task framing** — both 3-class (control / depression /
  schizophrenia) and 2-class (control / patient). The contrast between
  the two is the finding.
- **Dual aggregation** — both per-day and per-participant
  classification, with participant-level cross-validation in both cases
  to prevent within-subject leakage.
- **Three classifiers** — stratified dummy (floor), logistic regression
  (interpretable linear baseline), XGBoost (non-linear). All compared
  fairly on identical folds.
- **Bootstrap 95% CIs on every metric** — point estimates lie when
  n=75; intervals are honest.
- **SHAP-based feature attribution** on the best non-linear model.
- **Config-driven, schema-validated, pytest-tested** — same
  architectural philosophy as my RNA-seq pipeline.

---

## Pipeline overview

```
Raw OBF metadata (5 cohorts) + features.csv (3 cohorts)
                 │
                 ▼
        Schema-validated loader
                 │
                 ▼
   Preprocessing
   • drop participants with < 7 recording days
   • drop q25 (uninformative — saturates at zero)
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
   Per-day  ‖  Per-participant aggregation
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
    loader.py                   # schema-validated loaders
    preprocess.py               # min-days filter, feature exclusion
    split.py                    # participant-level GroupKFold
  models/
    classifiers.py              # estimator factories
    aggregate.py                # per-day → per-participant
    relabel.py                  # 3-class → 2-class
    evaluate.py                 # metrics + bootstrap CIs
    train.py                    # 2×2×3 experiment grid
  viz/
    eda.py                      # 5 EDA plots
    confusion.py                # confusion matrices
    roc.py                      # ROC curves
    shap_plots.py               # SHAP attribution
config/config.yaml              # paths, split seed, exclusions
scripts/                        # CLI entry points
tests/                          # 17 pytest tests
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
`*info.csv` and `features.csv`).

```powershell
python scripts/load_data.py        # smoke test
python scripts/run_eda.py          # generates 5 EDA plots
python scripts/train_models.py     # runs the 2×2×3 experiment grid
python scripts/run_viz.py          # confusion matrices, ROC, SHAP
pytest tests/                      # 17 tests
```

---

## Key design decisions

**Why dual task framing (3-class and 2-class)?**
Exploratory PCA showed that depression and schizophrenia largely
overlap in feature space while both separate cleanly from controls.
Reporting only the 3-class result would have hidden the actual
structure of the data. Reporting both — and the gap between them —
makes the finding visible.

**Why participant-level splitting, not row-level?**
Each participant contributes ~12 rows (one per recording day). Random
row-level splitting puts the same participant in both train and test,
which inflates measured performance via subject-identity leakage. This
is the most common methodological error in actigraphy-based
classification literature. `GroupKFold(groups=user)` enforces
participant-level independence.

**Why both per-day and per-participant aggregation?**
Per-day uses more data (~880 samples) but day-rows from the same
person are not independent — confidence intervals computed on per-day
data are artificially tight. Per-participant aggregation (~75 samples)
respects the true unit of clinical inference and gives honest CIs.
Both are reported; the per-participant numbers are the ones to trust.

**Why logistic regression as a baseline alongside XGBoost?**
Two reasons. First, multicollinearity: the six features collapse to
roughly two independent dimensions (PC1 = 65.6%, PC2 = 17.6%), so
linear models are well-matched to the geometry. Second,
interpretability: when logreg matches or exceeds XGBoost, it tells you
the problem is near-linear and that model complexity isn't the path
to better performance — *feature engineering is*. That finding shapes
the next iteration of this work.

**Why no hyperparameter tuning?**
With 22 participants in the smallest cohort, nested CV hyperparameter
search is mostly noise. Default sklearn / XGBoost hyperparameters were
used (XGBoost: `n_estimators=200, max_depth=4, learning_rate=0.05`,
logreg: `C=1.0`, both with `class_weight="balanced"`). The focus of
this work is methodological framing, not hyperparameter optimization.

**Why bootstrap confidence intervals?**
At n=75 participants, point estimates of macro-F1 are unstable across
resampling. The bootstrap CI honestly reflects how much uncertainty
remains in the headline number. A reported macro-F1 of 0.798 without
its 0.700–0.881 interval would be misleading.

---

## Findings

**1. Binary classification is solid; ternary classification is partial.**
Control-vs-patient hits macro-F1 0.80 (CI 0.70–0.88). Three-class
hits 0.60 (CI 0.48–0.71). Both clear their dummy baselines convincingly.
The 20-point gap between them is the project's central finding.

**2. Linear models match XGBoost on this feature set.**
Logistic regression beat or matched XGBoost across most experiments
(0.798 vs 0.755 on binary per-participant, 0.595 vs 0.499 on ternary
per-participant). The feature space is near-linear; gradient boosting's
extra capacity has nothing to exploit. This is a finding, not a
disappointment — it points directly at where modeling effort should
go next: feature engineering, not model complexity.

**3. One pre-computed feature carries no signal.**
The 25th percentile of per-minute activity (`q25`) saturates at zero
across all classes — every participant spends substantial time
motionless during sleep, regardless of diagnosis. This feature was
excluded from modeling.

**4. Confidence intervals widen honestly with proper aggregation.**
Per-day CIs were ~0.06 wide; per-participant CIs were ~0.18 wide for
the same metric. The wider intervals are the honest ones; the narrow
ones reflect within-participant correlation, not statistical certainty.

---

## Limitations

- **Sample size: n = 77 participants total** (32 control, 23 depression,
  22 schizophrenia). Bootstrap CIs reflect this; readers should weight
  point estimates accordingly.
- **Distributional features only.** The pre-computed feature matrix
  contains six summary statistics per day. It captures *how much* a
  participant moves but not *when*, *for how long*, or *with what
  rhythmic structure*. Circadian and temporal-structure features are
  the natural next step.
- **Inpatient cohorts on medication.** Both patient groups were
  recorded during inpatient stays. Antipsychotic and antidepressant
  medications have known motor effects. Results characterize "psychiatric
  inpatient on medication" as much as "depression" or "schizophrenia"
  per se, and do not generalize to outpatient or unmedicated populations.
- **Two patient cohorts only.** ADHD and clinical-control metadata are
  in the OBF dataset but the pre-computed feature matrix excludes them.
  Extending to a 4- or 5-class problem requires computing features
  from raw actigraphy.
- **The OBF-Psychiatric dataset combines two earlier studies**
  (Depresjon and Psykose) with different recording protocols. Some
  cross-cohort confounds may reflect study-of-origin rather than
  diagnosis.
- **No external validation.** All performance numbers come from
  cross-validation on a single dataset. Generalization to other
  actigraphy datasets is untested.

---

## Bridge to next work

The next iteration of this pipeline will compute custom temporal
features from raw actigraphy — interdaily stability (IS), intradaily
variability (IV), L5/M10 (least-active 5h, most-active 10h), cosinor
amplitude and acrophase, and activity fragmentation indices — to test
whether circadian and rhythmic structure can disambiguate the patient
cohorts that distributional features cannot.

Whether they can or cannot is itself the question.

This pipeline establishes the methodological backbone — schema-validated
loading, participant-level CV, bootstrap-CI'd evaluation, dual task
framing — for that experiment.

---

## Citation

Data: Enrique Garcia-Ceja, Andrea Stautland, Michael A. Riegler, Pål
Halvorsen, Salvador Hinojosa, Gilberto Ochoa-Ruiz, Ketil Joachim
Oedegaard, and Petter Jakobsen. *OBF-Psychiatric, a motor activity
dataset of depressed, schizophrenic, and attention deficit
hyperactivity disorder patients*, 2024.

---

## About

Built by [Arash Rahmani](https://github.com/arash-rahmani) — M.Sc.
Bioinformatics, Julius-Maximilians-Universität Würzburg.

This is the second in a series of reproducible analysis pipelines
spanning biological and behavioral data:

- **[rnaseq-python-pipeline](https://github.com/arash-rahmani/rnaseq-python-pipeline)** — RNA-seq differential expression and pathway enrichment
- **obf-psychiatric-pipeline** — motor-activity classification of psychiatric conditions *(this repo)*

The same architectural principles — config-driven, schema-validated,
pytest-tested, modular — apply across both. The data domain shifts
from genome to mind; the rigor doesn't.

[LinkedIn](https://linkedin.com/in/arash-rahmani-544684242)