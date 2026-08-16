# MASTER THESIS IN COMPUTER SCIENCE

## Assessing Fairness in Sequential Decision-Making under Uncertainty with Contextual Multi-Armed Bandits

**Author:** Jean-Nicolas Grégoire  
**Promoter:** Professor Tom Lenaerts
**Supervisor:** Axel Abels

This repository contains the code and notebooks used for my master's thesis on fairness with contextual multi-armed bandits. The project studies how fairness-aware interventions behave when decisions are made sequentially under uncertainty, and whether group disparities can be reduced while preserving predictive utility.

The repository provides reusable Python modules for dataset preparation, contextual bandit policies, experiment runners, metrics, plotting, post-processing, statistical tests, and the notebooks used to reproduce the final thesis experiments and figures.

---

## Repository Layout

- [`src/fair_bandits`](src/fair_bandits): reusable library code.
  - `data`: dataset loading and preprocessing.
  - `experiments`: experiment runners and replay logic.
  - `io`: loading, saving, and LaTeX/CSV table export.
  - `metrics`: utility, fairness, temporal, and statistical metrics.
  - `plots`: plotting utilities for Adult, COMPAS, and synthetic experiments.
  - `policies`: contextual bandit policies and fairness-aware variants.
  - `postprocessing`: group-specific threshold post-processing.
- [`notebooks`](notebooks): notebooks used for experiments, robustness checks, and figure generation.
- [`notebooks/results`](notebooks/results): generated notebook outputs such as CSV files and PNG figures.
- [`results`](results): additional generated outputs from module-based runs.
- [`requirement.txt`](requirement.txt): project dependencies.
- [`pyproject.toml`](pyproject.toml): project/package configuration.

---

## Recommended Notebooks

The cleanest and most recent notebooks are the module-based thesis notebooks:

1. [`notebooks/synthetic_cmab_fairness.ipynb`](notebooks/synthetic_cmab_fairness.ipynb)  
   Final synthetic CMAB notebook. It evaluates regime-matched fairness robustness across deterministic, stochastic, and adversarial synthetic regimes, including the 80/20 group-imbalance sensitivity analysis.

2. [`notebooks/adult_sex_cmab_fairness.ipynb`](notebooks/adult_sex_cmab_fairness.ipynb)  
   Final Adult Income experiment using sex as the sensitive attribute. It compares LinUCB, Linear Thompson Sampling, and EXP4 policy families with pre-processing, in-processing, and post-processing fairness interventions. The notebook generates temporal figures, final summary tables, post-processing analyses, and paired statistical tests.

3. [`notebooks/compas_race_binary_fairness.ipynb`](notebooks/compas_race_binary_fairness.ipynb)  
   Final COMPAS experiment using binary race as the sensitive attribute. It compares LinUCB, Linear Thompson Sampling, and EXP4 policy families with uniform and reweighted preprocessing, fairness-aware in-processing variants, post-processing, final summary tables, and paired statistical tests.

These three notebooks should be treated as the canonical entry points for reproducing the final thesis results.

---

## What the Project Studies

The experiments evaluate fairness in contextual bandit settings. Adult Income, COMPAS, and synthetic environments are reformulated as binary contextual bandit problems:

- each individual or simulated observation is represented by a context vector;
- the available actions are binary decisions;
- in the Adult and COMPAS experiments, the reward is equal to 1 when the selected action matches the recoded binary label, and 0 otherwise.

The meaning of class 1 is dataset-specific and must not be confused with a reward of 1:

| Dataset | Original outcome | Recoded label and action semantics | Reward |
| --- | --- | --- | --- |
| Adult Income | Annual income category | `y = 1` and `a = 1` denote income `>50K`; `y = 0` and `a = 0` denote income `<=50K`. | `r = 1` if `a == y`, and `r = 0` otherwise. |
| COMPAS | `two_year_recid`, where 1 denotes recidivism | The code applies `y = 1 - two_year_recid`. Consequently, `y = 1` and `a = 1` denote **non-recidivism**, whereas `y = 0` and `a = 0` denote recidivism. | `r = 1` if `a == y`, and `r = 0` otherwise. |

Thus, a value of 1 has three distinct interpretations: it denotes the positive recoded class when used as a label, the prediction of that class when used as an action, and a correct prediction when used as a reward. Because Adult and COMPAS are offline classification-derived bandit environments, **average reward is equivalent to accuracy** in those experiments. The final analyses report average reward together with the selected fairness gaps and UtilityGap; cumulative prediction error is retained only in legacy or exploratory outputs.

The project evaluates fairness interventions at three levels:

1. **Pre-processing:** group-label reweighting before or during the replay stream.
2. **In-processing:** demographic-parity-aware corrections of the action-selection rule.
3. **Post-processing:** group-specific threshold calibration on held-out data.

---

## Policy Families

The codebase includes three main contextual bandit families and fairness-aware variants:

- **LinUCB family**
  - `LinUCB`
  - `FairLinUCB` / `FairLinUCB+PP`
- **Linear Thompson Sampling family**
  - `LinTS`
  - `FairLinTS` / `FairLinTS+PP`
- **EXP4 family**
  - `EXP4`
  - `FairEXP4` / `FairEXP4+PP`

Some older notebooks also contain supervised baselines, exploratory comparisons, or earlier versions of the experiments.

---

## Metrics

### Utility and performance metrics

- Average reward
- UtilityGap
- (Cumulative prediction error)

In Adult and COMPAS, average reward corresponds to classification accuracy because the reward is defined as `1(action == y_true)`.

### Fairness metrics

- Demographic Parity Gap (`DP_gap`)
- True Positive Rate Gap (`TPR_gap`)
- False Positive Rate Gap (`FPR_gap`)
- Equalized Odds Gap (`EO_gap`)
- Positive Predictive Value Gap (`PPV_gap`)
- UtilityGap

Lower values indicate smaller disparities for the fairness-gap metrics and UtilityGap. Higher average reward indicates better predictive utility.

---

## Legacy and Exploratory Notebooks

Several older notebooks are retained in the repository for provenance and reproducibility of earlier work, including the experiments prepared for the **MLG Student Days** and intermediate thesis-development stages.

Examples include:

- `fairness2`
- `adult_experiments.ipynb`
- `adult_processing_methods_comparison.ipynb`
- `compas_experiments.ipynb`
- `compas_postprocessing.ipynb`
- `compas_snapshot_benchmark.ipynb`
- `synthetic_cmab_experiments.ipynb`
- `final_figures.ipynb`
- `processing_methods_comparison.ipynb`

These notebooks are useful for understanding the development history of the project, but the three notebooks listed in the **Recommended Notebooks** section are the cleanest and latest versions.

---

## Statistical Analysis

The final workflow uses paired comparisons across random seeds. Pairwise Wilcoxon signed-rank tests are used for paired comparisons, and Holm correction is applied for multiple comparisons.

Temporal plots report mean trajectories across seeds with pointwise confidence intervals. Compact final tables report mean and standard deviation across seeds.

---

## Reproducibility Notes

The final notebooks can either regenerate experiments or load cached outputs, depending on the run flags set inside each notebook. Full runs may take substantial time, especially for multi-seed Adult, COMPAS, and synthetic experiments.

Generated outputs are written to the configured results directories and include:

- temporal trajectory CSV files;
- endpoint summary CSV files;
- post-processing result CSV files;
- paired statistical-test tables;
- LaTeX tables for Overleaf;
- PNG figures for the main thesis results and appendix sections.

---

## Requirements

Python 3.11 or newer is recommended.

Install dependencies with:

```bash
pip install -r requirement.txt
```

---

## Suggested Workflow

A typical workflow is:

1. Create and activate a virtual environment.
2. Install the dependencies.
3. Open one of the three recommended notebooks.
4. Select whether to regenerate experiments or load cached results.
5. Run the notebook to reproduce figures, tables, and statistical summaries.
