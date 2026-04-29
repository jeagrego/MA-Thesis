# MASTER THESIS IN COMPUTER SCIENCE

## Assessing Fairness in Decision-Making Algorithms for Contextual Bandit Problems

## Author : Jean-Nicolas Grégoire

## Promoter : Professor Tom Lenaerts

This repository contains the code and notebooks used for my master's thesis on fairness in contextual bandits. It provides reusable Python modules for dataset preparation, policy implementations, experiment runners, metrics, plotting, and the notebooks used to reproduce the thesis experiments and figures.

## Repository Layout

- [src/fair_bandits](src/fair_bandits) contains the reusable library code.
- [notebooks](notebooks) contains the thesis notebooks for Adult, COMPAS, synthetic CMAB, postprocessing, benchmark comparisons, and final figures.
- [notebooks/results](notebooks/results) stores generated CSV and PNG outputs from notebook runs.
- [requirement.txt](requirement.txt) lists the project dependencies.

## What The Project Studies

The experiments study the trade-off between predictive utility and group fairness in contextual bandit settings. The repository compares standard and fairness-aware policies across multiple benchmarks, including Adult, COMPAS, and synthetic CMAB environments.

## Policies And Baselines

The codebase currently includes these policy families:

- LinUCB
- GroupAwareDPLinUCB / FairLinUCB_DP_GroupAware
- LinearThompsonSampling
- GroupAwareDPLinearThompsonSampling / FairLinearThompsonSampling_DP_GroupAware
- EXP4
- GroupAwareDPEXP4 / FairEXP4_DP
- supervised baselines used in the benchmark notebooks

## Metrics

Utility and performance:

- Accuracy
- Utility gap
- Cumulative regret

### Fairness metrics

- Demographic Parity gap (`DP_gap`)
- True Positive Rate gap (`TPR_gap`)
- False Positive Rate gap (`FPR_gap`)
- Equalized Odds gap (`EO_gap`)
- Positive Predictive Value gap (`PPV_gap`)

## Notebook Set

The notebooks in this repository cover:

1. Adult experiments
2. COMPAS experiments
3. COMPAS postprocessing
4. COMPAS snapshot benchmark
5. Synthetic CMAB experiments
6. Final figures and statistical summaries

The synthetic CMAB notebook includes multiple regimes, horizon sweeps, per-seed summaries, confidence intervals, and statistical tests.

## Statistical Analysis

The experiment workflow includes Friedman omnibus tests, pairwise Wilcoxon tests, and Holm correction for multiple comparisons.

## Requirements

Python 3.11 or newer is recommended.

Install dependencies with:

```bash
pip install -r requirement.txt
```
