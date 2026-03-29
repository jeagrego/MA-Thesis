# Fairness in Bandit Settings on COMPAS

This repository contains the notebook used for my master's thesis experiments on fairness in bandit settings.  
The goal is to compare a standard contextual bandit approach with a fairness-aware variant, and to benchmark them against supervised baselines on the COMPAS dataset.

## Main idea

The notebook studies the trade-off between predictive utility and group fairness.  
The main methods compared are:

- **LinUCB**
- **FairLinUCB_DP_GroupAware**
- **Logistic Regression**
- **Random Forest**
- **FairLinUCB_DP_GroupAware_PostProc** (main addition for the thesis: postprocessed version with group-specific thresholds)

## Main metrics

The experiments report:

- Accuracy
- Demographic Parity gap (`DP_gap`)
- True Positive Rate gap (`TPR_gap`)
- False Positive Rate gap (`FPR_gap`)
- Equalized Odds gap (`EO_gap`)
- Positive Predictive Value gap (`PPV_gap`)
- Utility gap
- Cumulative regret

## Notebook content

The notebook contains four main parts:

1. **Quick temporal CMAB experiment**  
   Used mainly for diagnostic temporal plots.

2. **Unified snapshot benchmark**  
   Comparison of LinUCB, FairLinUCB, Logistic Regression, and Random Forest across several horizons.

3. **Postprocessing benchmark**  
   Evaluation of a postprocessed fairness-aware bandit using group-specific thresholds learned on a calibration split.

4. **Statistical testing and final tables**  
   Friedman omnibus tests, pairwise Wilcoxon tests, Holm correction, and final summary tables.

## Important note

The **main reported thesis results** are the snapshot benchmark and the postprocessing benchmark.  
The quick temporal plots are exploratory/diagnostic and are not the main quantitative comparison.


## Required environment

Recommended Python version: **3.10+**

Install dependencies with:

```bash
pip install -r requirements.txt
