# abroca

Predictive performance fairness analysis metric for model outputs.\
\
**Data:** 
- predicted probabilities
- actual target
- sensitive attributes (from metadata).

**Output:**
- ABROCA value
- p-value from ABROCA permutation test (if number of bootstraps is specified)
- side-by-side graph of ROC curves with ABROCA and permutation histogram with $99^{th}$ percentile

## abroca.py
The file that has the abroca, graph, and bootstrap functions

## execute_abroca.ipynb
The notebook that executes abroca.py with guides.
