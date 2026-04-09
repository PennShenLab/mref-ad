"""Explore ADNI diagnosis summaries (DXSUM-style table).

Loads a longitudinal diagnosis extract and prints:
  - column names (sanity check),
  - value counts for ``DIAGNOSIS`` and ``DXAD``,
  - a crosstab of ``DIAGNOSIS`` × ``DXAD``,
  - ``DXAD`` distribution restricted to dementia rows (``DIAGNOSIS == 3``).

The default CSV path is relative to the **repository root** (run this script after
``cd`` to the repo, or adjust the path below).

Example::

    python data_preprocessing/explore_diagnosis.py
"""

import pandas as pd

# Load the diagnosis file
df = pd.read_csv("data/DXSUM_PDXCONV_22Aug2024.csv")

# Peek at the columns (to confirm names)
print(df.columns)

# Let's assume the high-level diagnosis is in a column called 'DIAGNOSIS'
# and the AD attribution flag is in 'DXAD'

# First, get counts for the overall DIAGNOSIS column
print("\nDiagnosis value counts:")
print(df['DIAGNOSIS'].value_counts(dropna=False))

# Now look at how DXAD is distributed overall
print("\nDXAD value counts:")
print(df['DXAD'].value_counts(dropna=False))

# Cross-tab: Dementia vs. DXAD
crosstab = pd.crosstab(df['DIAGNOSIS'], df['DXAD'], dropna=False)
print("\nCrosstab of DIAGNOSIS vs DXAD:")
print(crosstab)

# Specifically for DIAGNOSIS = 3 (Dementia)
dementia_cases = df[df['DIAGNOSIS'] == 3]

print("\nBreakdown of DXAD values within Dementia cases:")
print(dementia_cases['DXAD'].value_counts(dropna=False))

# Summary percentages
print("\nPercent breakdown of DXAD values within Dementia:")
print(dementia_cases['DXAD'].value_counts(normalize=True, dropna=False) * 100)