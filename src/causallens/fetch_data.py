"""
fetch_data.py — Downloads the Criteo Uplift dataset via scikit-uplift
instead of a manual browser download, and saves it where
criteo_validate.py expects to find it.
"""

from sklift.datasets import fetch_criteo

# percent10=True grabs a pre-made 10% sample (~2.5M rows) instead of
# the full 25M -- much faster, and plenty for a portfolio demo
data = fetch_criteo(target_col="conversion", treatment_col="treatment", percent10=True)

df = data.data.copy()
df["treatment"] = data.treatment
df["conversion"] = data.target
df.to_csv("data/raw/criteo-uplift.csv", index=False)

print(df.shape)
print(df.columns.tolist())