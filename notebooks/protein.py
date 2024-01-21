# ---
# Dataset obtained from
# Physicochemical Properties of Protein Tertiary Structure [Protein](https://doi.org/10.24432/C5QW3H)
# ---

# %%
import os
import tomllib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
parent_dir = os.path.dirname(os.path.dirname(__file__))
config_file = os.path.join(parent_dir, "inferences", "config.toml")

with open(file=config_file, mode="rb") as f:
    config = tomllib.load(f)

# %%
file_path = os.path.join(parent_dir, "data", "CASP", "CASP.csv")
protein = pd.read_csv(file_path)

# %%
sns.pairplot(protein)

# %%
sns.histplot(data=protein["RMSD"])

# %%
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
axes = axes.flatten()

for i, column in enumerate(protein.columns[1:]):
    ax = axes[i]
    sns.histplot(
        protein[column], bins=20, kde=True, color="skyblue", edgecolor="black", ax=ax
    )
    ax.set_title(column)
    ax.grid(True)

plt.tight_layout()
plt.show()
