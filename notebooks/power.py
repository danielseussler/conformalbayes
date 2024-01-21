# ---
#
#
# ---

# %%
import os
import tomllib

import cmdstanpy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from conformalbayes.utils import calculate_confidence_intervals, calculate_coverage

# %%
parent_dir = os.path.dirname(os.path.dirname(__file__))
config_file = os.path.join(parent_dir, "inferences", "config.toml")

with open(file=config_file, mode="rb") as f:
    config = tomllib.load(f)


# %%
file_path = os.path.join(parent_dir, "data", "CCPP", "Folds5x2_pp.xlsx")
power = pd.read_excel(file_path, sheet_name="Sheet1")


# %%
print(power.shape)
sns.pairplot(power)


# %%
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flatten()

for i, column in enumerate(power.columns):
    ax = axes[i]
    sns.histplot(
        power[column], bins=20, kde=True, color="skyblue", edgecolor="black", ax=ax
    )
    ax.set_title(column)
    ax.grid(True)

plt.tight_layout()
plt.show()


# %%
# technically, I should do this after splitting datasets
power_scaled = np.log(power / power.mean())

# %%
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flatten()

for i, column in enumerate(power.columns):
    ax = axes[i]
    sns.histplot(
        power_scaled[column],
        bins=20,
        kde=True,
        color="skyblue",
        edgecolor="black",
        ax=ax,
    )
    ax.set_title(column)
    ax.grid(True)

plt.tight_layout()
plt.show()


# %%
dataset = power.to_numpy()  # type: ignore

np.random.seed(seed=293874)
shuffled_idx = np.random.permutation(len(dataset))

train_percentage = 0.7
train_size = int(len(dataset) * train_percentage)

train_idx, test_idx = shuffled_idx[:train_size], shuffled_idx[train_size:]
x_train, x_test = dataset[train_idx, :3], dataset[test_idx, :3]
y_train, y_test = dataset[train_idx, 4], dataset[test_idx, 4]

x_train = np.log(x_train)
x_test = np.log(x_test)

y_train -= y_train.mean()
y_test -= y_test.mean()

# %%
stan_data = {
    "N": x_train.shape[0],
    "K": x_train.shape[1],
    "xmat": x_train,
    "y": y_train,
}


# %%
stan_file = os.path.join(parent_dir, "src", "stan", "linear.stan")
model = cmdstanpy.CmdStanModel(stan_file=stan_file)
samples = model.sample(
    data=stan_data,
    iter_warmup=1_000,
    iter_sampling=2_000,
    output_dir=os.path.join(parent_dir, "inferences"),
)

# %%
print(f"draws as array:  {samples.draws().shape}")
print(f"draws as structured object:\n\t{samples.stan_variables().keys()}")
print(f"sampler diagnostics:\n\t{samples.method_variables().keys()}")

print(samples.diagnose())
print(samples.summary())


# %%
df = samples.summary()
df.head(n=10)


# %%
# check in-sample calibration
ci = samples.stan_variable("y_rep")
ci = calculate_confidence_intervals(samples=ci)

# %%
plt.scatter(power.iloc[train_idx, 4], ci[:, 0])
plt.xlim((400, 600))
plt.ylim((400, 600))

plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()


# %%
plt.errorbar(
    power.iloc[train_idx, 4],
    ci[:, 0],
    yerr=ci[:, 1:3].T,
    fmt="o",
    color="red",
    label="Predicted vs True",
)
plt.xlim((400, 600))
plt.ylim((-100, 1000))

plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()


# %%
# compute coverage
coverage = calculate_coverage(ci, y_train)
print(f"Coverage: {coverage}")

# %%
