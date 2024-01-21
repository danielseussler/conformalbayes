# ---
# Conformal prediction intervals for misspecified regression
#
# ---

# %%
import os
import numpy as np
import cmdstanpy
import tomllib

from sklearn.datasets import load_diabetes
from conformalbayes.utils import calculate_confidence_intervals, calculate_coverage
from conformalbayes.core import calculate_conformal_intervals

# %%
parent_dir = os.path.dirname(os.path.dirname(__file__))
config_file = os.path.join(parent_dir, "inferences", "config.toml")

with open(file=config_file, mode="rb") as f:
    config = tomllib.load(f)


# %%
data, target = load_diabetes(return_X_y=True, as_frame=True, scaled=False)

data["sex"] -= 1.0
target = (target - np.median(target)) / np.median(  # type: ignore
    np.absolute(target - np.median(target))  # type: ignore
)

for col in ["age", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]:
    data[col] = (data[col] - np.median(data[col])) / np.median(
        np.absolute(data[col] - np.median(data[col]))
    )

data = data.to_numpy()
target = target.to_numpy()

# create split data sets
np.random.seed(seed=293874)
shuffled_idx = np.random.permutation(len(data))

train_percentage = 0.7
train_size = int(len(data) * train_percentage)

train_idx, test_idx = shuffled_idx[:train_size], shuffled_idx[train_size:]
x_train, x_test = data[train_idx, :3], data[test_idx, :3]
y_train, y_test = target[train_idx], target[test_idx]

#
y_grid = np.linspace(np.min(y_test) - 1, np.max(y_test) + 1, 200)


# %%
stan_data = {
    "N": x_train.shape[0],
    "K": x_train.shape[1],
    "xmat": x_train,
    "y": y_train,
    "P": x_test.shape[0],
    "Q": y_grid.shape[0],
    "xtest": x_test,
    "ytest": y_test,
    "ygrid": y_grid,
    "misspec": 1,
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
# print(f"draws as array:  {samples.draws().shape}")
# print(f"draws as structured object:\n\t{samples.stan_variables().keys()}")
# print(f"sampler diagnostics:\n\t{samples.method_variables().keys()}")


# %%
# print(samples.diagnose())
# print(samples.summary())


# %%
# compute coverage
ci = samples.stan_variable("y_rep_test")
ci = calculate_confidence_intervals(samples=ci, alpha=0.1)

coverage = calculate_coverage(ci, y_test)
width = np.mean(ci[:, 2] - ci[:, 1])

print(f"Coverage: {coverage}")
print(f"Average interval width: {coverage}")


# %%
log_lik = samples.stan_variable("log_lik")
log_lik_grid = samples.stan_variable("log_lik_grid")

conformal = calculate_conformal_intervals(log_lik, log_lik_grid, y_grid, alpha=0.1)
cib = np.concatenate([ci[:, 0][:, np.newaxis], conformal], axis=1)


# %%
# out of sample coverage
# cf. average interval width
coverage = calculate_coverage(cib, y_test)
width = np.mean(cib[:, 2] - cib[:, 1])

print(f"Coverage: {coverage}")
print(f"Average interval width: {width}")
