import numba as nb
import numpy as np


@nb.njit(fastmath=True)
def calculate_rank(log_lik: np.ndarray, log_lik_grid: np.ndarray) -> np.ndarray:
    # log_lik T x N
    # log_lik_grid T x 100 (y_n+1)
    # T = np.shape(log_lik)[0]
    N = np.shape(log_lik)[1]
    P = np.shape(log_lik_grid)[1]

    lik = np.exp(log_lik)
    w = np.exp(log_lik_grid)
    w_sum = np.sum(w, axis=0)  # sum over T
    w_normalisedT = np.transpose(w / w_sum)

    ppd = np.zeros((P, N + 1))
    ppd[:, :-1] = np.dot(w_normalisedT, lik)
    ppd[:, -1] = np.sum(w**2, axis=0) / w_sum

    rank = np.sum(ppd <= ppd[:, -1][:, np.newaxis], axis=1)

    return rank


@nb.njit(fastmath=True)
def create_mask(
    log_lik: np.ndarray, log_lik_grid: np.ndarray, alpha: float = 0.05
) -> np.ndarray:
    assert 0 < alpha < 0.5, "Value should be between 0 and 0.5"
    assert len(log_lik.shape) == 2, "Dimensions do not match length 2"

    N = np.shape(log_lik)[1]

    rank = calculate_rank(log_lik, log_lik_grid)
    mask = rank > alpha * (N + 1)

    return mask


@nb.njit(fastmath=True)
def calculate_conformal_intervals(
    log_lik: np.ndarray, log_lik_grid: np.ndarray, grid: np.ndarray, alpha: float = 0.05
) -> np.ndarray:
    assert 0 < alpha < 0.5, "Value should be between 0 and 0.5"
    assert np.shape(log_lik)[0] == np.shape(log_lik_grid)[0]
    assert grid.shape[0] == log_lik_grid.shape[2], "Shapes do not match"

    # T = np.shape(log_lik)[0]
    # N = np.shape(log_lik)[1]
    P = np.shape(log_lik_grid)[1]
    ci = np.zeros((P, 2))

    # iterate through observations in xtest
    for k in range(P):
        log_lik_grid_subset = log_lik_grid[:, k, :]

        masked_ci = create_mask(
            log_lik=log_lik, log_lik_grid=log_lik_grid_subset, alpha=alpha
        )

        indices_of_ones = np.where(masked_ci == 1)[0]

        ci[k, 0] = grid[indices_of_ones[0]]
        ci[k, 1] = grid[indices_of_ones[-1]]

    return ci
