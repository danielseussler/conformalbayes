import numpy as np


def calculate_confidence_intervals(
    samples: np.ndarray, alpha: float = 0.05, func: str = "mean"
) -> np.ndarray:
    """
    Compute the credible interval for each row of the input array
    based on the empirical distribution of the posterior.
    """

    assert 0 < alpha < 0.5, "Value should be between 0 and 0.5"
    assert len(samples.shape) == 2, "Dimensions do not match length 2"
    assert func in ["mean", "median"], "func should be equal to 'mean' or 'median'"

    _, n = samples.shape
    result = np.zeros((n, 3))

    if func == "median":
        result[:, 0] = np.median(samples, axis=0)
    else:
        result[:, 0] = np.mean(samples, axis=0)

    result[:, 1] = np.percentile(samples, (alpha / 2) * 100, axis=0)
    result[:, 2] = np.percentile(samples, (1 - alpha / 2) * 100, axis=0)

    return result


def calculate_coverage(ci: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate coverage of posterior interval."""
    
    assert len(y_true.shape) == 1
    coverage = (y_true >= ci[:, 1]) & (y_true <= ci[:, 2])
    
    return np.mean(coverage) # type: ignore
    