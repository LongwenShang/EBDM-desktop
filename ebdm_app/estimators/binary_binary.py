from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammaln, logsumexp
from scipy.stats import chi2, norm


ConfidenceIntervalMethod = Literal["none", "normal", "lr"]


@dataclass(frozen=True)
class BinaryBinaryResult:
    """Results returned by the binary–binary estimator."""

    p1_hat: float
    p2_hat: float
    p11_hat: float
    var_hat: float | None
    se_hat: float | None
    ci_lower: float | None
    ci_upper: float | None


def estimate_binary_binary(
    ni: list[int] | np.ndarray,
    xi: list[int] | np.ndarray,
    yi: list[int] | np.ndarray,
    ci_method: ConfidenceIntervalMethod = "lr",
) -> BinaryBinaryResult:
    """
    Estimate the joint probability of two binary variables from marginal counts.

    Parameters
    ----------
    ni
        Study sample sizes.
    xi
        Marginal event counts for variable 1.
    yi
        Marginal event counts for variable 2.
    ci_method
        One of "none", "normal", or "lr".
    """
    n = np.asarray(ni, dtype=float)
    x = np.asarray(xi, dtype=float)
    y = np.asarray(yi, dtype=float)

    _validate_inputs(n, x, y, ci_method)

    total_n = float(np.sum(n))
    p1_hat = float(np.sum(x) / total_n)
    p2_hat = float(np.sum(y) / total_n)

    epsilon = 1e-8
    lower_bound = max(0.0, p1_hat + p2_hat - 1.0) + epsilon
    upper_bound = min(p1_hat, p2_hat) - epsilon

    if lower_bound >= upper_bound:
        raise ValueError(
            "The marginal summaries do not leave a valid interior range "
            "for the joint probability."
        )

    def log_likelihood(p11: float) -> float:
        return _log_likelihood(
            p11=p11,
            n=n,
            x=x,
            y=y,
            p1=p1_hat,
            p2=p2_hat,
        )

    optimization = minimize_scalar(
        lambda p11: -log_likelihood(p11),
        bounds=(lower_bound, upper_bound),
        method="bounded",
        options={
            "xatol": 1e-10,
            "maxiter": 1000,
        },
    )

    if not optimization.success:
        raise RuntimeError(
            f"Numerical optimization failed: {optimization.message}"
        )

    p11_hat = float(optimization.x)
    maximum_log_likelihood = log_likelihood(p11_hat)

    var_hat = _estimate_variance(
        p11=p11_hat,
        n=n,
        x=x,
        y=y,
        p1=p1_hat,
        p2=p2_hat,
    )

    if var_hat is None:
        se_hat = None
    else:
        se_hat = float(np.sqrt(var_hat))

    ci_lower: float | None = None
    ci_upper: float | None = None

    if ci_method == "normal" and se_hat is not None:
        critical_value = float(norm.ppf(0.975))

        ci_lower = max(
            lower_bound,
            p11_hat - critical_value * se_hat,
        )
        ci_upper = min(
            upper_bound,
            p11_hat + critical_value * se_hat,
        )

    elif ci_method == "lr":
        ci_lower, ci_upper = _likelihood_ratio_interval(
            log_likelihood=log_likelihood,
            p11_hat=p11_hat,
            maximum_log_likelihood=maximum_log_likelihood,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    return BinaryBinaryResult(
        p1_hat=p1_hat,
        p2_hat=p2_hat,
        p11_hat=p11_hat,
        var_hat=var_hat,
        se_hat=se_hat,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def _validate_inputs(
    n: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    ci_method: str,
) -> None:
    """Validate study-level marginal count inputs."""
    if n.ndim != 1 or x.ndim != 1 or y.ndim != 1:
        raise ValueError("All inputs must be one-dimensional vectors.")

    if not (len(n) == len(x) == len(y)):
        raise ValueError("ni, xi, and yi must have equal lengths.")

    if len(n) == 0:
        raise ValueError("At least one study is required.")

    if not (
        np.all(np.isfinite(n))
        and np.all(np.isfinite(x))
        and np.all(np.isfinite(y))
    ):
        raise ValueError("Inputs must not contain missing or infinite values.")

    if np.any(n <= 0):
        raise ValueError("All sample sizes must be positive.")

    if np.any(x < 0) or np.any(y < 0):
        raise ValueError("Event counts cannot be negative.")

    if np.any(x > n) or np.any(y > n):
        raise ValueError("Each event count must be no greater than its sample size.")

    if not (
        np.all(np.equal(n, np.floor(n)))
        and np.all(np.equal(x, np.floor(x)))
        and np.all(np.equal(y, np.floor(y)))
    ):
        raise ValueError("Sample sizes and event counts must be integers.")

    if ci_method not in {"none", "normal", "lr"}:
        raise ValueError(
            "ci_method must be one of 'none', 'normal', or 'lr'."
        )


def _study_log_terms(
    n_i: int,
    x_i: int,
    y_i: int,
    p1: float,
    p2: float,
    p11: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return possible joint event counts and their unnormalized log probabilities.
    """
    p10 = p1 - p11
    p01 = p2 - p11
    p00 = 1.0 - p1 - p2 + p11

    if min(p11, p10, p01, p00) <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    z_min = max(0, x_i + y_i - n_i)
    z_max = min(x_i, y_i)
    z = np.arange(z_min, z_max + 1, dtype=float)

    log_coefficient = (
        gammaln(n_i + 1)
        - gammaln(z + 1)
        - gammaln(x_i - z + 1)
        - gammaln(y_i - z + 1)
        - gammaln(n_i - x_i - y_i + z + 1)
    )

    log_probability = (
        z * np.log(p11)
        + (x_i - z) * np.log(p10)
        + (y_i - z) * np.log(p01)
        + (n_i - x_i - y_i + z) * np.log(p00)
    )

    return z, log_coefficient + log_probability


def _log_likelihood(
    p11: float,
    n: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p1: float,
    p2: float,
) -> float:
    """Compute the total marginal log-likelihood."""
    total = 0.0

    for n_value, x_value, y_value in zip(n, x, y):
        z, log_terms = _study_log_terms(
            n_i=int(n_value),
            x_i=int(x_value),
            y_i=int(y_value),
            p1=p1,
            p2=p2,
            p11=p11,
        )

        if z.size == 0 or not np.any(np.isfinite(log_terms)):
            return float("-inf")

        total += float(logsumexp(log_terms))

    return total


def _estimate_variance(
    p11: float,
    n: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p1: float,
    p2: float,
) -> float | None:
    """Estimate variance using the observed curvature of the likelihood."""
    second_derivative_total = 0.0

    p10 = p1 - p11
    p01 = p2 - p11
    p00 = 1.0 - p1 - p2 + p11

    for n_value, x_value, y_value in zip(n, x, y):
        n_i = int(n_value)
        x_i = int(x_value)
        y_i = int(y_value)

        z, log_terms = _study_log_terms(
            n_i=n_i,
            x_i=x_i,
            y_i=y_i,
            p1=p1,
            p2=p2,
            p11=p11,
        )

        weights = np.exp(log_terms - logsumexp(log_terms))

        first_component = (
            z / p11
            - (x_i - z) / p10
            - (y_i - z) / p01
            + (n_i - x_i - y_i + z) / p00
        )

        curvature_component = (
            z / (p11**2)
            + (x_i - z) / (p10**2)
            + (y_i - z) / (p01**2)
            + (n_i - x_i - y_i + z) / (p00**2)
        )

        expected_second_term = np.sum(
            weights * (first_component**2 - curvature_component)
        )
        expected_first_term = np.sum(weights * first_component)

        second_derivative_total += float(
            expected_second_term - expected_first_term**2
        )

    if (
        not np.isfinite(second_derivative_total)
        or second_derivative_total >= 0
    ):
        return None

    variance = -1.0 / second_derivative_total

    if not np.isfinite(variance) or variance <= 0:
        return None

    return float(variance)



def _likelihood_ratio_interval(
    log_likelihood,
    p11_hat: float,
    maximum_log_likelihood: float,
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, float]:
    """
    Construct the connected 95% likelihood-ratio interval containing the MLE.

    A grid is used to locate the nearest likelihood-ratio crossing on
    each side of the maximum. Brent's method then refines each crossing.
    """
    target = maximum_log_likelihood - 0.5 * float(
        chi2.ppf(0.95, df=1)
    )

    def root_function(value: float) -> float:
        return log_likelihood(value) - target

    def find_crossing(
        grid: np.ndarray,
        prefer_last: bool,
    ) -> float | None:
        values = np.asarray(
            [root_function(value) for value in grid],
            dtype=float,
        )

        brackets: list[tuple[float, float]] = []

        for index in range(len(grid) - 1):
            left_value = values[index]
            right_value = values[index + 1]

            if not (
                np.isfinite(left_value)
                and np.isfinite(right_value)
            ):
                continue

            if left_value == 0:
                return float(grid[index])

            if left_value * right_value < 0:
                brackets.append(
                    (
                        float(grid[index]),
                        float(grid[index + 1]),
                    )
                )

        if not brackets:
            return None

        bracket = brackets[-1] if prefer_last else brackets[0]

        return float(
            brentq(
                root_function,
                bracket[0],
                bracket[1],
                xtol=1e-10,
            )
        )

    left_grid = np.linspace(
        lower_bound,
        p11_hat,
        2000,
    )
    right_grid = np.linspace(
        p11_hat,
        upper_bound,
        2000,
    )

    ci_lower = find_crossing(
        left_grid,
        prefer_last=True,
    )
    ci_upper = find_crossing(
        right_grid,
        prefer_last=False,
    )

    if ci_lower is None:
        ci_lower = float(lower_bound)

    if ci_upper is None:
        ci_upper = float(upper_bound)

    return ci_lower, ci_upper
