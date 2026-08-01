from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import gammaln, ive
from scipy.stats import chi2, norm


EstimationMethod = Literal["proposed", "weighted"]
ConfidenceIntervalMethod = Literal["none", "normal", "lr"]


@dataclass(frozen=True)
class ContinuousContinuousResult:
    """Results returned by the continuous–continuous estimator."""

    mu_x: float
    mu_y: float
    sigma_x: float
    sigma_y: float
    rho: float
    se: float | None
    ci_lower: float | None
    ci_upper: float | None
    method: str


def estimate_continuous_continuous(
    n: list[int] | np.ndarray,
    xbar: list[float] | np.ndarray,
    ybar: list[float] | np.ndarray,
    s2x: list[float] | np.ndarray | None = None,
    s2y: list[float] | np.ndarray | None = None,
    method: EstimationMethod = "proposed",
    ci_method: ConfidenceIntervalMethod = "lr",
) -> ContinuousContinuousResult:
    """
    Estimate a bivariate normal distribution from marginal summaries.

    The proposed method uses sample sizes, means, and variances.
    The weighted method uses only sample sizes and means.
    """
    n_array = np.asarray(n, dtype=float)
    xbar_array = np.asarray(xbar, dtype=float)
    ybar_array = np.asarray(ybar, dtype=float)

    s2x_array = None if s2x is None else np.asarray(s2x, dtype=float)
    s2y_array = None if s2y is None else np.asarray(s2y, dtype=float)

    _validate_inputs(
        n=n_array,
        xbar=xbar_array,
        ybar=ybar_array,
        s2x=s2x_array,
        s2y=s2y_array,
        method=method,
        ci_method=ci_method,
    )

    total_n = float(np.sum(n_array))
    number_of_studies = len(n_array)

    mu_x = float(np.sum(n_array * xbar_array) / total_n)
    mu_y = float(np.sum(n_array * ybar_array) / total_n)

    if method == "weighted":
        sx2 = float(
            np.sum(n_array * (xbar_array - mu_x) ** 2)
            / number_of_studies
        )
        sy2 = float(
            np.sum(n_array * (ybar_array - mu_y) ** 2)
            / number_of_studies
        )

        if sx2 <= 0 or sy2 <= 0:
            raise ValueError(
                "The weighted method requires variation in both sets "
                "of study-level means."
            )

        covariance = float(
            np.sum(
                n_array
                * (xbar_array - mu_x)
                * (ybar_array - mu_y)
            )
            / number_of_studies
        )

        rho = float(
            np.clip(
                covariance / np.sqrt(sx2 * sy2),
                -1.0,
                1.0,
            )
        )

        return ContinuousContinuousResult(
            mu_x=mu_x,
            mu_y=mu_y,
            sigma_x=float(np.sqrt(sx2)),
            sigma_y=float(np.sqrt(sy2)),
            rho=rho,
            se=None,
            ci_lower=None,
            ci_upper=None,
            method="Weighted mean method",
        )

    assert s2x_array is not None
    assert s2y_array is not None

    sx2 = float(
        np.sum(
            n_array * (xbar_array - mu_x) ** 2
            + (n_array - 1.0) * s2x_array
        )
        / total_n
    )
    sy2 = float(
        np.sum(
            n_array * (ybar_array - mu_y) ** 2
            + (n_array - 1.0) * s2y_array
        )
        / total_n
    )

    if sx2 <= 0 or sy2 <= 0:
        raise ValueError(
            "The pooled marginal variances must be positive."
        )

    sigma_x = float(np.sqrt(sx2))
    sigma_y = float(np.sqrt(sy2))

    def log_likelihood(rho: float) -> float:
        return _log_likelihood(
            rho=rho,
            n=n_array,
            xbar=xbar_array,
            ybar=ybar_array,
            s2x=s2x_array,
            s2y=s2y_array,
            mu_x=mu_x,
            mu_y=mu_y,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
        )

    grid = np.linspace(-0.99, 0.99, 801)
    grid_values = np.asarray(
        [log_likelihood(rho) for rho in grid],
        dtype=float,
    )

    if not np.any(np.isfinite(grid_values)):
        raise RuntimeError(
            "The likelihood could not be evaluated for these inputs."
        )

    start_rho = float(grid[int(np.nanargmax(grid_values))])

    optimization = minimize(
        fun=lambda parameter: -log_likelihood(float(parameter[0])),
        x0=np.asarray([start_rho]),
        method="L-BFGS-B",
        bounds=[(-0.99, 0.99)],
    )

    if optimization.success:
        rho_hat = float(optimization.x[0])
    else:
        rho_hat = start_rho

    if log_likelihood(start_rho) > log_likelihood(rho_hat):
        rho_hat = start_rho

    maximum_log_likelihood = log_likelihood(rho_hat)

    se_hat = _estimate_standard_error(
        log_likelihood=log_likelihood,
        rho_hat=rho_hat,
    )

    ci_lower: float | None = None
    ci_upper: float | None = None

    if ci_method == "normal" and se_hat is not None:
        critical_value = float(norm.ppf(0.975))
        ci_lower = max(-1.0, rho_hat - critical_value * se_hat)
        ci_upper = min(1.0, rho_hat + critical_value * se_hat)

    elif ci_method == "lr":
        ci_lower, ci_upper = _likelihood_ratio_interval(
            log_likelihood=log_likelihood,
            rho_hat=rho_hat,
            maximum_log_likelihood=maximum_log_likelihood,
        )

    return ContinuousContinuousResult(
        mu_x=mu_x,
        mu_y=mu_y,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        rho=rho_hat,
        se=se_hat,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        method="Proposed MLE method",
    )


def _validate_inputs(
    n: np.ndarray,
    xbar: np.ndarray,
    ybar: np.ndarray,
    s2x: np.ndarray | None,
    s2y: np.ndarray | None,
    method: str,
    ci_method: str,
) -> None:
    """Validate continuous–continuous summary inputs."""
    if method not in {"proposed", "weighted"}:
        raise ValueError(
            "method must be either 'proposed' or 'weighted'."
        )

    if ci_method not in {"none", "normal", "lr"}:
        raise ValueError(
            "ci_method must be one of 'none', 'normal', or 'lr'."
        )

    if n.ndim != 1 or xbar.ndim != 1 or ybar.ndim != 1:
        raise ValueError(
            "n, xbar, and ybar must be one-dimensional vectors."
        )

    if not (len(n) == len(xbar) == len(ybar)):
        raise ValueError(
            "n, xbar, and ybar must have equal lengths."
        )

    if len(n) < 2:
        raise ValueError("At least two studies are required.")

    if not (
        np.all(np.isfinite(n))
        and np.all(np.isfinite(xbar))
        and np.all(np.isfinite(ybar))
    ):
        raise ValueError(
            "Inputs must not contain missing or infinite values."
        )

    if np.any(n <= 0):
        raise ValueError("All sample sizes must be positive.")

    if not np.all(np.equal(n, np.floor(n))):
        raise ValueError("Sample sizes must be integers.")

    if method == "proposed":
        if s2x is None or s2y is None:
            raise ValueError(
                "s2x and s2y are required for the proposed method."
            )

        if len(s2x) != len(n) or len(s2y) != len(n):
            raise ValueError(
                "s2x and s2y must have the same length as n."
            )

        if not (
            np.all(np.isfinite(s2x))
            and np.all(np.isfinite(s2y))
        ):
            raise ValueError(
                "Sample variances must be finite."
            )

        if np.any(s2x <= 0) or np.any(s2y <= 0):
            raise ValueError(
                "Sample variances must be positive."
            )

        if np.any(n < 4):
            raise ValueError(
                "The proposed method requires each sample size "
                "to be at least 4."
            )


def _log_modified_bessel_first_kind(
    order: float,
    value: float,
) -> float:
    """Calculate log(I_order(value)) with scaling for stability."""
    if value <= 1e-12:
        return float(
            order * np.log(value / 2.0)
            - gammaln(order + 1.0)
        )

    scaled_value = ive(order, value)

    if np.isfinite(scaled_value) and scaled_value > 0:
        return float(np.log(scaled_value) + abs(value))

    kappa = float(np.sqrt(order**2 + value**2))
    eta = float(
        kappa
        + order * np.log(value / (order + kappa))
    )
    log_prefactor = float(
        -0.5 * np.log(2.0 * np.pi * kappa)
    )

    return eta + log_prefactor


def _bessel_likelihood_component(
    a: float,
    b: float,
) -> float:
    """Evaluate the Bessel-related likelihood term stably."""
    order = a + 0.5

    if b <= 1e-10:
        return float(
            0.5 * np.log(np.pi)
            + gammaln(a + 1.0)
            - gammaln(order + 1.0)
        )

    log_bessel = _log_modified_bessel_first_kind(
        order=order,
        value=b,
    )

    return float(
        0.5 * np.log(np.pi)
        + order * (np.log(2.0) - np.log(b))
        + gammaln(a + 1.0)
        + log_bessel
    )


def _log_likelihood(
    rho: float,
    n: np.ndarray,
    xbar: np.ndarray,
    ybar: np.ndarray,
    s2x: np.ndarray,
    s2y: np.ndarray,
    mu_x: float,
    mu_y: float,
    sigma_x: float,
    sigma_y: float,
) -> float:
    """Evaluate the proposed marginal log-likelihood."""
    if not -0.999 < rho < 0.999:
        return float("-inf")

    one_minus_rho_squared = 1.0 - rho**2

    if one_minus_rho_squared <= 0:
        return float("-inf")

    total = 0.0

    for index in range(len(n)):
        n_i = float(n[index])
        s2x_i = float(s2x[index])
        s2y_i = float(s2y[index])

        x_difference = float(xbar[index] - mu_x)
        y_difference = float(ybar[index] - mu_y)

        a = (n_i - 4.0) / 2.0

        b = abs(
            (n_i - 1.0)
            * rho
            * np.sqrt(s2x_i)
            * np.sqrt(s2y_i)
            / (
                one_minus_rho_squared
                * sigma_x
                * sigma_y
            )
        )

        quadratic_component = (
            x_difference**2 / sigma_x**2
            - (
                2.0
                * rho
                * x_difference
                * y_difference
                / (sigma_x * sigma_y)
            )
            + y_difference**2 / sigma_y**2
        )

        study_log_likelihood = (
            np.log(
                n_i
                / (
                    2.0
                    * np.pi
                    * sigma_x
                    * sigma_y
                    * np.sqrt(one_minus_rho_squared)
                )
            )
            - (
                n_i
                / (2.0 * one_minus_rho_squared)
                * quadratic_component
            )
            + ((n_i - 4.0) / 2.0)
            * np.log(s2x_i * s2y_i)
            - (
                (n_i - 1.0)
                / (2.0 * one_minus_rho_squared)
                * (
                    s2x_i / sigma_x**2
                    + s2y_i / sigma_y**2
                )
            )
            - (n_i - 1.0)
            * np.log(
                (
                    2.0
                    * sigma_x
                    * sigma_y
                    * np.sqrt(one_minus_rho_squared)
                )
                / (n_i - 1.0)
            )
            - (
                0.5 * np.log(np.pi)
                + gammaln((n_i - 1.0) / 2.0)
                + gammaln(n_i / 2.0 - 1.0)
            )
            + _bessel_likelihood_component(a=a, b=b)
        )

        if not np.isfinite(study_log_likelihood):
            return float("-inf")

        total += float(study_log_likelihood)

    return total


def _estimate_standard_error(
    log_likelihood,
    rho_hat: float,
) -> float | None:
    """Estimate the standard error from numerical likelihood curvature."""
    distance_to_boundary = min(
        rho_hat + 0.99,
        0.99 - rho_hat,
    )

    step = min(1e-4, distance_to_boundary / 4.0)

    if step <= 1e-8:
        return None

    second_derivative = (
        log_likelihood(rho_hat + step)
        - 2.0 * log_likelihood(rho_hat)
        + log_likelihood(rho_hat - step)
    ) / step**2

    if not np.isfinite(second_derivative) or second_derivative >= 0:
        return None

    variance = -1.0 / second_derivative

    if not np.isfinite(variance) or variance <= 0:
        return None

    return float(np.sqrt(variance))


def _likelihood_ratio_interval(
    log_likelihood,
    rho_hat: float,
    maximum_log_likelihood: float,
) -> tuple[float, float]:
    """Construct the connected 95% LR interval containing the MLE."""
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
                xtol=1e-9,
            )
        )

    left_grid = np.linspace(-0.99, rho_hat, 800)
    right_grid = np.linspace(rho_hat, 0.99, 800)

    ci_lower = find_crossing(
        left_grid,
        prefer_last=True,
    )
    ci_upper = find_crossing(
        right_grid,
        prefer_last=False,
    )

    if ci_lower is None:
        ci_lower = -0.99

    if ci_upper is None:
        ci_upper = 0.99

    return float(ci_lower), float(ci_upper)
