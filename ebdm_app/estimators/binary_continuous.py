from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


EstimationMethod = Literal["gmm", "naive"]


@dataclass(frozen=True)
class BinaryContinuousResult:
    """Results returned by the binary–continuous estimator."""

    mu1_hat: float
    mu0_hat: float
    sigma1_hat: float
    sigma0_hat: float

    se_mu1: float | None
    se_mu0: float | None
    se_sigma1: float | None
    se_sigma0: float | None

    ci_mu1: tuple[float, float] | None
    ci_mu0: tuple[float, float] | None
    ci_sigma1: tuple[float, float] | None
    ci_sigma0: tuple[float, float] | None

    method: str
    converged: bool
    objective_value: float | None


def estimate_binary_continuous(
    ni: list[int] | np.ndarray,
    xbar: list[float] | np.ndarray,
    mi: list[int] | np.ndarray,
    s2: list[float] | np.ndarray | None = None,
    method: EstimationMethod = "gmm",
) -> BinaryContinuousResult:
    """
    Estimate group-specific normal distributions from aggregate summaries.

    Parameters
    ----------
    ni
        Sample size for each study.
    xbar
        Overall continuous-outcome mean for each study.
    mi
        Number of observations in binary group Y = 1.
    s2
        Overall sample variance for each study. Required for GMM.
    method
        Either "gmm" or "naive".
    """
    n = np.asarray(ni, dtype=float)
    mean_values = np.asarray(xbar, dtype=float)
    group1_counts = np.asarray(mi, dtype=float)

    variance_values = (
        None
        if s2 is None
        else np.asarray(s2, dtype=float)
    )

    _validate_inputs(
        n=n,
        xbar=mean_values,
        mi=group1_counts,
        s2=variance_values,
        method=method,
    )

    naive_result = _fit_naive_estimator(
        n=n,
        xbar=mean_values,
        mi=group1_counts,
        s2=variance_values,
    )

    (
        mu1_naive,
        mu0_naive,
        sigma1_naive,
        sigma0_naive,
        naive_converged,
        naive_objective,
    ) = naive_result

    if method == "naive":
        return BinaryContinuousResult(
            mu1_hat=mu1_naive,
            mu0_hat=mu0_naive,
            sigma1_hat=sigma1_naive,
            sigma0_hat=sigma0_naive,
            se_mu1=None,
            se_mu0=None,
            se_sigma1=None,
            se_sigma0=None,
            ci_mu1=None,
            ci_mu0=None,
            ci_sigma1=None,
            ci_sigma0=None,
            method="Naive estimator",
            converged=naive_converged,
            objective_value=naive_objective,
        )

    assert variance_values is not None

    valid_mask = (
        (group1_counts > 1)
        & (group1_counts < n - 1)
    )

    valid_n = n[valid_mask]
    valid_xbar = mean_values[valid_mask]
    valid_mi = group1_counts[valid_mask]
    valid_s2 = variance_values[valid_mask]

    if len(valid_n) < 5:
        raise ValueError(
            "The GMM method requires at least five studies with "
            "1 < mi < ni - 1."
        )

    pooled_sd = float(
        np.sqrt(max(np.mean(valid_s2), 1e-6))
    )

    initial_sigma1 = max(
        sigma1_naive,
        0.25 * pooled_sd,
        1e-3,
    )
    initial_sigma0 = max(
        sigma0_naive,
        0.25 * pooled_sd,
        1e-3,
    )

    initial_theta = np.asarray(
        [
            mu1_naive,
            mu0_naive,
            initial_sigma1**2,
            initial_sigma0**2,
        ],
        dtype=float,
    )

    def moment_matrix(theta: np.ndarray) -> np.ndarray:
        return _compute_moment_matrix(
            theta=theta,
            n=valid_n,
            xbar=valid_xbar,
            mi=valid_mi,
            s2=valid_s2,
        )

    def objective(theta: np.ndarray) -> float:
        if (
            not np.all(np.isfinite(theta))
            or theta[2] <= 0
            or theta[3] <= 0
        ):
            return 1e100

        moments = moment_matrix(theta)

        if not np.all(np.isfinite(moments)):
            return 1e100

        scales = _moment_scales(moments)
        scaled_mean = np.mean(moments, axis=0) / scales

        return float(scaled_mean @ scaled_mean)

    optimization = minimize(
        fun=objective,
        x0=initial_theta,
        method="L-BFGS-B",
        bounds=[
            (None, None),
            (None, None),
            (1e-8, None),
            (1e-8, None),
        ],
        options={
            "maxiter": 3000,
            "ftol": 1e-12,
            "gtol": 1e-8,
        },
    )

    if not np.all(np.isfinite(optimization.x)):
        raise RuntimeError(
            "GMM optimization returned non-finite parameter estimates."
        )

    theta_hat = np.asarray(
        optimization.x,
        dtype=float,
    )

    mu1_hat = float(theta_hat[0])
    mu0_hat = float(theta_hat[1])
    sigma1_hat = float(np.sqrt(theta_hat[2]))
    sigma0_hat = float(np.sqrt(theta_hat[3]))

    standard_errors = _compute_gmm_standard_errors(
        theta_hat=theta_hat,
        moment_matrix_function=moment_matrix,
    )

    if standard_errors is None:
        se_mu1 = None
        se_mu0 = None
        se_sigma1 = None
        se_sigma0 = None
    else:
        (
            se_mu1,
            se_mu0,
            se_sigma1,
            se_sigma0,
        ) = standard_errors

    critical_value = float(norm.ppf(0.975))

    ci_mu1 = _normal_interval(
        estimate=mu1_hat,
        standard_error=se_mu1,
        critical_value=critical_value,
    )
    ci_mu0 = _normal_interval(
        estimate=mu0_hat,
        standard_error=se_mu0,
        critical_value=critical_value,
    )
    ci_sigma1 = _normal_interval(
        estimate=sigma1_hat,
        standard_error=se_sigma1,
        critical_value=critical_value,
        lower_limit=0.0,
    )
    ci_sigma0 = _normal_interval(
        estimate=sigma0_hat,
        standard_error=se_sigma0,
        critical_value=critical_value,
        lower_limit=0.0,
    )

    return BinaryContinuousResult(
        mu1_hat=mu1_hat,
        mu0_hat=mu0_hat,
        sigma1_hat=sigma1_hat,
        sigma0_hat=sigma0_hat,
        se_mu1=se_mu1,
        se_mu0=se_mu0,
        se_sigma1=se_sigma1,
        se_sigma0=se_sigma0,
        ci_mu1=ci_mu1,
        ci_mu0=ci_mu0,
        ci_sigma1=ci_sigma1,
        ci_sigma0=ci_sigma0,
        method="Scaled GMM estimator",
        converged=bool(optimization.success),
        objective_value=float(optimization.fun),
    )


def _validate_inputs(
    n: np.ndarray,
    xbar: np.ndarray,
    mi: np.ndarray,
    s2: np.ndarray | None,
    method: str,
) -> None:
    """Validate aggregate binary–continuous inputs."""
    if method not in {"gmm", "naive"}:
        raise ValueError(
            "method must be either 'gmm' or 'naive'."
        )

    if n.ndim != 1 or xbar.ndim != 1 or mi.ndim != 1:
        raise ValueError(
            "ni, xbar, and mi must be one-dimensional vectors."
        )

    if not (len(n) == len(xbar) == len(mi)):
        raise ValueError(
            "ni, xbar, and mi must have equal lengths."
        )

    if len(n) < 2:
        raise ValueError("At least two studies are required.")

    if not (
        np.all(np.isfinite(n))
        and np.all(np.isfinite(xbar))
        and np.all(np.isfinite(mi))
    ):
        raise ValueError(
            "Inputs must not contain missing or infinite values."
        )

    if np.any(n <= 0):
        raise ValueError("All sample sizes must be positive.")

    if not np.all(np.equal(n, np.floor(n))):
        raise ValueError("Sample sizes must be integers.")

    if not np.all(np.equal(mi, np.floor(mi))):
        raise ValueError("Group counts mi must be integers.")

    if np.any(mi < 0) or np.any(mi > n):
        raise ValueError(
            "Each mi must satisfy 0 ≤ mi ≤ ni."
        )

    proportions = mi / n

    if np.ptp(proportions) < 1e-8:
        raise ValueError(
            "The group proportions mi / ni must vary across studies."
        )

    if method == "gmm":
        if s2 is None:
            raise ValueError(
                "Sample variances s2 are required for GMM."
            )

        if s2.ndim != 1 or len(s2) != len(n):
            raise ValueError(
                "s2 must have the same length as ni."
            )

        if not np.all(np.isfinite(s2)):
            raise ValueError(
                "Sample variances must be finite."
            )

        if np.any(s2 <= 0):
            raise ValueError(
                "Sample variances must be positive."
            )


def _fit_naive_estimator(
    n: np.ndarray,
    xbar: np.ndarray,
    mi: np.ndarray,
    s2: np.ndarray | None,
) -> tuple[float, float, float, float, bool, float]:
    """Fit the likelihood estimator based on study-level means."""
    proportions = mi / n

    design_matrix = np.column_stack(
        [
            np.ones_like(proportions),
            proportions,
        ]
    )

    regression_weights = np.sqrt(n)

    coefficients = np.linalg.lstsq(
        design_matrix * regression_weights[:, None],
        xbar * regression_weights,
        rcond=None,
    )[0]

    mu0_initial = float(coefficients[0])
    mu1_initial = float(
        coefficients[0] + coefficients[1]
    )

    if s2 is None:
        initial_variance = float(
            np.var(xbar, ddof=1)
        )
    else:
        initial_variance = float(np.mean(s2))

    initial_sd = float(
        np.sqrt(max(initial_variance, 1e-4))
    )

    initial_parameters = np.asarray(
        [
            mu1_initial,
            mu0_initial,
            np.log(initial_sd),
            np.log(initial_sd),
        ],
        dtype=float,
    )

    def negative_log_likelihood(
        parameters: np.ndarray,
    ) -> float:
        mu1 = float(parameters[0])
        mu0 = float(parameters[1])
        sigma1 = float(np.exp(parameters[2]))
        sigma0 = float(np.exp(parameters[3]))

        expected_means = (
            proportions * mu1
            + (1.0 - proportions) * mu0
        )

        mean_variances = (
            mi / n**2 * sigma1**2
            + (n - mi) / n**2 * sigma0**2
        )

        if (
            not np.all(np.isfinite(mean_variances))
            or np.any(mean_variances <= 0)
        ):
            return 1e100

        residuals = xbar - expected_means

        return float(
            0.5
            * np.sum(
                np.log(2.0 * np.pi * mean_variances)
                + residuals**2 / mean_variances
            )
        )

    optimization = minimize(
        fun=negative_log_likelihood,
        x0=initial_parameters,
        method="L-BFGS-B",
        bounds=[
            (None, None),
            (None, None),
            (np.log(1e-4), np.log(1e4)),
            (np.log(1e-4), np.log(1e4)),
        ],
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
        },
    )

    if np.all(np.isfinite(optimization.x)):
        parameters = optimization.x
    else:
        parameters = initial_parameters

    return (
        float(parameters[0]),
        float(parameters[1]),
        float(np.exp(parameters[2])),
        float(np.exp(parameters[3])),
        bool(optimization.success),
        float(negative_log_likelihood(parameters)),
    )


def _compute_moment_matrix(
    theta: np.ndarray,
    n: np.ndarray,
    xbar: np.ndarray,
    mi: np.ndarray,
    s2: np.ndarray,
) -> np.ndarray:
    """Calculate the five raw GMM moment conditions by study."""
    mu1 = float(theta[0])
    mu0 = float(theta[1])
    sigma1_squared = float(theta[2])
    sigma0_squared = float(theta[3])

    proportions = mi / n

    expected_mean = (
        proportions * mu1
        + (1.0 - proportions) * mu0
    )

    g1 = xbar - expected_mean

    variance_of_mean = (
        sigma1_squared * mi / n**2
        + sigma0_squared * (n - mi) / n**2
    )

    g3 = (
        (xbar - expected_mean) ** 2
        - variance_of_mean
    )

    mean_difference = mu1 - mu0

    expected_sample_variance = (
        (
            mi * sigma1_squared
            + (n - mi) * sigma0_squared
        )
        / n
        + (
            mi
            * (n - mi)
            / (n * (n - 1.0))
            * mean_difference**2
        )
    )

    g2 = s2 - expected_sample_variance

    tau_squared = (
        sigma1_squared / mi
        + sigma0_squared / (n - mi)
    )

    variance_component_1 = (
        2.0
        * (mi - 1.0)
        * sigma1_squared**2
    )

    variance_component_2 = (
        2.0
        * (n - mi - 1.0)
        * sigma0_squared**2
    )

    variance_component_3 = (
        (mi * (n - mi) / n) ** 2
        * (
            2.0 * tau_squared**2
            + 4.0
            * tau_squared
            * mean_difference**2
        )
    )

    variance_of_sample_variance = (
        variance_component_1
        + variance_component_2
        + variance_component_3
    ) / (n - 1.0) ** 2

    g4 = (
        (s2 - expected_sample_variance) ** 2
        - variance_of_sample_variance
    )

    modeled_covariance = (
        2.0
        * mi
        * (n - mi)
        / (n**2 * (n - 1.0))
        * mean_difference
        * (
            sigma1_squared
            - sigma0_squared
        )
    )

    g5 = (
        (xbar - expected_mean)
        * (s2 - expected_sample_variance)
        - modeled_covariance
    )

    return np.column_stack(
        [
            g1,
            g2,
            g3,
            g4,
            g5,
        ]
    )


def _moment_scales(
    moment_matrix: np.ndarray,
) -> np.ndarray:
    """Calculate stable scaling factors for the five moments."""
    scales = np.std(
        moment_matrix,
        axis=0,
        ddof=1,
    )

    return np.where(
        np.isfinite(scales) & (scales > 1e-8),
        scales,
        1.0,
    )


def _compute_gmm_standard_errors(
    theta_hat: np.ndarray,
    moment_matrix_function,
) -> tuple[float, float, float, float] | None:
    """Calculate sandwich standard errors for the GMM estimates."""
    raw_moments = moment_matrix_function(theta_hat)
    scales = _moment_scales(raw_moments)

    scaled_moments = raw_moments / scales
    number_of_studies = scaled_moments.shape[0]

    if number_of_studies < 2:
        return None

    covariance_matrix = np.cov(
        scaled_moments,
        rowvar=False,
        bias=True,
    )

    covariance_matrix = np.atleast_2d(
        covariance_matrix
    )

    def scaled_mean_moments(
        theta: np.ndarray,
    ) -> np.ndarray:
        return (
            np.mean(
                moment_matrix_function(theta),
                axis=0,
            )
            / scales
        )

    number_of_parameters = len(theta_hat)
    number_of_moments = len(scales)

    jacobian = np.empty(
        (
            number_of_moments,
            number_of_parameters,
        ),
        dtype=float,
    )

    base_moments = scaled_mean_moments(theta_hat)

    for parameter_index in range(
        number_of_parameters
    ):
        step = (
            1e-5
            * max(
                1.0,
                abs(theta_hat[parameter_index]),
            )
        )

        upper_theta = theta_hat.copy()
        lower_theta = theta_hat.copy()

        upper_theta[parameter_index] += step
        lower_theta[parameter_index] -= step

        if (
            parameter_index >= 2
            and lower_theta[parameter_index] <= 1e-8
        ):
            jacobian[:, parameter_index] = (
                scaled_mean_moments(upper_theta)
                - base_moments
            ) / step
        else:
            jacobian[:, parameter_index] = (
                scaled_mean_moments(upper_theta)
                - scaled_mean_moments(lower_theta)
            ) / (2.0 * step)

    matrix_a = jacobian.T @ jacobian
    matrix_b = (
        jacobian.T
        @ covariance_matrix
        @ jacobian
    )

    inverse_a = np.linalg.pinv(
        matrix_a,
        rcond=1e-10,
    )

    covariance_theta = (
        inverse_a
        @ matrix_b
        @ inverse_a
        / number_of_studies
    )

    diagonal = np.diag(covariance_theta)

    if not np.all(np.isfinite(diagonal)):
        return None

    standard_errors_theta = np.sqrt(
        np.clip(diagonal, 0.0, None)
    )

    sigma1_hat = float(np.sqrt(theta_hat[2]))
    sigma0_hat = float(np.sqrt(theta_hat[3]))

    if sigma1_hat <= 0 or sigma0_hat <= 0:
        return None

    se_mu1 = float(standard_errors_theta[0])
    se_mu0 = float(standard_errors_theta[1])

    se_sigma1 = float(
        0.5
        * standard_errors_theta[2]
        / sigma1_hat
    )

    se_sigma0 = float(
        0.5
        * standard_errors_theta[3]
        / sigma0_hat
    )

    return (
        se_mu1,
        se_mu0,
        se_sigma1,
        se_sigma0,
    )


def _normal_interval(
    estimate: float,
    standard_error: float | None,
    critical_value: float,
    lower_limit: float | None = None,
) -> tuple[float, float] | None:
    """Construct a two-sided normal-approximation interval."""
    if (
        standard_error is None
        or not np.isfinite(standard_error)
    ):
        return None

    lower = (
        estimate
        - critical_value * standard_error
    )
    upper = (
        estimate
        + critical_value * standard_error
    )

    if lower_limit is not None:
        lower = max(lower_limit, lower)

    return float(lower), float(upper)
