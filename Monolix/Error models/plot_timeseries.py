import math
import random
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt


RealToReal = Callable[[float], float]
ODEModel = Callable[[float, float, list[float]], list[float]]
ErrorModel = Literal["combined1", "combined2"]
ObsFormat = Literal["exact", "log10"]

@dataclass
class Distribution:
    """
    Contains properties `forward` and `inv` such that for a given value X,
    `forward(X)` transforms it into a space where it is normally distributed,
    and `inv` is the inverse transform.
    
    This mimics the role of h in the Monolix individual model.
    """
    forward: RealToReal
    inv: RealToReal

distributions = {
    "normal": Distribution(forward=lambda x: x, inv=lambda x: x),
    "lognormal": Distribution(forward=math.log, inv=math.exp),
}

def sample_parameters(
    pop_params: list[float],
    distributions: list[Distribution],
    pop_stds: list[float],
    n_samples: int,
    seed=42,
) -> list[list[float]]:
    """
    Generate n samples of parameters based on population parameters,
    a distribution type to sample the parameters from, and standard deviations
    of the parameters after transforming them to be normally distributed.
    """
    assert len(pop_params) == len(pop_stds) == len(distributions)
    random.seed(seed)
    return [
        [
            distribution.inv(random.gauss(distribution.forward(mean), std))
            for mean, std, distribution in zip(pop_params, pop_stds, distributions)
        ]
        for _ in range(n_samples)
    ]

def simulate_model(
    model: ODEModel,
    sampled_params_all: list[list[float]],
    initial_conditions_all: list[list[float]],
    obs_times_all: list[list[float]],
    error_models: ErrorModel,
    error_param_tuples: list[tuple[float, float, float]],
    error_dists: list[Distribution],
    err_c=1.0,
    seed=42,
) -> list[np.ndarray]:
    """
    Simulate observations for multiple parameter sets based on a structural model.
    Noise is added to the simulated (ODE) predictions based on the provided error model,
    error parameters, and error distributions.
    """
    assert (
        0 < len(sampled_params_all) == len(initial_conditions_all) == len(obs_times_all)
    )
    assert (
        len(initial_conditions_all[0])
        == len(error_models)
        == len(error_param_tuples)
        == len(error_dists)
    )
    random.seed(seed)
    # Each element of the list below corresponds to an individual.
    # Each returned array is transposed so that rows correspond to variables.
    observations_all: list[np.ndarray] = [
        odeint(model, initial_conditions, times, args=(params,)).T
        for initial_conditions, times, params in zip(
            initial_conditions_all, obs_times_all, sampled_params_all
        )
    ]
    for observations in observations_all:
        for row, error_model, (err_a, err_b, err_c), error_dist in zip(
            observations, error_models, error_param_tuples, error_dists
        ):
            for i, _ in enumerate(row):
                transformed_obs = error_dist.forward(row[i])
                error = 0.0
                match error_model:
                    case "combined1":
                        error = (err_a + err_b * (transformed_obs ** err_c)) * random.gauss(0, 1)
                    case "combined2":
                        error = math.sqrt(err_a**2 + (err_b * (transformed_obs ** err_c))**2) * random.gauss(0, 1)
                    case _:
                        raise ValueError(f"Invalid error model {error_model}")
                row[i] = error_dist.inv(transformed_obs + error)
    return observations_all

def generate_ground_truth(model: ODEModel, params: list[float], initial_conditions: list[float], obs_times: list[float]) -> np.ndarray:
    """
    Generate ground truth timeseries for a single patient using an ODE model.
    
    Returns:
        ground_truth: np.ndarray of shape (n_variables, n_time_points)
    """
    ground_truth = odeint(model, initial_conditions, obs_times, args=(params,)).T
    return ground_truth

def add_noise_to_timeseries(ground_truth: np.ndarray, error_model: ErrorModel, error_params: tuple[float, float, float], error_dist: Distribution, seed: int = None) -> np.ndarray:
    """
    Apply an error model to a ground truth timeseries (1D array) to generate a noisy timeseries.
    
    Parameters:
        ground_truth: 1D array-like of ground truth values.
        error_model: Either "combined1" or "combined2".
        error_params: Tuple (err_a, err_b, err_c) controlling noise.
        error_dist: Distribution object (with forward and inv methods).
        seed: Optional seed for reproducibility.
        
    Returns:
        noisy_series: np.ndarray of the same shape as ground_truth with noise applied.
    """
    if seed is not None:
        random.seed(seed)
    ground_truth = np.array(ground_truth)
    noisy_series = np.empty_like(ground_truth)
    err_a, err_b, err_c = error_params
    for i, obs in enumerate(ground_truth):
        transformed_obs = error_dist.forward(obs)
        gaussian_noise = random.gauss(0, 1)
        if error_model == "combined1":
            noise = (err_a + err_b * (transformed_obs ** err_c)) * gaussian_noise
        elif error_model == "combined2":
            noise = math.sqrt(err_a**2 + (err_b * (transformed_obs ** err_c))**2) * gaussian_noise
        else:
            raise ValueError(f"Unsupported error model: {error_model}")
        noisy_series[i] = error_dist.inv(transformed_obs + noise)
    return noisy_series

def generate_noisy_trials(ground_truth: np.ndarray, error_model: ErrorModel, error_params: tuple[float, float, float], error_dist: Distribution, num_trials: int, base_seed: int = 42) -> list[np.ndarray]:
    """
    Generate multiple noisy trials from a fixed ground truth timeseries.
    
    Parameters:
        ground_truth: 1D array-like of the ground truth timeseries.
        error_model: Either "combined1" or "combined2".
        error_params: Tuple (err_a, err_b, err_c).
        error_dist: Distribution object.
        num_trials: Number of noisy realizations to generate.
        base_seed: Base seed value (each trial uses base_seed + trial index).
        
    Returns:
        List of noisy timeseries (each is a np.ndarray).
    """
    trials = []
    for i in range(num_trials):
        trial = add_noise_to_timeseries(ground_truth, error_model, error_params, error_dist, seed=base_seed + i)
        trials.append(trial)
    return trials

def plot_ground_truth_with_noisy_overlays(ax, times: list[float], ground_truth: np.ndarray, noisy_trials: list[np.ndarray], noisy_labels: list[str] = None, ground_truth_label: str = "Ground Truth"):
    """
    Plot a ground truth timeseries with several noisy overlays on a given Axes.
    
    Parameters:
        ax: Matplotlib Axes object.
        times: 1D array-like of time points.
        ground_truth: 1D array-like of ground truth values.
        noisy_trials: List of 1D array-like noisy timeseries.
        noisy_labels: Optional list of labels for the noisy trials.
        ground_truth_label: Label for the ground truth timeseries.
    """
    ax.plot(times, ground_truth, label=ground_truth_label, linewidth=2)
    for i, trial in enumerate(noisy_trials):
        label = noisy_labels[i] if noisy_labels and i < len(noisy_labels) else f"Noisy Trial {i+1}"
        ax.plot(times, trial, linestyle='--', label=label)
    ax.set_xlabel("Time")
    ax.set_ylabel("Measurement")
    ax.set_title("Ground Truth vs Noisy Data Overlays")
    ax.legend()
    ax.grid(True)

# Example Usage
if __name__ == '__main__':
    # Define a simple ODE model (exponential decay)
    # This is where we would define the ODE model for the CAR-T cell model and use it
    # As a sanity check I made a temporary simple decay model
    def simple_decay(y, t, params):
        # y: state variable; params[0] is the decay rate.
        return [-params[0] * y[0]]
    
    # Parameters for the ODE model
    decay_rate = 0.5
    params = [decay_rate]
    initial_conditions = [1.0]
    obs_times = np.linspace(0, 10, 100)
    
    # Generate ground truth using the ODE model
    ground_truth_matrix = generate_ground_truth(simple_decay, params, initial_conditions, obs_times)
    # For a single-variable model, extract the first row
    ground_truth = ground_truth_matrix[0]
    
    # Define an error distribution using the "normal" distribution
    error_dist = distributions["normal"]
    
    # Set error model parameters and type
    error_params = (0.1, 0.2, 1.0)
    error_model = "combined1"
    
    # Generate multiple noisy trials from the fixed ground truth
    num_trials = 5
    noisy_trials = generate_noisy_trials(ground_truth, error_model, error_params, error_dist, num_trials)
    
    # Plot the ground truth and noisy trials
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_ground_truth_with_noisy_overlays(ax, obs_times, ground_truth, noisy_trials)
    plt.show()
