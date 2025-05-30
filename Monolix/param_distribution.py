# """
# Utility functions for generating and plotting synthetic data
# based on a nonlinear mixed-effects model for CAR-T cell therapy.
# """

# import math
# import random
# from dataclasses import dataclass
# from typing import Callable, Literal

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import odeint

# # =============================================================================
# # TYPE DEFINITIONS AND CORE DATACLASSES
# # =============================================================================

# type RealToReal = Callable[[float], float]
# type ODEModel = Callable[[list[float], float, list[float]], list[float]]
# type ErrorModel = Literal["combined1", "combined2"]

# @dataclass
# class Distribution:
#     """
#     A class to handle different statistical distributions by providing
#     a forward transformation (to a normal distribution) and its inverse.
#     This allows for sampling from various distributions using a standard
#     Gaussian random number generator.
#     """
#     forward: RealToReal
#     inv: RealToReal

# distributions = {
#     "normal": Distribution(forward=lambda x: x, inv=lambda x: x),
#     "lognormal": Distribution(forward=math.log, inv=math.exp),
# }

# # =============================================================================
# # ODE MODEL DEFINITION
# # =============================================================================

# def cart_tumor_model(y: list[float], t: float, params: list[float]) -> list[float]:
#     """
#     A Lotka-Volterra type model for CAR-T and Tumor cell interactions.

#     Args:
#         y (list[float]): A list containing the current population sizes
#                          of [Tumor cells, CAR-T cells].
#         t (float): The current time point (unused in this model but required by odeint).
#         params (list[float]): A list of parameters for the model:
#                               [r, K, a, b, e].
#                               r: tumor growth rate
#                               K: tumor carrying capacity
#                               a: tumor killing rate by CAR-T cells
#                               b: CAR-T cell expansion rate upon tumor contact
#                               e: CAR-T cell exhaustion/death rate

#     Returns:
#         list[float]: The list of derivatives [dT/dt, dC/dt] for tumor and CAR-T cells.
#     """
#     T, C = y
#     r, K, a, b, e = params
#     dT_dt = r * T * (1 - T / K) - a * T * C
#     dC_dt = b * T * C - e * C
#     return [dT_dt, dC_dt]

# # =============================================================================
# # DATA GENERATION PIPELINE
# # =============================================================================

# def sampler(
#     n_indivs: int,
#     pop_params: list[float],
#     param_dists: list[Distribution],
#     param_stds: list[float],
#     seed: int,
# ) -> list[list[float]]:
#     """
#     Samples parameter sets for a number of individuals based on population averages.

#     Args:
#         n_indivs (int): The number of individuals (parameter sets) to sample.
#         pop_params (list[float]): A list of the mean values for each parameter.
#         param_dists (list[Distribution]): A list of Distribution objects for each parameter.
#         param_stds (list[float]): The standard deviations for the *transformed* parameters.
#         seed (int): A random seed for reproducibility.

#     Returns:
#         list[list[float]]: A list of parameter lists, one for each individual.
#     """
#     assert len(pop_params) == len(param_dists) == len(param_stds)
#     random.seed(seed)
#     # Generate a list of parameter sets, one for each individual
#     return [
#         [
#             dist.inv(random.gauss(dist.forward(mean), std))
#             for mean, std, dist in zip(pop_params, param_stds, param_dists)
#         ]
#         for _ in range(n_indivs)
#     ]

# def generate_ground_truths(
#     model: ODEModel,
#     sampled_params: list[list[float]],
#     initial_conditions: list[list[float]],
#     observation_times: list[list[float]],
# ) -> list[np.ndarray]:
#     """
#     Generates ground truth trajectories by solving the ODE model for each parameter set.

#     Args:
#         model (ODEModel): The ordinary differential equation model to solve.
#         sampled_params (list[list[float]]): A list of parameter sets.
#         initial_conditions (list[list[float]]): A list of initial conditions for each individual.
#         observation_times (list[list[float]]): A list of time points for each individual.

#     Returns:
#         list[np.ndarray]: A list of ground truth trajectories. Each trajectory is a
#                           NumPy array of shape (n_variables, n_time_points).
#     """
#     return [
#         odeint(model, ic, times, args=(params,)).T
#         for ic, times, params in zip(initial_conditions, observation_times, sampled_params)
#     ]

# def add_noise(
#     ground_truths: list[np.ndarray],
#     error_params: list[tuple[float, float]],
#     seed: int,
# ) -> list[np.ndarray]:
#     """
#     Adds noise to ground truth trajectories to simulate measurement error.

#     This function uses a "combined1" error model: error = a + b * f(y).
#     The error is assumed to be normally distributed around the true value.

#     Args:
#         ground_truths (list[np.ndarray]): The clean model output.
#         error_params (list[tuple[float, float]]): A list of tuples (a, b) for each
#                                                    observed variable, where 'a' is the
#                                                    constant error and 'b' is the
#                                                    proportional error.
#         seed (int): A random seed for reproducibility.

#     Returns:
#         list[np.ndarray]: A list of noisy observation trajectories.
#     """
#     random.seed(seed)
#     noisy_observations = [gt.copy() for gt in ground_truths]

#     for individual_trajectory in noisy_observations:
#         # Each row in the trajectory corresponds to an observed variable (T or C)
#         for i, (err_a, err_b) in enumerate(error_params):
#             variable_trajectory = individual_trajectory[i]
#             # Add noise to each observation point for that variable
#             for j, value in enumerate(variable_trajectory):
#                 # Proportional error term
#                 proportional_error = err_b * value
#                 # Combine constant and proportional error to get the standard deviation
#                 error_std_dev = err_a + proportional_error
#                 # Sample noise from a normal distribution and add it to the true value
#                 noise = random.gauss(0, error_std_dev)
#                 variable_trajectory[j] += noise
#                 # Ensure biological values are not negative
#                 if variable_trajectory[j] < 0:
#                      variable_trajectory[j] = 0

#     return noisy_observations

# # =============================================================================
# # EVALUATION AND PLOTTING
# # =============================================================================

# def evaluate_trajectories(
#     ground_truths: list[np.ndarray],
#     noisy_observations: list[np.ndarray]
# ) -> None:
#     """
#     Calculates and prints the Root Mean Squared Error (RMSE) and Median Relative Error (MRE)
#     between the ground truth and noisy observations for Tumor and CAR-T cells.

#     Args:
#         ground_truths (list[np.ndarray]): The clean model output.
#         noisy_observations (list[np.ndarray]): The data with added noise.
#     """
#     rmses_T, mres_T = [], []
#     rmses_C, mres_C = [], []

#     for gt, noisy in zip(ground_truths, noisy_observations):
#         T_gt, C_gt = gt[0], gt[1]
#         T_noisy, C_noisy = noisy[0], noisy[1]

#         # Calculate RMSE
#         rmses_T.append(np.sqrt(np.mean((T_gt - T_noisy)**2)))
#         rmses_C.append(np.sqrt(np.mean((C_gt - C_noisy)**2)))

#         # Calculate Median Relative Error, avoiding division by zero
#         safe_T_gt = np.where(T_gt == 0, 1e-9, T_gt)
#         safe_C_gt = np.where(C_gt == 0, 1e-9, C_gt)
#         mres_T.append(np.median(np.abs((T_gt - T_noisy) / safe_T_gt)))
#         mres_C.append(np.median(np.abs((C_gt - C_noisy) / safe_C_gt)))

#     print("--- Evaluation (Noisy vs. Ground Truth) ---")
#     print(f"  Tumor Cells | Mean RMSE: {np.mean(rmses_T):.2e} | Mean MRE: {np.mean(mres_T):.2f}")
#     print(f"  CAR-T Cells | Mean RMSE: {np.mean(rmses_C):.2e} | Mean MRE: {np.mean(mres_C):.2f}")
#     print("------------------------------------------")


# def generate_and_plot_trajectories(
#     n_responders: int,
#     n_non_responders: int,
#     responder_params: dict,
#     non_responder_params: dict,
#     initial_conditions: list,
#     time_points: np.ndarray,
#     noise_params: list | None = None,
#     seed: int = 42
# ) -> None:
#     """
#     Orchestrates the full process: sampling, generation of ground truths,
#     optional noise addition, and plotting.

#     Args:
#         n_responders (int): Number of responding individuals.
#         n_non_responders (int): Number of non-responding individuals.
#         responder_params (dict): Parameter configuration for the "Responders" group.
#         non_responder_params (dict): Parameter configuration for the "Non-responders" group.
#         initial_conditions (list): Initial [Tumor, CAR-T] counts.
#         time_points (np.ndarray): Array of time points for the simulation.
#         noise_params (list | None): Error parameters for adding noise. If None, only
#                                     ground truths are plotted.
#         seed (int): The master random seed.
#     """
#     # --- 1. Sample Parameters for Both Groups ---
#     params_r = sampler(n_responders, seed=seed, **responder_params)
#     params_nr = sampler(n_non_responders, seed=seed + 1, **non_responder_params)

#     # --- 2. Define Initial Conditions & Observation Times ---
#     # For this simulation, all individuals start with the same conditions and time points
#     ics_r = [initial_conditions] * n_responders
#     ics_nr = [initial_conditions] * n_non_responders
#     times_r = [time_points] * n_responders
#     times_nr = [time_points] * n_non_responders

#     # --- 3. Generate Ground Truths ---
#     gt_r = generate_ground_truths(cart_tumor_model, params_r, ics_r, times_r)
#     gt_nr = generate_ground_truths(cart_tumor_model, params_nr, ics_nr, times_nr)

#     # --- 4. Add Noise (Optional) ---
#     if noise_params:
#         noisy_r = add_noise(gt_r, noise_params, seed=seed + 2)
#         noisy_nr = add_noise(gt_nr, noise_params, seed=seed + 3)
#         # Evaluate the effect of the added noise
#         evaluate_trajectories(gt_r + gt_nr, noisy_r + noisy_nr)
#         plot_data_r, plot_data_nr = noisy_r, noisy_nr
#         plot_title = "Noisy Observations"
#     else:
#         plot_data_r, plot_data_nr = gt_r, gt_nr
#         plot_title = "Ground Truth Trajectories"


#     # --- 5. Plot the Results ---
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
#     fig.suptitle(plot_title, fontsize=16)

#     # Plot Responders (Green)
#     for i, data in enumerate(plot_data_r):
#         label = "Responders" if i == 0 else ""
#         axes[0].plot(time_points, data[0], color="darkgreen", alpha=0.7, label=label)
#         axes[1].plot(time_points, data[1], color="darkgreen", alpha=0.7, label=label)

#     # Plot Non-responders (Orange)
#     for i, data in enumerate(plot_data_nr):
#         label = "Non-responders" if i == 0 else ""
#         axes[0].plot(time_points, data[0], color="darkorange", alpha=0.7, label=label)
#         axes[1].plot(time_points, data[1], color="darkorange", alpha=0.7, label=label)

#     axes[0].set_title("Tumor Cells")
#     axes[0].set_xlabel("Time post infusion (days)")
#     axes[0].set_ylabel("Cell Count")
#     axes[0].set_yscale('log')
#     axes[0].legend()

#     axes[1].set_title("CAR T Cells")
#     axes[1].set_xlabel("Time post infusion (days)")
#     axes[1].set_ylabel("Cell Count")
#     axes[1].set_yscale('log')
#     axes[1].legend()

#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.show()


# # =============================================================================
# # MAIN EXECUTION BLOCK
# # =============================================================================
# if __name__ == "__main__":
#     # --- Simulation Configuration ---

#     # Define population parameters [r, K, a, b, e] and their distributions
#     # Responders have high kill rate (a) and high CAR-T expansion (b)
#     RESPONDER_PARAMS = {
#         "pop_params": [0.5, 1e8, 5e-7, 8e-7, 0.2],
#         "param_dists": [distributions["lognormal"]] * 5,
#         "param_stds": [0.2, 0.1, 0.3, 0.3, 0.2]
#     }

#     # Non-responders have low kill rate (a) and low CAR-T expansion (b)
#     NON_RESPONDER_PARAMS = {
#         "pop_params": [0.5, 1e8, 1e-9, 1e-8, 0.4],
#         "param_dists": [distributions["lognormal"]] * 5,
#         "param_stds": [0.2, 0.1, 0.3, 0.3, 0.2]
#     }

#     # Define error parameters [a, b] for [Tumor, CAR-T]
#     # 'a' = constant error, 'b' = proportional error
#     NOISE_PARAMS = [
#         (1e4, 0.15),  # Error for Tumor cells
#         (1e3, 0.15)   # Error for CAR-T cells
#     ]

#     # Initial conditions [T_initial, C_initial] and time points
#     INITIAL_CONDITIONS = [1e6, 2e5]
#     TIME_POINTS = np.linspace(0, 100, 50)

#     # --- Run and Plot ---

#     # First, plot the ground truths without noise
#     print("Generating plot for ground truth trajectories...")
#     generate_and_plot_trajectories(
#         n_responders=5,
#         n_non_responders=3,
#         responder_params=RESPONDER_PARAMS,
#         non_responder_params=NON_RESPONDER_PARAMS,
#         initial_conditions=INITIAL_CONDITIONS,
#         time_points=TIME_POINTS,
#         noise_params=None, # No noise for this plot
#         seed=42
#     )

#     # Second, plot the trajectories with added noise
#     print("\nGenerating plot for noisy observations...")
#     generate_and_plot_trajectories(
#         n_responders=5,
#         n_non_responders=3,
#         responder_params=RESPONDER_PARAMS,
#         non_responder_params=NON_RESPONDER_PARAMS,
#         initial_conditions=INITIAL_CONDITIONS,
#         time_points=TIME_POINTS,
#         noise_params=NOISE_PARAMS, # Add noise for this plot
#         seed=42
#     )

# Claude Below - going to polish this entire file up.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random
from typing import List, Tuple, Callable
import pandas as pd

# Define the CAR-T cell therapy ODE model
def cart_model(y, t, params):
    """
    CAR-T cell therapy model with tumor cells and CAR-T cells
    y[0] = Tumor cells
    y[1] = CAR-T cells
    
    Parameters:
    - params[0]: tumor growth rate (r)
    - params[1]: carrying capacity (K) 
    - params[2]: CAR-T killing efficacy (k)
    - params[3]: CAR-T expansion rate (alpha)
    - params[4]: CAR-T decay rate (delta)
    """
    T, C = y  # Tumor cells, CAR-T cells
    r, K, k, alpha, delta = params
    
    # Tumor dynamics: logistic growth - CAR-T killing
    dT_dt = r * T * (1 - T/K) - k * T * C
    
    # CAR-T dynamics: expansion from tumor killing - decay
    dC_dt = alpha * k * T * C - delta * C
    
    return [dT_dt, dC_dt]

def normal_sampler(param_distributions: List[Tuple[float, float]]) -> Callable:
    """
    Creates a sampler function for normally distributed parameters
    
    Args:
        param_distributions: List of (mean, std) tuples for each parameter
    
    Returns:
        Function that samples parameters given n_indivs and seed
    """
    def sampler(n_indivs: int, seed: int = 42) -> List[List[float]]:
        """
        Sample parameters for n individuals
        
        Args:
            n_indivs: Number of individuals to sample for
            seed: Random seed for reproducibility
            
        Returns:
            List of parameter lists, one for each individual
        """
        np.random.seed(seed)
        random.seed(seed)
        
        sampled_params = []
        for i in range(n_indivs):
            individual_params = []
            for mean, std in param_distributions:
                param_value = np.random.normal(mean, std)
                # Ensure positive parameters (important for biological models)
                param_value = max(param_value, 0.001)
                individual_params.append(param_value)
            sampled_params.append(individual_params)
        
        return sampled_params
    
    return sampler

def observation_time_scheduler(n_indivs: int, max_time: float = 30, n_points: int = 50, seed: int = 42) -> List[List[float]]:
    """
    Generate observation times for each individual
    
    Args:
        n_indivs: Number of individuals
        max_time: Maximum observation time
        n_points: Number of observation points
        seed: Random seed
        
    Returns:
        List of time arrays for each individual
    """
    np.random.seed(seed)
    
    observation_times = []
    for i in range(n_indivs):
        # Base time points with some individual variation
        base_times = np.linspace(0, max_time, n_points)
        # Add some jitter to make it more realistic
        jitter = np.random.normal(0, 0.5, n_points)
        times = np.maximum(base_times + jitter, 0)  # Ensure non-negative times
        times = np.sort(times)  # Keep chronological order
        observation_times.append(times.tolist())
    
    return observation_times

def generate_ground_truths(
    sampled_params: List[List[float]], 
    model: Callable,
    observation_times: List[List[float]],
    initial_conditions: List[List[float]] = None
) -> List[np.ndarray]:
    """
    Generate ground truth trajectories for multiple individuals
    
    Args:
        sampled_params: List of parameter sets for each individual
        model: ODE model function
        observation_times: List of time points for each individual
        initial_conditions: Initial conditions for each individual
        
    Returns:
        List of trajectory arrays for each individual
    """
    n_indivs = len(sampled_params)
    
    # Default initial conditions if not provided
    if initial_conditions is None:
        initial_conditions = [[100.0, 10.0] for _ in range(n_indivs)]  # [Tumor, CAR-T]
    
    ground_truths = []
    
    for i in range(n_indivs):
        params = sampled_params[i]
        times = observation_times[i]
        init_cond = initial_conditions[i]
        
        # Solve ODE
        solution = odeint(model, init_cond, times, args=(params,))
        ground_truths.append(solution.T)  # Transpose so each row is a variable
    
    return ground_truths

def add_noise(
    ground_truths: List[np.ndarray], 
    noise_level: float = 0.1, 
    seed: int = 42
) -> List[np.ndarray]:
    """
    Add noise to ground truth observations
    
    Args:
        ground_truths: List of ground truth trajectories
        noise_level: Standard deviation of relative noise
        seed: Random seed
        
    Returns:
        List of noisy observations
    """
    np.random.seed(seed)
    
    noisy_observations = []
    
    for ground_truth in ground_truths:
        noisy_obs = ground_truth.copy()
        
        # Add proportional noise to each observation
        for i in range(ground_truth.shape[0]):  # For each variable
            for j in range(ground_truth.shape[1]):  # For each time point
                noise = np.random.normal(0, noise_level * ground_truth[i, j])
                noisy_obs[i, j] = max(ground_truth[i, j] + noise, 0)  # Ensure non-negative
        
        noisy_observations.append(noisy_obs)
    
    return noisy_observations

def plot_trajectories(
    n_indivs: int,
    sampler_func: Callable,
    seed: int = 42,
    add_noise_flag: bool = True,
    noise_level: float = 0.1
):
    """
    Complete pipeline: sample parameters, generate trajectories, and plot
    
    Args:
        n_indivs: Number of individuals
        sampler_func: Function to sample parameters
        seed: Random seed
        add_noise_flag: Whether to add noise and plot noisy observations
        noise_level: Level of noise to add
    """
    # Sample parameters
    sampled_params = sampler_func(n_indivs, seed)
    
    # Schedule observation times
    observation_times = observation_time_scheduler(n_indivs, max_time=30, seed=seed)
    
    # Generate ground truths
    ground_truths = generate_ground_truths(sampled_params, cart_model, observation_times)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Determine responders vs non-responders based on final tumor burden
    responder_threshold = 50  # Tumor cells below this at end = responder
    
    for i, (ground_truth, times) in enumerate(zip(ground_truths, observation_times)):
        tumor_trajectory = ground_truth[0]
        cart_trajectory = ground_truth[1]
        
        # Classify as responder or non-responder
        final_tumor = tumor_trajectory[-1]
        is_responder = final_tumor < responder_threshold
        
        color = 'green' if is_responder else 'orange'
        alpha = 0.7
        
        # Plot tumor cells
        ax1.plot(times, tumor_trajectory, color=color, alpha=alpha, linewidth=2)
        
        # Plot CAR-T cells
        ax2.plot(times, cart_trajectory, color=color, alpha=alpha, linewidth=2)
    
    # Add noise and plot if requested
    if add_noise_flag:
        noisy_observations = add_noise(ground_truths, noise_level, seed)
        
        for i, (noisy_obs, times) in enumerate(zip(noisy_observations, observation_times)):
            tumor_noisy = noisy_obs[0]
            cart_noisy = noisy_obs[1]
            
            # Classify based on ground truth
            final_tumor = ground_truths[i][0][-1]
            is_responder = final_tumor < responder_threshold
            
            color = 'darkgreen' if is_responder else 'darkorange'
            
            # Plot noisy observations as scatter points
            ax1.scatter(times[::5], tumor_noisy[::5], color=color, alpha=0.6, s=20)
            ax2.scatter(times[::5], cart_noisy[::5], color=color, alpha=0.6, s=20)
    
    # Customize plots
    ax1.set_xlabel('Time post infusion')
    ax1.set_ylabel('Tumor cells')
    ax1.set_title('Tumor Cell Dynamics')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Time post infusion')
    ax2.set_ylabel('CAR-T cells')
    ax2.set_title('CAR-T Cell Dynamics')
    ax2.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Responders (Ground Truth)'),
        Line2D([0], [0], color='orange', lw=2, label='Non-responders (Ground Truth)')
    ]
    if add_noise_flag:
        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen', 
                   markersize=8, label='Responders (Noisy Obs)', linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange', 
                   markersize=8, label='Non-responders (Noisy Obs)', linestyle='None')
        ])
    
    ax1.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return sampled_params, ground_truths, observation_times, noisy_observations if add_noise_flag else None

# Example usage and parameter definitions
def create_cart_model_sampler():
    """
    Create a sampler for CAR-T model parameters with realistic distributions
    """
    # Parameter distributions: (mean, std)
    # [tumor_growth_rate, carrying_capacity, killing_efficacy, expansion_rate, decay_rate]
    param_distributions = [
        (0.1, 0.02),    # tumor growth rate
        (1000, 200),    # carrying capacity  
        (0.01, 0.005),  # CAR-T killing efficacy
        (0.5, 0.1),     # CAR-T expansion rate
        (0.05, 0.01)    # CAR-T decay rate
    ]
    
    return normal_sampler(param_distributions)

# Example execution
if __name__ == "__main__":
    # Create sampler
    sampler = create_cart_model_sampler()
    
    # Generate and plot trajectories
    results = plot_trajectories(
        n_indivs=8, 
        sampler_func=sampler, 
        seed=42,
        add_noise_flag=True,
        noise_level=0.15
    )
    
    print(f"Generated trajectories for {len(results[0])} individuals")
    print(f"Parameter example for individual 1: {results[0][0]}")