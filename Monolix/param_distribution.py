import math
import random
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# =============================================================================
# TYPE DEFINITIONS AND CORE DATACLASSES
# =============================================================================

type RealToReal = Callable[[float], float]
type ODEModel = Callable[[list[float], float, list[float]], list[float]]
type ErrorModel = Literal["combined1", "combined2"] # Note: ErrorModel type is defined but not explicitly used in add_noise

@dataclass
class Distribution:
    """
    A class to handle different statistical distributions by providing
    a forward transformation (to a normal distribution) and its inverse.
    This allows for sampling from various distributions using a standard
    Gaussian random number generator.
    """
    forward: RealToReal
    inv: RealToReal

distributions = {
    "normal": Distribution(forward=lambda x: x, inv=lambda x: x),
    "lognormal": Distribution(forward=math.log, inv=math.exp),
}

# =============================================================================
# ODE MODEL DEFINITION (Owens-Bozic Model)
# =============================================================================

def cart_tumor_model(y: list[float], t: float, params: list[float]) -> list[float]:
    """
    Owens-Bozic model for Tumor, Effector T-cells, CAR-T cells, and Modulatory agent.
    States y: [T, E, C, M]
    Params: [a, b, dE, dC, g, jE, jC, K, k, l, mE, mC, qE, qC, s, KT, KE, KC, gamma] (19 parameters)
    """
    T, E, C, M = y
    a, b_param, dE, dC, g, jE, jC, K_param, k_param, l_param, mE, mC, qE, qC, s_param, KT, KE, KC, gamma = params

    # Ensure cell counts are non-negative
    C = max(C, 0)
    E = max(E, 0)
    T = max(T, 0) # Added for safety, though model dynamics should handle it.

    tol = 1e-10 # Tolerance for near-zero tumor values

    # DE term calculation (Effector cell killing)
    if T < tol:
        DE = 0
    elif E > 0 and T > 0 and E < T: # Check E > 0 and T > 0 to avoid division by zero if E/T is used
        DE = dE * (E / T)**l_param / (s_param + (E / T)**l_param) * T
    elif E > 0 and T > 0 : # Handles E >= T and ensures T/E is safe
        DE = dE * (1 - s_param / (s_param + (T/E)**(-l_param))) * T
    else: # E or T is zero or negative (though max(0,..) should prevent negative)
        DE = 0


    # DC term calculation (CAR-T cell killing)
    if T < tol:
        DC = 0
    elif C > 0 and T > 0 and C < T: # Check C > 0 and T > 0
        DC = dC * (C / T)**l_param / (s_param + (C / T)**l_param) * T
    elif C > 0 and T > 0: # Handles C >= T
        DC = dC * (1 - s_param / (s_param + (T/C)**(-l_param))) * T
    else: # C or T is zero or negative
        DC = 0


    dT_dt = a * T * (1 - b_param * T) - DE - DC - KT * (1 - np.exp(-M)) * T
    
    log_term_E = 0
    if K_param > 0 and (E + C) > 0: # Check K_param and sum to avoid log(0) or division by zero
        # Ensure argument of log is positive
        val_for_log_E = (E + C) / K_param
        if val_for_log_E > 0:
             log_term_E = np.log(val_for_log_E)

    log_term_C = 0
    if K_param > 0 and (E+C) > 0: # Check K_param and sum
        val_for_log_C = (E + C) / K_param
        if val_for_log_C > 0:
            log_term_C = np.log(val_for_log_C)


    # Check for k_param + DE**2 being zero
    denominator_E = k_param + DE**2
    term_jE = 0
    if denominator_E > 0 :
        term_jE = jE * log_term_E * (DE**2) / denominator_E * E

    denominator_C = k_param + DC**2
    term_jC = 0
    if denominator_C > 0:
        term_jC = jC * log_term_C * (DC**2) / denominator_C * C

    dE_dt = g - mE * E - term_jE  - qE * E * T - KE * (1 - np.exp(-M)) * E
    dC_dt = - mC * C - term_jC - qC * C * T - KC * (1 - np.exp(-M)) * C
    dM_dt = - gamma * M
    
    return [dT_dt, dE_dt, dC_dt, dM_dt]

# =============================================================================
# DATA GENERATION PIPELINE
# =============================================================================

def sampler(
    n_indivs: int,
    pop_params: list[float],
    param_dists: list[Distribution],
    param_stds: list[float],
    seed: int,
) -> list[list[float]]:
    assert len(pop_params) == len(param_dists) == len(param_stds)
    random.seed(seed)
    return [
        [
            dist.inv(random.gauss(dist.forward(mean), std))
            for mean, std, dist in zip(pop_params, param_stds, param_dists)
        ]
        for _ in range(n_indivs)
    ]

def generate_ground_truths(
    model: ODEModel,
    sampled_params: list[list[float]],
    initial_conditions: list[list[float]],
    observation_times: list[list[float]],
) -> list[np.ndarray]:
    return [
        odeint(model, ic, times, args=(params,)).T
        for ic, times, params in zip(initial_conditions, observation_times, sampled_params)
    ]

def add_noise(
    ground_truths: list[np.ndarray],
    error_params: list[tuple[float, float]], # e.g. [(err_T_a, err_T_b), (err_C_a, err_C_b)]
    observed_indices: tuple[int, int],       # e.g. (0, 2) for Tumor and CAR-T
    seed: int,
) -> list[np.ndarray]:
    random.seed(seed)
    noisy_observations = [gt.copy() for gt in ground_truths]

    idx_var1, idx_var2 = observed_indices
    err_params_var1, err_params_var2 = error_params[0], error_params[1]

    for individual_trajectory in noisy_observations:
        # Add noise to the first specified observed variable (e.g., Tumor)
        var1_trajectory = individual_trajectory[idx_var1]
        err_a1, err_b1 = err_params_var1
        for j, value in enumerate(var1_trajectory):
            proportional_error = err_b1 * abs(value)
            error_std_dev = err_a1 + proportional_error
            if error_std_dev < 0: error_std_dev = 0 # Ensure std_dev is not negative
            noise = random.gauss(0, error_std_dev)
            var1_trajectory[j] += noise
            if var1_trajectory[j] < 0:
                 var1_trajectory[j] = max(0, value * 0.1) 

        # Add noise to the second specified observed variable (e.g., CAR-T)
        var2_trajectory = individual_trajectory[idx_var2]
        err_a2, err_b2 = err_params_var2
        for j, value in enumerate(var2_trajectory):
            proportional_error = err_b2 * abs(value)
            error_std_dev = err_a2 + proportional_error
            if error_std_dev < 0: error_std_dev = 0 # Ensure std_dev is not negative
            noise = random.gauss(0, error_std_dev)
            var2_trajectory[j] += noise
            if var2_trajectory[j] < 0:
                 var2_trajectory[j] = max(0, value * 0.1)
    return noisy_observations

# =============================================================================
# EVALUATION AND PLOTTING
# =============================================================================

def evaluate_trajectories(
    ground_truths: list[np.ndarray],
    noisy_observations: list[np.ndarray],
    observed_indices: tuple[int, int] # (index_Tumor, index_CAR-T)
) -> None:
    rmses_T, mres_T = [], []
    rmses_C, mres_C = [], []

    idx_T, idx_C = observed_indices

    for gt, noisy in zip(ground_truths, noisy_observations):
        T_gt, C_gt = gt[idx_T], gt[idx_C]
        T_noisy, C_noisy = noisy[idx_T], noisy[idx_C]

        rmses_T.append(np.sqrt(np.mean((T_gt - T_noisy)**2)))
        rmses_C.append(np.sqrt(np.mean((C_gt - C_noisy)**2)))

        safe_T_gt = np.where(T_gt == 0, 1e-9, T_gt) # Avoid division by zero
        safe_C_gt = np.where(C_gt == 0, 1e-9, C_gt) # Avoid division by zero
        mres_T.append(np.median(np.abs((T_gt - T_noisy) / safe_T_gt)))
        mres_C.append(np.median(np.abs((C_gt - C_noisy) / safe_C_gt)))

    print("--- Evaluation (Noisy vs. Ground Truth) ---")
    print(f"  Tumor Cells | Mean RMSE: {np.mean(rmses_T):.2e} | Mean MRE: {np.mean(mres_T):.2f}")
    print(f"  CAR-T Cells | Mean RMSE: {np.mean(rmses_C):.2e} | Mean MRE: {np.mean(mres_C):.2f}")
    print("------------------------------------------")


def generate_and_plot_trajectories(
    n_responders: int,
    n_non_responders: int,
    responder_params_config: dict, 
    non_responder_params_config: dict, 
    initial_conditions: list, 
    time_points: np.ndarray,
    observed_indices_for_plot_noise_eval: tuple[int, int], 
    noise_params: list | None = None,
    seed: int = 42
) -> dict:
    idx_T_plot, idx_C_plot = observed_indices_for_plot_noise_eval

    params_r = sampler(n_responders, seed=seed, **responder_params_config)
    params_nr = sampler(n_non_responders, seed=seed + 1, **non_responder_params_config)

    ics_r = [initial_conditions] * n_responders
    ics_nr = [initial_conditions] * n_non_responders
    times_r = [time_points.tolist()] * n_responders
    times_nr = [time_points.tolist()] * n_non_responders

    gt_r = generate_ground_truths(cart_tumor_model, params_r, ics_r, times_r)
    gt_nr = generate_ground_truths(cart_tumor_model, params_nr, ics_nr, times_nr)

    noisy_r_data, noisy_nr_data = None, None 
    if noise_params:
        noisy_r_data = add_noise(gt_r, noise_params, observed_indices_for_plot_noise_eval, seed=seed + 2)
        noisy_nr_data = add_noise(gt_nr, noise_params, observed_indices_for_plot_noise_eval, seed=seed + 3)
        if noisy_r_data and noisy_nr_data: # Ensure data was generated
            evaluate_trajectories(gt_r + gt_nr, noisy_r_data + noisy_nr_data, observed_indices_for_plot_noise_eval)
        plot_data_r, plot_data_nr = noisy_r_data, noisy_nr_data
        plot_title = "Noisy Observations"
    else:
        plot_data_r, plot_data_nr = gt_r, gt_nr
        plot_title = "Ground Truth Trajectories"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    fig.suptitle(plot_title, fontsize=16)

    # Ensure there is data to plot
    if plot_data_r:
        for i, data in enumerate(plot_data_r):
            label = "Responders" if i == 0 else ""
            axes[0].plot(time_points, data[idx_T_plot], color="darkgreen", alpha=0.7, label=label, linewidth=2) # Tumor
            axes[1].plot(time_points, data[idx_C_plot], color="darkgreen", alpha=0.7, label=label, linewidth=2) # CAR-T
    
    if plot_data_nr:
        for i, data in enumerate(plot_data_nr):
            label = "Non-responders" if i == 0 else ""
            axes[0].plot(time_points, data[idx_T_plot], color="darkorange", alpha=0.7, label=label, linewidth=2) # Tumor
            axes[1].plot(time_points, data[idx_C_plot], color="darkorange", alpha=0.7, label=label, linewidth=2) # CAR-T

    axes[0].set_title("Tumor Cells", fontsize=14)
    axes[0].set_xlabel("Time post infusion (days)", fontsize=12)
    axes[0].set_ylabel("Cell Count", fontsize=12)
    axes[0].set_yscale('log') # Using log scale
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("CAR T Cells", fontsize=14)
    axes[1].set_xlabel("Time post infusion (days)", fontsize=12)
    axes[1].set_ylabel("Cell Count", fontsize=12)
    axes[1].set_yscale('log') # Using log scale
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return {
        'responder_params': params_r,
        'non_responder_params': params_nr,
        'ground_truths_r': gt_r,
        'ground_truths_nr': gt_nr,
        'noisy_r': noisy_r_data,
        'noisy_nr': noisy_nr_data
    }

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    # --- Simulation Configuration ---
    N_PARAMS = 19 # Number of parameters for Owens-Bozic model
    
    # If these are coefficients of variation, f_pop might be 1 and stds would be relative.
    # If stds are absolute log-stds, f_pop might be 1.
    f_pop = 1.0 

    # Parameter definitions from user's partial_pop_param_info
    # Structure: (param_name, mean_value, std_dev_on_transformed_scale, distribution_object)
    # The std_dev provided (e.g., 0.6 * f_pop) is assumed to be the standard deviation
    # for the parameter AFTER transformation (e.g., std dev of log(parameter) for lognormal).
    pop_param_definitions = [
        ("a", 3.3e-1, 0.06 * f_pop, distributions["lognormal"]),
        ("b", 2.26e-11, 5.1 * f_pop, distributions["lognormal"]),
        ("dE", 3.17, 0.6 * f_pop, distributions["lognormal"]),
        ("dC", 2.25, 0.01 * f_pop, distributions["lognormal"]), # Base for CAR-T killing
        ("g", 1.03e4, 1.7 * f_pop, distributions["lognormal"]),
        ("jE", 1.56e-2, 0.75 * f_pop, distributions["lognormal"]),
        ("jC", 3.46e-1, 0.75 * f_pop, distributions["lognormal"]),
        ("K", 5.21e8, 1.5 * f_pop, distributions["lognormal"]), # K_param in model
        ("k", 8.67e6, 3.0 * f_pop, distributions["lognormal"]), # k_param in model
        ("l", 1.418, 0.013 * f_pop, distributions["lognormal"]),# l_param in model
        ("mE", 1.76e-2, 0.76 * f_pop, distributions["lognormal"]),
        ("mC", 0.293, 0.01 * f_pop, distributions["lognormal"]), # Base for CAR-T death
        ("qE", 1.28e-10, 1.35 * f_pop, distributions["lognormal"]),
        ("qC", 2.14e-10, 2.6 * f_pop, distributions["lognormal"]),
        ("s", 3.02e-1, 0.17 * f_pop, distributions["lognormal"]), # s_param in model
        ("KT", 0.7, 0.01 * f_pop, distributions["lognormal"]),
        ("KE", 0.6, 0.01 * f_pop, distributions["lognormal"]),
        ("KC", 0.6, 0.01 * f_pop, distributions["lognormal"]),
        ("gamma", 0.9, 0.01 * f_pop, distributions["lognormal"]),
    ]

    # Extract base population parameters, standard deviations, and distributions
    base_pop_params = [p[1] for p in pop_param_definitions]
    PARAM_STDS = [p[2] for p in pop_param_definitions]
    PARAM_DISTS = [p[3] for p in pop_param_definitions]

    # Create specific parameter sets for Responders and Non-responders
    # Modifying dC (index 3) and mC (index 11)
    RESPONDER_POP_PARAMS = list(base_pop_params)
    RESPONDER_POP_PARAMS[3] = base_pop_params[3] * 2.0  # e.g., Higher dC (CAR-T killing) for responders
    RESPONDER_POP_PARAMS[11] = base_pop_params[11] * 0.5 # e.g., Lower mC (CAR-T death) for responders

    NON_RESPONDER_POP_PARAMS = list(base_pop_params)
    NON_RESPONDER_POP_PARAMS[3] = base_pop_params[3] * 0.5  # e.g., Lower dC for non-responders
    NON_RESPONDER_POP_PARAMS[11] = base_pop_params[11] * 1.5 # e.g., Higher mC for non-responders
    
    # Ensure that any modified params remain positive if they are lognormal
    for i in range(len(RESPONDER_POP_PARAMS)):
        if PARAM_DISTS[i] == distributions["lognormal"] and RESPONDER_POP_PARAMS[i] <= 0:
            RESPONDER_POP_PARAMS[i] = 1e-9 # A small positive number
    for i in range(len(NON_RESPONDER_POP_PARAMS)):
        if PARAM_DISTS[i] == distributions["lognormal"] and NON_RESPONDER_POP_PARAMS[i] <= 0:
            NON_RESPONDER_POP_PARAMS[i] = 1e-9 # A small positive number


    RESPONDER_PARAMS_CONFIG = {
        "pop_params": RESPONDER_POP_PARAMS,
        "param_dists": PARAM_DISTS, # Use the globally defined PARAM_DISTS
        "param_stds": PARAM_STDS    # Use the globally defined PARAM_STDS
    }
    NON_RESPONDER_PARAMS_CONFIG = {
        "pop_params": NON_RESPONDER_POP_PARAMS,
        "param_dists": PARAM_DISTS, # Use the globally defined PARAM_DISTS
        "param_stds": PARAM_STDS    # Use the globally defined PARAM_STDS
    }

    NOISE_PARAMS_FOR_T_AND_C = [
        (1e3, 0.20),  # Error for Tumor cells (constant_std, proportional_std)
        (1e2, 0.20)   # Error for CAR-T cells (constant_std, proportional_std)
    ]

    # Initial conditions: [T0, E0, C0, M0]
    INITIAL_CONDITIONS = [1e7, 5e5, 2e6, 1.0] 
    TIME_POINTS = np.linspace(0, 100, 100) 

    OBSERVED_INDICES = (0, 2) # Tumor (idx 0), CAR-T (idx 2) in [T, E, C, M]

    print("Generating plot for ground truth trajectories (Owens-Bozic model with updated params)...")
    results_gt = generate_and_plot_trajectories(
        n_responders=5,
        n_non_responders=3,
        responder_params_config=RESPONDER_PARAMS_CONFIG,
        non_responder_params_config=NON_RESPONDER_PARAMS_CONFIG,
        initial_conditions=INITIAL_CONDITIONS,
        time_points=TIME_POINTS,
        observed_indices_for_plot_noise_eval=OBSERVED_INDICES,
        noise_params=None,
        seed=42
    )

    print("\nGenerating plot for noisy observations (Owens-Bozic model with updated params)...")
    results_noisy = generate_and_plot_trajectories(
        n_responders=5,
        n_non_responders=3,
        responder_params_config=RESPONDER_PARAMS_CONFIG,
        non_responder_params_config=NON_RESPONDER_PARAMS_CONFIG,
        initial_conditions=INITIAL_CONDITIONS,
        time_points=TIME_POINTS,
        observed_indices_for_plot_noise_eval=OBSERVED_INDICES,
        noise_params=NOISE_PARAMS_FOR_T_AND_C,
        seed=42
    )

    print("\nSimulation completed successfully with user-provided parameter details!")