from run_experiment import (
    MlxParam,
    MlxErrorModelParam,
    MlxObsError,
    MlxObsVar,
    generate_mlxtran_file,
    run_experiment,
)

MODEL_PARAMS = [
    ("a", 3.3e-1, 0.6),
    ("b", 2.26e-11, 5.1),
    ("dE", 3.17, 0.6),
    ("dC", 2.25, 0.01),
    ("g", 1.03e4, 1.7),
    ("jE", 1.56e-2, 0.75),
    ("jC", 3.46e-1, 0.75),
    ("K", 5.21e8, 1.5),
    ("k", 8.67e6, 3.0),
    ("l", 1.418, 0.013),
    ("mE", 1.76e-2, 0.76),
    ("mC", 0.293, 0.01),
    ("qE", 1.28e-10, 1.35),
    ("qC", 2.14e-10, 2.6),
    ("s", 3.02e-1, 0.17),
    ("KT", 0.7, 0.01),
    ("KE", 0.6, 0.01),
    ("KC", 0.6, 0.01),
    ("gamma", 0.9, 0.01),
]

OBS_VARS = [
    ("T", 1.58e9, 1.5),
    ("E", 4e5, 0.01),
    ("C", 6.3e7, 2.0),
    ("M", 6.0, 0.2),
]

def csv_and_indiv_param_vals(
    param: str,
    noise_level: float,
    n_obs: int,
    param_seed: int,
    noise_seed: int,
    csv_filename: str | None = None,
    n_indivs=10,
    max_day_number=100,
):
    f_pop = 1.0     # scale factor for pop param std

    partial_pop_param_info = [
        ("a", 3.3e-1, 0.6 * f_pop, distributions["lognormal"]),
        ("b", 2.26e-11, 5.1 * f_pop, distributions["lognormal"]),
        ("dE", 3.17, 0.6 * f_pop, distributions["lognormal"]),
        ("dC", 2.25, 0.01 * f_pop, distributions["lognormal"]),
        ("g", 1.03e4, 1.7 * f_pop, distributions["lognormal"]),
        ("jE", 1.56e-2, 0.75 * f_pop, distributions["lognormal"]),
        ("jC", 3.46e-1, 0.75 * f_pop, distributions["lognormal"]),
        ("K", 5.21e8, 1.5 * f_pop, distributions["lognormal"]),
        ("k", 8.67e6, 3.0 * f_pop, distributions["lognormal"]),
        ("l", 1.418, 0.013 * f_pop, distributions["lognormal"]),
        ("mE", 1.76e-2, 0.76 * f_pop, distributions["lognormal"]),
        ("mC", 0.293, 0.01 * f_pop, distributions["lognormal"]),
        ("qE", 1.28e-10, 1.35 * f_pop, distributions["lognormal"]),
        ("qC", 2.14e-10, 2.6 * f_pop, distributions["lognormal"]),
        ("s", 3.02e-1, 0.17 * f_pop, distributions["lognormal"]),
        ("KT", 0.7, 0.01 * f_pop, distributions["lognormal"]),
        ("KE", 0.6, 0.01 * f_pop, distributions["lognormal"]),
        ("KC", 0.6, 0.01 * f_pop, distributions["lognormal"]),
        ("gamma", 0.9, 0.01 * f_pop, distributions["lognormal"]),
    ]

    param_idx = -1
    for i, x in enumerate(partial_pop_param_info):
        if x[0] == param:
            param_idx = i
            break

    if param_idx == -1:
        raise ValueError(f"unknown param {param}")
    
    pop_param_info = [(*x, x[0] != param) for x in partial_pop_param_info]
    obs_var_info = [
        ("T", 1.58e9, 1.5, distributions["normal"], "combined1", 0, noise_level, 1, distributions["normal"], "log10", True),
        ("E", 4e5, 0.01, distributions["normal"], "combined1", 0, noise_level, 1, distributions["normal"], "log10", True),
        ("C", 6.3e7, 2.0, distributions["normal"], "combined1", 0, noise_level, 1, distributions["normal"], "log10", True),
        ("M", 6.0, 0.2, distributions["normal"], "combined1", 0, noise_level, 1, distributions["normal"], "log10", True),
    ]
    obs_times_all = [[max_day_number * i // n_obs for i in range(1, n_obs + 1)] for _ in range(n_indivs)]
    sampled_params_all, initial_conditions_all, _, _, _ = generate_sample_csv(
        csv_filename or "owens_bozic_noiseexp",
        owens_bozic_model,
        pop_param_info,
        obs_var_info,
        n_indivs,
        n_obs,
        max_day_number,
        obs_times_all=obs_times_all,
        param_seed=param_seed,
        noise_seed=noise_seed,
    )

    return [x[param_idx] for x in sampled_params_all], sampled_params_all, initial_conditions_all, param_idx
