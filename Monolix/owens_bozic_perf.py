import numpy as np
from run_experiment import (
    MlxErrorModelParam,
    MlxObsVar,
    MlxParam,
    generate_mlxtran_file,
    run_experiment,
)
from synthetic_data import distributions, generate_ground_truths, generate_sample_csv

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


def owens_bozic_model(y, t, params):
    T, E, C, M = y
    a, b, dE, dC, g, jE, jC, K, k, l, mE, mC, qE, qC, s, KT, KE, KC, gamma = params

    C = max(C, 0)
    E = max(E, 0)

    tol = 1e-10

    if T < tol:
        DE = 0
    elif E < T:
        DE = dE * (E / T) ** l / (s + (E / T) ** l) * T
    else:
        DE = dE * (1 - s / (s + (T / E) ** (-l))) * T

    if T < tol:
        DC = 0
    elif C < T:
        DC = dC * (C / T) ** l / (s + (C / T) ** l) * T
    else:
        DC = dC * (1 - s / (s + (T / C) ** (-l))) * T

    dT_dt = a * T * (1 - b * T) - DE - DC - KT * (1 - np.exp(-M)) * T
    dE_dt = (
        g
        - mE * E
        - jE * np.log((E + C) / K) * (DE**2) / (k + DE**2) * E
        - qE * E * T
        - KE * (1 - np.exp(-M)) * E
    )
    dC_dt = (
        -mC * C
        - jC * np.log((E + C) / K) * (DC**2) / (k + DC**2) * C
        - qC * C * T
        - KC * (1 - np.exp(-M)) * C
    )
    dM_dt = -gamma * M

    return [dT_dt, dE_dt, dC_dt, dM_dt]


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
    partial_pop_param_info = [(*x, distributions["lognormal"]) for x in MODEL_PARAMS]

    param_idx = -1
    for i, x in enumerate(partial_pop_param_info):
        if x[0] == param:
            param_idx = i
            break
    if param_idx == -1:
        raise ValueError(f"unknown param {param}")

    pop_param_info = [(*x, x[0] != param) for x in partial_pop_param_info]
    obs_var_info = [
        (
            *x,
            distributions["normal"],  # initial value distribution
            "combined1",  # error model
            0,  # constant error param
            noise_level,  # proportional error param
            1,
            distributions["normal"],  # error distribution
            "log10",  # output format
            True,  # fix initial value?
        )
        for x in OBS_VARS
    ]
    obs_times_all = [
        [max_day_number * i // n_obs for i in range(1, n_obs + 1)]
        for _ in range(n_indivs)
    ]
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

    return (
        [x[param_idx] for x in sampled_params_all],
        sampled_params_all,
        initial_conditions_all,
        param_idx,
    )


def get_pred_params(param: str):  # fix soon to read the other file
    """
    Get a list of the predicted values of `param` for each patient
    """
    fp = "./owens_bozic/IndividualParameters/simulatedIndividualParameters.txt"
    with open(fp, "r") as f:
        cols = f.readline().split(",")
        col_idx = cols.index(param)
        pred_params = []

        curr_indiv_preds = []
        while True:
            line = f.readline()
            vals = line.split(",")
            if not line or int(vals[0]) == 1 and curr_indiv_preds:
                pred_params.append(np.median(curr_indiv_preds))
                curr_indiv_preds = []
            if not line:
                break
            curr_indiv_preds.append(float(vals[col_idx]))

    return pred_params


def get_pred_params_all(
    pred_params: list[float], true_params_all: list[list[float]], param_idx: int
):
    pred_params_all = [[x for x in true_params] for true_params in true_params_all]
    for params, pred_param in zip(pred_params_all, pred_params):
        params[param_idx] = pred_param

    return pred_params_all


n_indivs = 10
obs_times_all = [list(range(5, 101, 5)) for _ in range(n_indivs)]


def compute_rms(
    true_params_all: list[list[float]],
    pred_params_all: list[list[float]],
    initial_conditions_all: list[list[float]],
    model=owens_bozic_model,
    obs_times_all=obs_times_all,
):
    true_ground_truths = generate_ground_truths(
        model, true_params_all, initial_conditions_all, obs_times_all
    )
    pred_ground_truths = generate_ground_truths(
        model, pred_params_all, initial_conditions_all, obs_times_all
    )
    rms_all = []
    for true_ground_truth, pred_ground_truth in zip(
        true_ground_truths, pred_ground_truths
    ):
        squared_diffs = (pred_ground_truth - true_ground_truth) ** 2
        rms = np.sqrt(np.mean(squared_diffs, axis=1))
        rms_all.append(rms)
    avg_rms_all = np.mean(rms_all, axis=0)
    return avg_rms_all


if __name__ == "__main__":
    from itertools import product

    combos = list(product([20, 10, 5], [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]))
    param = "a"
    with open(f"results_{param}.txt", "a") as f:
        # for idx in range(len(combos)):
        for idx in range(17, 18):
            n_pts, noise_level = combos[idx]
            model_params = [
                MlxParam(
                    name=name,
                    dist_name="logNormal",
                    init_est_pop=mean,
                    init_est_sd=sd,
                    is_fixed=param != name,
                )
                for name, mean, sd in MODEL_PARAMS
            ]
            obs_vars = [
                MlxObsVar(
                    name=name,
                    pred_name=name * (2 if name in "TM" else 3),
                    id=id,
                    dist_name="logNormal",
                    init_est_pop=mean,
                    init_est_sd=sd,
                    error_model_name="proportional",
                    error_model_dist_name="normal",
                    error_params=[
                        MlxErrorModelParam(name="b", init_est=0.1, is_fixed=False),
                        MlxErrorModelParam(name="c", init_est=1, is_fixed=True),
                    ],
                    is_fixed=True,
                )
                for id, (name, mean, sd) in enumerate(OBS_VARS)
            ]
            csv_path = "owens_bozic_perf.csv"
            model_path = "owens_bozic.txt"
            mlxtran_name = "owens_bozic"

            param_seed = 9  # may want to adjust depending on param
            num_trials = 5
            for trial in range(num_trials):
                noise_seed = sum([ord(c) for c in param]) + num_trials * idx + trial
                (
                    true_indiv_params,
                    sampled_params_all,
                    initial_conditions_all,
                    param_idx,
                ) = csv_and_indiv_param_vals(
                    param,
                    noise_level,
                    n_pts,
                    param_seed=param_seed,
                    noise_seed=noise_seed,
                    csv_filename=csv_path,
                )
                print(f"csv generated at {csv_path}")
                generate_mlxtran_file(
                    name=mlxtran_name,
                    csv_path=csv_path,
                    model_path=model_path,
                    model_params=model_params,
                    obs_vars=obs_vars,
                )

                import shutil
                import os
                import time
                from pathlib import Path

                if "owens_bozic" in os.listdir("."):
                    cwd = Path.cwd()
                    shutil.rmtree(str(cwd.joinpath("owens_bozic")))
                # wait until monolix finishes running before trying to compute metrics
                run_experiment(
                    mlxtran_path=f"{mlxtran_name}.mlxtran", mode="basic", n_threads=32
                )
                result_fp = Path(
                    "./owens_bozic/IndividualParameters/simulatedIndividualParameters.txt"
                )
                while not result_fp.exists():
                    print("\n" * 3 + "waiting..." + "\n" * 3)
                    time.sleep(5)
                time.sleep(5)  # make sure results are in

                pred_params = get_pred_params(param)
                avg_abs_err = np.mean(
                    np.abs(np.array(pred_params) - np.array(true_indiv_params))
                )
                avg_rel_err = np.mean(
                    np.abs(np.array(pred_params) - np.array(true_indiv_params))
                    / np.array(true_indiv_params)
                )
                print(np.array(true_indiv_params))
                print(np.array(pred_params) - np.array(true_indiv_params))
                print(
                    f"{param=}, {noise_level=}, {n_pts=}: {avg_abs_err:.6f}, {100 * avg_rel_err:.6f}%"
                )
                pred_params_all = get_pred_params_all(
                    pred_params, sampled_params_all, param_idx
                )
                rms_vals = compute_rms(
                    sampled_params_all, pred_params_all, initial_conditions_all
                )
                print(rms_vals)

                def compute_rms_log(
                    true_params_all: list[list[float]],
                    pred_params_all: list[list[float]],
                    initial_conditions_all: list[list[float]],
                    model=owens_bozic_model,
                    obs_times_all=obs_times_all,
                ):
                    true_ground_truths = generate_ground_truths(
                        model, true_params_all, initial_conditions_all, obs_times_all
                    )
                    pred_ground_truths = generate_ground_truths(
                        model, pred_params_all, initial_conditions_all, obs_times_all
                    )
                    rms_all = []
                    for true_ground_truth, pred_ground_truth in zip(
                        true_ground_truths, pred_ground_truths
                    ):
                        squared_diffs = (
                            (
                                np.log(np.abs(pred_ground_truth) + 1e-2)
                                - np.log(np.abs(true_ground_truth) + 1e-2)
                            )
                            / np.log(10)
                        ) ** 2
                        rms = np.sqrt(np.mean(squared_diffs, axis=1))
                        rms_all.append(rms)
                    avg_rms_all = np.mean(rms_all, axis=0)
                    return avg_rms_all

                rms_log_vals = compute_rms_log(
                    sampled_params_all, pred_params_all, initial_conditions_all
                )
                print(rms_log_vals)

                output = (
                    f"{param=},{noise_level=},{n_pts=},{param_seed=},{noise_seed=},"
                    + f"relative_err={100 * avg_rel_err}%,{rms_log_vals=},{rms_vals=}\n"
                )
                f.write(output)
