import os
import shutil
import time
from pathlib import Path
from typing import TextIO

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


# TODO: make this accept a list of parameters (call it `params`) instead of a single parameter
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

    # TODO: maintain a list `param_idxs` instead of a single `param_idx`
    param_idx = -1
    for i, x in enumerate(partial_pop_param_info):
        if x[0] == param:
            param_idx = i
            break
    if param_idx == -1:
        raise ValueError(f"unknown param {param}")

    # TODO: instead of doing x[0] != param, check if x[0] is not in `params`
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
        [
            x[param_idx] for x in sampled_params_all
        ],  # TODO: make this a list of lists, one for each element of `param_idxs`
        sampled_params_all,
        initial_conditions_all,
        param_idx,  # TODO: use the `param_idxs` list instead
    )


def get_pred_params(param: str):
    """
    Get a list of the predicted values of `param` for each patient
    """
    fp = "./owens_bozic/IndividualParameters/estimatedIndividualParameters.txt"
    with open(fp, "r") as f:
        cols = f.readline().split(",")
        col_idx = cols.index(f"{param}_mode")
        pred_params = []

        while True:
            line = f.readline()
            if not line:
                break
            vals = line.split(",")
            pred_params.append(float(vals[col_idx]))

    return pred_params


def get_pred_params_all(
    pred_params: list[
        float
    ],  # TODO: make this `pred_params_list: list[list[float]]` where each list[float] corresponds to a different param
    true_params_all: list[list[float]],
    param_idx: int,  # TODO: make this `param_idxs: list[int]`
):
    pred_params_all = [[x for x in true_params] for true_params in true_params_all]
    # TODO: within each list in `pred_params_all`, update the value at each index in `param_idxs`
    # based on the corresponding value in the corresponding list in `pred_params_list`
    for params, pred_param in zip(pred_params_all, pred_params):
        params[param_idx] = pred_param

    return pred_params_all


def compute_rms(
    true_params_all: list[list[float]],
    pred_params_all: list[list[float]],
    initial_conditions_all: list[list[float]],
    obs_times_all: list[list[int]],
    model=owens_bozic_model,
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


def compute_rms_log(
    true_params_all: list[list[float]],
    pred_params_all: list[list[float]],
    initial_conditions_all: list[list[float]],
    obs_times_all: list[list[int]],
    model=owens_bozic_model,
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


def get_outcome_type(
    params: list[float],
    initial_conditions: list[float],
    obs_times: list[int],
    model=owens_bozic_model,
):
    ground_truth = generate_ground_truths(
        model, [params], [initial_conditions], [obs_times]
    )[0]

    ground_truth_tumor = ground_truth[0]
    if ground_truth_tumor[-1] > 1e3:
        return "NR"
    return "CR"


def run_and_record(
    param: str,  # TODO: make this `params: list[str]`
    n_pts: int,
    noise_level: float,
    n_indivs: int,
    param_seed: int,
    result_file: TextIO,
    num_trials=5,
    csv_path="owens_bozic_perf.csv",
    model_path="owens_bozic.txt",
    mlxtran_name="owens_bozic",
    hidden_obs_var_idxs: list[int] = [],
):
    model_params = [
        MlxParam(
            name=name,
            dist_name="logNormal",
            init_est_pop=mean,
            init_est_sd=sd,
            is_fixed=param
            != name,  # TODO: make this check if `name` is not in `params`
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

    temp_csv_path = f"{csv_path.removesuffix(".csv")}-temp.csv"
    for trial in range(num_trials):
        noise_seed = (
            sum([ord(c) for c in param])  # TODO: use `params[0]` instead of `param`
            + int(100 * noise_level * n_pts * n_indivs)
            + trial
        )
        (
            true_indiv_params,  # TODO: update accordingly
            sampled_params_all,
            initial_conditions_all,
            param_idx,  # TODO: update accordingly
        ) = csv_and_indiv_param_vals(
            param,  # TODO: update accordingly
            noise_level,
            n_pts,
            param_seed=param_seed,
            noise_seed=noise_seed,
            csv_filename=temp_csv_path,
            n_indivs=n_indivs,
        )
        # strip observations for all hidden observation variables
        with open(temp_csv_path, "r") as f_orig:
            with open(csv_path, "w") as f:
                for line in f_orig.readlines():
                    if any(
                        x in line
                        for x in ["time"]
                        + [
                            obs_var
                            for i, (obs_var, *_) in enumerate(OBS_VARS)
                            if i not in hidden_obs_var_idxs
                        ]
                    ):
                        f.write(line)
        print(f"csv generated at {csv_path}")
        generate_mlxtran_file(
            name=mlxtran_name,
            csv_path=csv_path,
            model_path=model_path,
            model_params=model_params,
            obs_vars=obs_vars,
            hidden_obs_var_idxs=hidden_obs_var_idxs,
        )

        if "owens_bozic" in os.listdir("."):
            cwd = Path.cwd()
            shutil.rmtree(str(cwd.joinpath("owens_bozic")))
        # wait until monolix finishes running before trying to compute metrics
        run_experiment(
            mlxtran_path=f"{mlxtran_name}.mlxtran",
            mode="basic",
            n_threads=32,
        )
        result_fp = Path(
            "./owens_bozic/IndividualParameters/estimatedIndividualParameters.txt"
        )
        while not result_fp.exists():
            print("\nwaiting for SAEM...\n")
            time.sleep(5)
        while True:
            with open(result_fp, "r") as f_results:
                first_line = f_results.readline()
                if f"{param}_mode" in first_line.split(","):  # TODO: use `params[0]`
                    break
                print("\nwaiting for individual estimation...\n")
                time.sleep(5)

        summary_fp = Path("./owens_bozic/summary.txt")
        while (
            not summary_fp.exists()
        ):  # the file should exist by now, but just in case...
            print("\nwaiting for summary...\n")
            time.sleep(5)

        exploratory_did_converge = None
        smoothing_did_converge = None
        with open(summary_fp, "r") as f_summary:
            for line in f_summary.readlines():
                if "Exploratory phase" in line:
                    exploratory_did_converge = "Autostop" in line
                    print(f"{exploratory_did_converge=}")
                if "Smoothing phase" in line:
                    smoothing_did_converge = "Autostop" in line
                    print(f"{smoothing_did_converge=}")

        pred_params = get_pred_params(
            param
        )  # TODO: make this `pred_params_list` using `params`

        get_avg_rel_err = lambda pred, true: np.mean(
            np.abs(np.array(pred) - np.array(true)) / np.array(true)
        )

        # TODO: update the rest of the code based on the fact that
        # we have multiple parameters `params` instead of a single one `param`.
        # In particular, in the variable `output`, every line that includes "param"
        # besides the one with "param_seed" should be modified to reflect that
        # we are fitting multiple parameters instead of just one.
        # There will be another todo right before the `output` variable
        pred_minus_true = np.array(pred_params) - np.array(true_indiv_params)
        avg_abs_err = np.mean(np.abs(pred_minus_true))
        avg_rel_err = get_avg_rel_err(pred_params, true_indiv_params)
        print(np.array(true_indiv_params))
        print(
            f"{param=}, {noise_level=}, {n_pts=}: {avg_abs_err:.6f}, {100 * avg_rel_err:.6f}%"
        )
        pred_params_all = get_pred_params_all(
            pred_params, sampled_params_all, param_idx
        )
        rms_obs_times_all = [list(range(5, 105, 5)) for _ in range(n_indivs)]
        outcomes = [
            get_outcome_type(sampled_params, initial_conditions, rms_obs_times)
            for sampled_params, initial_conditions, rms_obs_times in zip(
                sampled_params_all, initial_conditions_all, rms_obs_times_all
            )
        ]
        print(outcomes)

        keep = lambda lst, outcome_type: list(
            map(
                lambda x: x[0],
                filter(lambda x: x[1] == outcome_type, zip(lst, outcomes)),
            )
        )

        true_params_CR = keep(true_indiv_params, "CR")
        true_params_NR = keep(true_indiv_params, "NR")
        pred_single_params_CR = keep(pred_params, "CR")
        pred_single_params_NR = keep(pred_params, "NR")

        avg_rel_err_CR = (
            -1
            if not true_params_CR
            else get_avg_rel_err(pred_single_params_CR, true_params_CR)
        )
        avg_rel_err_NR = (
            -1
            if not true_params_NR
            else get_avg_rel_err(pred_single_params_NR, true_params_NR)
        )

        sampled_params_CR = keep(sampled_params_all, "CR")
        sampled_params_NR = keep(sampled_params_all, "NR")
        pred_params_CR = keep(pred_params_all, "CR")
        pred_params_NR = keep(pred_params_all, "NR")
        initial_conditions_CR = keep(initial_conditions_all, "CR")
        initial_conditions_NR = keep(initial_conditions_all, "NR")

        rms_vals_CR = []
        if sampled_params_CR:
            rms_vals_CR = compute_rms(
                sampled_params_CR,
                pred_params_CR,
                initial_conditions_CR,
                obs_times_all=rms_obs_times_all[: len(sampled_params_CR)],
            )
        print("CR", rms_vals_CR)

        rms_vals_NR = []
        if sampled_params_NR:
            rms_vals_NR = compute_rms(
                sampled_params_NR,
                pred_params_NR,
                initial_conditions_NR,
                obs_times_all=rms_obs_times_all[: len(sampled_params_NR)],
            )
        print("NR", rms_vals_NR)

        rms_log_vals_CR = []
        if sampled_params_CR:
            rms_log_vals_CR = compute_rms_log(
                sampled_params_CR,
                pred_params_CR,
                initial_conditions_CR,
                obs_times_all=rms_obs_times_all[: len(sampled_params_CR)],
            )
        print("CR_log", rms_log_vals_CR)

        rms_log_vals_NR = []
        if sampled_params_NR:
            rms_log_vals_NR = compute_rms_log(
                sampled_params_NR,
                pred_params_NR,
                initial_conditions_NR,
                obs_times_all=rms_obs_times_all[: len(sampled_params_NR)],
            )
        print("NR_log", rms_log_vals_NR)

        rms_vals = compute_rms(
            sampled_params_all,
            pred_params_all,
            initial_conditions_all,
            obs_times_all=rms_obs_times_all,
        )
        print(rms_vals)

        rms_log_vals = compute_rms_log(
            sampled_params_all,
            pred_params_all,
            initial_conditions_all,
            obs_times_all=rms_obs_times_all,
        )
        print(f"{setting_idx=}, {trial=}", rms_log_vals)

        obs_var_names = ["T", "E", "C", "M"]

        # TODO: we should have a list of `outputs`, one per parameter,
        # in the same format as below
        output = ",".join(
            [
                f"{param=}",
                f"{noise_level=}",
                f"{n_indivs=}",
                f"{n_pts=}",
                f"{param_seed=}",
                f"{noise_seed=}",
                f"{exploratory_did_converge=}",
                f"{smoothing_did_converge=}",
                f"param_relative_err={avg_rel_err}",
                f"param_relative_err_CR={avg_rel_err_CR}",
                f"param_relative_err_NR={avg_rel_err_NR}",
                ",".join(
                    [
                        f"rms_log_{obs_var}={y}"
                        for obs_var, y in zip(obs_var_names, rms_log_vals)
                    ]
                ),
                ",".join(
                    [
                        f"rms_{obs_var}={y}"
                        for obs_var, y in zip(obs_var_names, rms_vals)
                    ]
                ),
                ",".join(
                    [
                        f"rms_log_CR_{obs_var}={y}"
                        for obs_var, y in zip(
                            obs_var_names,
                            (
                                [-1] * len(obs_var_names)
                                if len(rms_log_vals_CR) == 0
                                else rms_log_vals_CR
                            ),
                        )
                    ]
                ),
                ",".join(
                    [
                        f"rms_CR_{obs_var}={y}"
                        for obs_var, y in zip(
                            obs_var_names,
                            (
                                [-1] * len(obs_var_names)
                                if len(rms_vals_CR) == 0
                                else rms_vals_CR
                            ),
                        )
                    ]
                ),
                ",".join(
                    [
                        f"rms_log_NR_{obs_var}={y}"
                        for obs_var, y in zip(
                            obs_var_names,
                            (
                                [-1] * len(obs_var_names)
                                if len(rms_log_vals_NR) == 0
                                else rms_log_vals_NR
                            ),
                        )
                    ]
                ),
                ",".join(
                    [
                        f"rms_NR_{obs_var}={y}"
                        for obs_var, y in zip(
                            obs_var_names,
                            (
                                [-1] * len(obs_var_names)
                                if len(rms_vals_NR) == 0
                                else rms_vals_NR
                            ),
                        )
                    ]
                ),
                f"true_params: {true_indiv_params}",
                f"pred_params: {pred_params}",
                f"outcomes: {outcomes}",
            ]
        )

        result_file.write(output + "\n")
        result_file.flush()     # so we can see results immediately

if __name__ == "__main__":
    from itertools import product

    setting_combos_1 = list(product([20, 10, 7, 5, 3], [0.1, 0.25, 0.5], [10]))
    setting_combos_2 = list(product([20, 10, 5], [0.1, 0.25], [1, 5, 20]))
    setting_combos_3 = list(product([20, 10, 5], [0.1, 0.2, 0.3, 0.4, 0.5, 0.7], [10]))

    setting_combos = setting_combos_1
    params_with_seeds = [("l", 1), ("a", 9), ("s", 105), ("jC", 10), ("dC", 10)]
    num_trials = 5

    # TODO: update this to select a list of two params,
    # but still only one param_seed (from the first param)
    param, param_seed = params_with_seeds[1]
    with open(f"results_{param}_TC_only.txt", "a") as f:
        # for setting_idx in range(len(setting_combos)):
        for setting_idx in range(2):
            n_pts, noise_level, n_indivs = setting_combos[setting_idx]
            run_and_record(
                param=param,
                n_pts=n_pts,
                noise_level=noise_level,
                n_indivs=n_indivs,
                param_seed=param_seed + abs(n_indivs - 10),
                result_file=f,
                num_trials=num_trials,
                model_path="owens_bozic_2.txt",
                hidden_obs_var_idxs=[1, 3],
            )
