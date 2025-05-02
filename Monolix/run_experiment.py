import os
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from string import Template

from dotenv import load_dotenv


@dataclass
class MlxParam:
    """Only supports MLE estimation or fixed"""

    name: str
    dist_name: str
    init_est_pop: float
    init_est_sd: float
    is_fixed: bool


@dataclass
class MlxErrorModelParam:
    """Only supports MLE estimation or fixed"""

    name: str
    init_est: float
    is_fixed: bool


@dataclass
class MlxObsError:
    error_model_name: str
    error_model_dist_name: str
    error_params: list[MlxErrorModelParam]


@dataclass
class MlxObsVar(MlxParam, MlxObsError):
    id: int
    pred_name: str


def generate_mlxtran_file(
    name: str,
    csv_path: str,
    model_path: str,
    model_params: list[MlxParam],
    obs_vars: list[MlxObsVar],
    hidden_obs_var_idxs: list[int] = [],
):
    # everything that needs to be generated
    INDIVIDUAL_input = []
    INDIV_DEFINITION_items_ = []
    LONGITUDINAL_input = []
    LONG_DEFINITION_items_ = []
    FIT_data = []  # also use for CONTENT.observation.yname
    FIT_model = []
    PARAMETER_items_ = []

    for model_param in model_params:
        name_pop = f"{model_param.name}_pop"
        omega_name = f"omega_{model_param.name}"

        INDIVIDUAL_input.append(name_pop)
        INDIVIDUAL_input.append(omega_name)
        INDIV_DEFINITION_items_.append(
            f"{model_param.name} = "
            + f"{{distribution={model_param.dist_name}, "
            + f"typical={name_pop}, "
            + (f"sd={omega_name}}}" if not model_param.is_fixed else "no-variability}")
        )
        PARAMETER_items_.append(
            f"{name_pop} = "
            + f"{{value={model_param.init_est_pop}, "
            + f"method={'MLE' if not model_param.is_fixed else 'FIXED'}}}"
        )
        if not model_param.is_fixed:
            PARAMETER_items_.append(
                f"{omega_name} = {{value={model_param.init_est_sd}, method=MLE}}"
            )

    for obs_var in obs_vars:
        name0 = f"{obs_var.name}0"
        name_pop = f"{name0}_pop"
        omega_name = f"omega_{name0}"
        nonfixed_error_param_names_id = [
            f"{error_param.name}{obs_var.id}"
            for error_param in obs_var.error_params
            if not error_param.is_fixed
        ]

        INDIVIDUAL_input.append(name_pop)
        INDIVIDUAL_input.append(omega_name)
        INDIV_DEFINITION_items_.append(
            f"{name0} = "
            + f"{{distribution={obs_var.dist_name}, "
            + f"typical={name_pop}, "
            + (f"sd={omega_name}}}" if not obs_var.is_fixed else "no-variability}")
        )
        if obs_var.id not in hidden_obs_var_idxs:
            LONGITUDINAL_input.extend(nonfixed_error_param_names_id)
            LONG_DEFINITION_items_.append(
                f"y{obs_var.id} = "
                + f"{{distribution={obs_var.error_model_dist_name}, "
                + f"prediction={obs_var.pred_name}, "
                + f"errorModel={obs_var.error_model_name}({', '.join(nonfixed_error_param_names_id)})}}"
            )
            FIT_data.append(obs_var.id)
            FIT_model.append(f"y{obs_var.id}")
        PARAMETER_items_.append(
            f"{name_pop} = "
            + f"{{value={obs_var.init_est_pop}, "
            + f"method={'MLE' if not obs_var.is_fixed else 'FIXED'}}}"
        )
        if not obs_var.is_fixed:
            PARAMETER_items_.append(
                f"{omega_name} = {{value={obs_var.init_est_sd}, method=MLE}}"
            )
        if obs_var.id not in hidden_obs_var_idxs:
            for error_param in obs_var.error_params:
                PARAMETER_items_.append(
                    f"{error_param.name}{obs_var.id} = "
                    + f"{{value={error_param.init_est}, "
                    + f"method={'FIXED' if error_param.is_fixed else 'MLE'}}}"
                )

    mlxtran_template = Template(
        textwrap.dedent(
            """\
            <DATAFILE>

            [FILEINFO]
            file={path='$csv_path'}
            delimiter = comma
            header={time, id, observation, observation_id, observation_type}

            [CONTENT]
            time = {use=time}
            id = {use=identifier}
            observation = {use=observation, yname={$CONTENT_observation_yname}, type={$CONTENT_observation_type}}
            observation_id = {use=observationtype}

            <MODEL>

            [INDIVIDUAL]
            input = {$INDIVIDUAL_input}

            DEFINITION:
            $INDIVIDUAL_DEFINITION

            [LONGITUDINAL]
            input = {$LONGITUDINAL_input}

            file = '$model_path'

            DEFINITION:
            $LONGITUDINAL_DEFINITION

            <FIT>
            data = {$FIT_data}
            model = {$FIT_model}

            <PARAMETER>
            $PARAMETER

            <MONOLIX>

            [TASKS]
            populationParameters()
            individualParameters(method = {conditionalMean, conditionalMode })

            [PLOTS]
            run = false
            plots = {indfits = {selected = true}, parameterdistribution = {selected = true}, obspred = {selected = true}, covariancemodeldiagnosis = {selected = true}, covariatemodeldiagnosis = {selected = true}, vpc = {selected = true}, residualsscatter = {selected = true}, residualsdistribution = {selected = true}, randomeffects = {selected = true}, saemresults = {selected = true}}

            [SETTINGS]
            GLOBAL:
            seed = 545331266
            exportpath = '$name'
            """
        )
    )
    mlxtran_text = mlxtran_template.substitute(
        csv_path=csv_path,
        CONTENT_observation_yname=", ".join(f"'{id}'" for id in FIT_data),
        CONTENT_observation_type=", ".join(["continuous"] * len(FIT_data)),
        INDIVIDUAL_input=", ".join(INDIVIDUAL_input),
        INDIVIDUAL_DEFINITION="\n".join(INDIV_DEFINITION_items_),
        LONGITUDINAL_input=", ".join(LONGITUDINAL_input),
        model_path=model_path,
        LONGITUDINAL_DEFINITION="\n".join(LONG_DEFINITION_items_),
        FIT_data=", ".join(f"'{id}'" for id in FIT_data),
        FIT_model=", ".join(FIT_model),
        PARAMETER="\n".join(PARAMETER_items_),
        name=name,
    )
    with open(f"{name}.mlxtran", "w") as f:
        f.write(mlxtran_text)


def run_experiment(mlxtran_path: str, mode="complete", n_threads=1):
    load_dotenv()
    if "MONOLIX_PATH" not in os.environ:
        raise LookupError("Couldn't find MONOLIX_PATH in .env file")
    cwd = Path.cwd()
    print(os.environ["MONOLIX_PATH"])
    subprocess.run(
        [
            os.environ["MONOLIX_PATH"],
            "--no-gui",
            "--mode",
            mode,
            "--thread",
            str(n_threads),
            "-p",
            str(cwd.joinpath(mlxtran_path)),
            "-o",
            str(cwd.joinpath(mlxtran_path.split(".")[0])),
        ]
    )


if __name__ == "__main__":
    model_params = [
        MlxParam(
            name="aaa",
            dist_name="normal",
            init_est_pop=1.1,
            init_est_sd=0.1,
        ),
        MlxParam(
            name="bb",
            dist_name="logNormal",
            init_est_pop=69,
            init_est_sd=0.69,
        ),
    ]

    obs_vars = [
        MlxObsVar(
            name="X",
            id=0,
            dist_name="logNormal",
            init_est_pop=100,
            init_est_sd=0.2,
            error_model_name="constant",
            error_model_dist_name="normal",
            error_params=[MlxErrorModelParam(name="a", init_est=10, is_fixed=False)],
        ),
        MlxObsVar(
            name="Y",
            id=1,
            dist_name="logNormal",
            init_est_pop=200,
            init_est_sd=0.2,
            error_model_name="combined1",
            error_model_dist_name="normal",
            error_params=[
                MlxErrorModelParam(name="a", init_est=10, is_fixed=False),
                MlxErrorModelParam(name="b", init_est=10, is_fixed=False),
                MlxErrorModelParam(name="c", init_est=1, is_fixed=True),
            ],
        ),
    ]

    generate_mlxtran_file(
        "test", "no_exist.csv", "no_exist.txt", model_params, obs_vars
    )

    # run_experiment("sahoo_et_al.mlxtran")
