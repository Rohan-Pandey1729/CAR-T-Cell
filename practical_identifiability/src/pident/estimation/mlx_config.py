"""
Monolix configuration management and .mlxtran file generation.

Provides dataclasses for configuring Monolix parameter estimation and a
template system for generating Monolix .mlxtran configuration files.

Tested with Monolix version 2024R1. Output formats from other versions
may not be compatible.
"""

import textwrap
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Literal

from pident.common.models import ODEModel

MlxParamDistName = Literal["normal", "logNormal"]
MlxResidualErrorDistName = Literal["normal", "logNormal"]
MlxResidualErrorModelType = Literal[
    "constant", "proportional", "combined1", "combined2"
]


@dataclass
class MlxParamConfig:
    """
    Configuration for a model parameter (excluding name, which is a dict key).
    - dist_name: Distribution name for the population distribution. Must be "normal" or "logNormal".
    - init_est_pop: Initial estimate for the population parameter median.
    - init_est_omega: Initial estimate for the omega parameter for this model parameter.
    - is_fixed: If True, this parameter is fixed during estimation. Otherwise, it is estimated via MLE.
    """

    dist_name: MlxParamDistName
    init_est_pop: float
    init_est_omega: float
    is_fixed: bool


@dataclass
class MlxResidualErrorParamConfig:
    """
    Configuration for a residual error parameter.
    - name: Name of the residual error parameter.
    - init_est: Initial estimate for the residual error parameter.
    - is_fixed: If True, this parameter is fixed during estimation. Otherwise, it is estimated via MLE.
    """

    name: str
    init_est: float
    is_fixed: bool


@dataclass
class MlxObsVarConfig:
    """
    Configuration data for an observation variable (excluding name (dict key) and id (inferred)).
    - pred_name: Name of the prediction variable for this observation variable in the model file.
    - ic_config: Configuration for the initial condition parameter for this observation variable.
    - residual_error_model_type: Type of residual error model to use for this observation variable.
        Must be one of "constant", "proportional", "combined1", or "combined2".
    - residual_error_dist_name: Distribution name for the residual error model. Must be "normal" or "logNormal".
    - residual_error_params: List of configurations for each residual error parameter for this observation variable.
        Must be consistent with residual_error_model_type.
    """

    pred_name: str
    ic_config: MlxParamConfig
    residual_error_model_type: MlxResidualErrorModelType
    residual_error_dist_name: MlxResidualErrorDistName
    residual_error_params: list[MlxResidualErrorParamConfig]


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
        exportpath = '$exportpath'
        """
    )
)


def generate_mlxtran_file(
    output_path: Path,
    data_csv_path: Path,
    mlx_model_path: Path,
    ode_model: ODEModel,
    model_params: dict[str, MlxParamConfig],
    obs_vars: dict[str, MlxObsVarConfig],
    hidden_obs_var_names: list[str] | None = None,
) -> None:
    """
    Generate an .mlxtran configuration file from ODE model parameters and observation variables.

    Args:
        output_path: Path where the .mlxtran file will be written
        data_csv_path: Path to the input data CSV file
        mlx_model_path: Path to the MONOLIX model file (e.g., owens_bozic.txt)
        ode_model: ODEModel instance providing canonical parameter and observation variable names
        model_params: Dict mapping parameter names to config values. All keys must match ode_model.param_names.
        obs_vars: Dict mapping observation variable names to config values. All keys must match ode_model.obs_var_names.
        hidden_obs_var_names: Optional list of observation variable names to exclude from the fit.
            All names must be valid observation variables in ode_model.obs_var_names.

    Raises:
        ValueError: If output_path does not end with .mlxtran, or if model_params/obs_vars dicts
            have missing/extra keys compared to ode_model properties, or if hidden_obs_var_names
            contains invalid variable names.

    Note:
        Observation variable IDs are assigned by enumerating ode_model.obs_var_names. The CSV data file's
        'observation_id' column must use sequential integer values (0, 1, 2, ...) corresponding to this order.
    """
    if not output_path.name.endswith(".mlxtran"):
        raise ValueError(
            f"Expected output_path to be an .mlxtran file, but got {output_path}"
        )

    # Validate model_params against ode_model
    missing_params = set(ode_model.param_names) - set(model_params.keys())
    extra_params = set(model_params.keys()) - set(ode_model.param_names)
    if missing_params:
        raise ValueError(
            f"model_params dict is missing keys for ODE model parameters: {missing_params}"
        )
    if extra_params:
        raise ValueError(
            f"model_params dict has extra keys not in ODE model parameters: {extra_params}"
        )

    # Validate obs_vars against ode_model
    missing_obs_vars = set(ode_model.obs_var_names) - set(obs_vars.keys())
    extra_obs_vars = set(obs_vars.keys()) - set(ode_model.obs_var_names)
    if missing_obs_vars:
        raise ValueError(
            f"obs_vars dict is missing keys for ODE model observation variables: {missing_obs_vars}"
        )
    if extra_obs_vars:
        raise ValueError(
            f"obs_vars dict has extra keys not in ODE model observation variables: {extra_obs_vars}"
        )

    # Validate and normalize hidden_obs_var_names
    if hidden_obs_var_names is None:
        hidden_obs_var_names = []
    invalid_hidden = set(hidden_obs_var_names) - set(ode_model.obs_var_names)
    if invalid_hidden:
        raise ValueError(
            f"hidden_obs_var_names contains invalid observation variable names: {invalid_hidden}"
        )

    # everything that needs to be generated
    INDIVIDUAL_input = []
    INDIV_DEFINITION_items_ = []
    LONGITUDINAL_input = []
    LONG_DEFINITION_items_ = []
    FIT_data = []  # also use for CONTENT.observation.yname
    FIT_model = []
    PARAMETER_items_ = []

    for param_name in ode_model.param_names:
        param_config = model_params[param_name]
        name_pop = f"{param_name}_pop"
        omega_name = f"omega_{param_name}"

        INDIVIDUAL_input.append(name_pop)
        INDIVIDUAL_input.append(omega_name)
        INDIV_DEFINITION_items_.append(
            f"{param_name} = "
            + f"{{distribution={param_config.dist_name}, "
            + f"typical={name_pop}, "
            + (f"sd={omega_name}}}" if not param_config.is_fixed else "no-variability}")
        )
        PARAMETER_items_.append(
            f"{name_pop} = "
            + f"{{value={param_config.init_est_pop}, "
            + f"method={'MLE' if not param_config.is_fixed else 'FIXED'}}}"
        )
        if not param_config.is_fixed:
            PARAMETER_items_.append(
                f"{omega_name} = {{value={param_config.init_est_omega}, method=MLE}}"
            )

    for obs_var_id, obs_var_name in enumerate(ode_model.obs_var_names):
        obs_var_config = obs_vars[obs_var_name]
        name0 = f"{obs_var_name}0"
        name_pop = f"{name0}_pop"
        omega_name = f"omega_{name0}"
        nonfixed_error_param_names_id = [
            f"{error_param.name}{obs_var_id}"
            for error_param in obs_var_config.residual_error_params
            if not error_param.is_fixed
        ]

        INDIVIDUAL_input.append(name_pop)
        INDIVIDUAL_input.append(omega_name)
        INDIV_DEFINITION_items_.append(
            f"{name0} = "
            + f"{{distribution={obs_var_config.ic_config.dist_name}, "
            + f"typical={name_pop}, "
            + (
                f"sd={omega_name}}}"
                if not obs_var_config.ic_config.is_fixed
                else "no-variability}"
            )
        )
        if obs_var_name not in hidden_obs_var_names:
            LONGITUDINAL_input.extend(nonfixed_error_param_names_id)
            LONG_DEFINITION_items_.append(
                f"y{obs_var_id} = "
                + f"{{distribution={obs_var_config.residual_error_dist_name}, "
                + f"prediction={obs_var_config.pred_name}, "
                + f"errorModel={obs_var_config.residual_error_model_type}({', '.join(nonfixed_error_param_names_id)})}}"
            )
            FIT_data.append(obs_var_id)
            FIT_model.append(f"y{obs_var_id}")
        PARAMETER_items_.append(
            f"{name_pop} = "
            + f"{{value={obs_var_config.ic_config.init_est_pop}, "
            + f"method={'MLE' if not obs_var_config.ic_config.is_fixed else 'FIXED'}}}"
        )
        if not obs_var_config.ic_config.is_fixed:
            PARAMETER_items_.append(
                f"{omega_name} = {{value={obs_var_config.ic_config.init_est_omega}, method=MLE}}"
            )
        if obs_var_name not in hidden_obs_var_names:
            for error_param in obs_var_config.residual_error_params:
                PARAMETER_items_.append(
                    f"{error_param.name}{obs_var_id} = "
                    + f"{{value={error_param.init_est}, "
                    + f"method={'FIXED' if error_param.is_fixed else 'MLE'}}}"
                )

    mlxtran_text = mlxtran_template.substitute(
        csv_path=str(data_csv_path.resolve()),
        CONTENT_observation_yname=", ".join(f"'{id}'" for id in FIT_data),
        CONTENT_observation_type=", ".join(["continuous"] * len(FIT_data)),
        INDIVIDUAL_input=", ".join(INDIVIDUAL_input),
        INDIVIDUAL_DEFINITION="\n".join(INDIV_DEFINITION_items_),
        LONGITUDINAL_input=", ".join(LONGITUDINAL_input),
        model_path=str(mlx_model_path.resolve()),
        LONGITUDINAL_DEFINITION="\n".join(LONG_DEFINITION_items_),
        FIT_data=", ".join(f"'{id}'" for id in FIT_data),
        FIT_model=", ".join(FIT_model),
        PARAMETER="\n".join(PARAMETER_items_),
        exportpath=output_path.name.removesuffix(".mlxtran"),
    )
    with open(output_path, "w") as f:
        f.write(mlxtran_text)
