from typing import TypedDict

from numpy.typing import ArrayLike
from pident.common.models import GroundTruthFunc


class Timeseries(TypedDict):
    t: ArrayLike
    y: ArrayLike


def subsample_trajectories(
    ground_truth_func: GroundTruthFunc, schedules: dict[str, ArrayLike]
) -> dict[str, Timeseries]:
    """
    Subsamples trajectories for each observation variable from the
    given ground truth represented by `ground_truth_func`.

    `schedules` should be a dict from observation variable names to
    an ndarray of times to include an observation of that variable at.

    The return value is a dict from observation variable names to
    dicts, each with a key "t" for the times at which observations
    were collected for that variable and a key "y" for the observation values
    at those times.

    Note: ground truths only support times in a certain interval, and
    it is the caller's responsibility to ensure that scheduled times
    are within the supported interval.
    """
    # this probably isn't the most efficient way to do it
    res: dict[str, Timeseries] = {}
    for obs_var, schedule in schedules.items():
        y_vals = ground_truth_func(schedule)[obs_var]
        res[obs_var] = {"t": schedule, "y": y_vals}
    return res
