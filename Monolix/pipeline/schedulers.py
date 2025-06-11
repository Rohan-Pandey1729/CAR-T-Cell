"""General ways of choosing times to take observations at"""

def uniform_scheduler(first_day: int, last_day: int):
    """
    Returns a scheduler mapping a number n of observations
    to an arithmetic sequence of length n starting at first_day and ending
    at or before last_day
    """
    def scheduler(n_obs: int):
        if n_obs < 1:
            raise ValueError("n_obs must be at least 1")
        if n_obs == 1:
            return [first_day]
        obs_gap = (last_day - first_day) / (n_obs - 1)
        return [first_day + k * obs_gap for k in range(n_obs)]
    return scheduler
