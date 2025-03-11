from ._base import Experiment

def get_experiment(experiment_type: str, target: str) -> Experiment:
    """
    Returns the experiment class for the given experiment type.

    Args:
        experiment_type (str): The type of the experiment.

    Returns:
        Experiment: The experiment class for the given experiment type.
    """
    
    if experiment_type == "TODO":
        raise NotImplementedError("Experiments not yet implemented.")
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")