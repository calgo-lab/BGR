from ._base import Experiment
from .depth_experiment import DepthExperiment

def get_experiment(experiment_type, training_args, target, dataprocessor) -> Experiment:    
    if experiment_type == "depth_experiment":
        return DepthExperiment(training_args, target, dataprocessor)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")