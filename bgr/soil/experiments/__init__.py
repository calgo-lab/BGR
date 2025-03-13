from ._base import Experiment
from .depth_experiment import DepthExperiment
from .simple_horizon_classification import SimpleHorizonClassificationExperiment
from .simple_horizon_classification_with_tabulars import SimpleHorizonClassificationWithTabularsExperiment

def get_experiment(experiment_type, training_args, target, dataprocessor) -> Experiment:    
    if experiment_type == "depth_experiment":
        return DepthExperiment(training_args, target, dataprocessor)
    elif experiment_type == "simple_horizon_classification":
        return SimpleHorizonClassificationExperiment(training_args, target, dataprocessor)
    elif experiment_type == "simple_horizon_classification_with_tabulars":
        return SimpleHorizonClassificationWithTabularsExperiment(training_args, target, dataprocessor)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")