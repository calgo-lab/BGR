from ._base import Experiment
from .depth_experiment import DepthExperiment
from .simple_horizon_classification_embed_geotmp_mlp import SimpleHorizonClassificationEmbeddingsGeotempMLP
from .simple_horizon_classification_embed_geotmp_mlp_tab_mlp import SimpleHorizonClassificationWithEmbeddingsGeotempsMLPTabMLP
from .simple_horizon_classification import SimpleHorizonClassification
from .simple_horizon_classification_embed_geotmp import SimpleHorizonClassificationEmbeddingsGeotemp
from .simple_horizon_classification_embed import SimpleHorizonClassificationEmbeddings

def get_experiment(experiment_type, training_args, target, dataprocessor) -> Experiment:    
    if experiment_type == "depth_experiment":
        return DepthExperiment(training_args, target, dataprocessor)
    elif experiment_type == "simple_horizon_classification_embed_geotmp_mlp":
        return SimpleHorizonClassificationEmbeddingsGeotempMLP(training_args, target, dataprocessor)
    elif experiment_type == "simple_horizon_classification_embed_geotmp_mlp_tab_mlp":
        return SimpleHorizonClassificationWithEmbeddingsGeotempsMLPTabMLP(training_args, target, dataprocessor)
    elif experiment_type == "simple_horizon_classification":
        return SimpleHorizonClassification(training_args, target, dataprocessor)
    elif experiment_type == "simple_horizon_classification_embed_geotmp":
        return SimpleHorizonClassificationEmbeddingsGeotemp(training_args, target, dataprocessor)
    elif experiment_type == "simple_horizon_classification_embed":
        return SimpleHorizonClassificationEmbeddings(training_args, target, dataprocessor)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
def get_experiment_hyperparameters(experiment_type) -> dict:
    if experiment_type == "depth_experiment":
        return DepthExperiment.get_experiment_hyperparameters()
    elif experiment_type == "simple_horizon_classification_embed_geotmp_mlp":
        return SimpleHorizonClassificationEmbeddingsGeotempMLP.get_experiment_hyperparameters()
    elif experiment_type == "simple_horizon_classification_embed_geotmp_mlp_tab_mlp":
        return SimpleHorizonClassificationWithEmbeddingsGeotempsMLPTabMLP.get_experiment_hyperparameters()
    elif experiment_type == "simple_horizon_classification":
        return SimpleHorizonClassification.get_experiment_hyperparameters()
    elif experiment_type == "simple_horizon_classification_embed_geotmp":
        return SimpleHorizonClassificationEmbeddingsGeotemp.get_experiment_hyperparameters()
    elif experiment_type == "simple_horizon_classification_embed":
        return SimpleHorizonClassificationEmbeddings.get_experiment_hyperparameters()
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")