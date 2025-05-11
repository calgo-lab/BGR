from ._base import Experiment
from .simple_depth.simple_depth_geotmp import SimpleDepthsGeotemps
from .simple_depth.simple_depth_geotmp_resnet import SimpleDepthsGeotempsResNet
from .simple_depth.simple_depth_geotmp_cross import SimpleDepthsGeotempsCrossAttention
from .simple_depth.simple_depth_geotmp_resnet_cross import SimpleDepthsGeotempsResNetCrossAttention
from .simple_depth.simple_depth_maskedresnet_lstm import SimpleDepthsMaskedResNetLSTM
from .simple_depth.simple_depth_maskedresnet_cross import SimpleDepthsMaskedResNetCrossAttention
from .simple_horizon.simple_horizon_classification_embed_geotmp_mlp import SimpleHorizonClassificationEmbeddingsGeotempMLP
from .simple_horizon.simple_horizon_classification_embed_geotmp_mlp_tab_mlp import SimpleHorizonClassificationWithEmbeddingsGeotempsMLPTabMLP
from .simple_horizon.simple_horizon_classification import SimpleHorizonClassification
from .simple_horizon.simple_horizon_classification_embed_geotmp import SimpleHorizonClassificationEmbeddingsGeotemp
from .simple_horizon.simple_horizon_classification_embed import SimpleHorizonClassificationEmbeddings
from .simple_horizon.simple_horizon_classification_lstm_embed_geotmp_mlp_tab_mlp import SimpleHorizonClassificationWithLSTMEmbeddingsGeotempsMLPTabMLP
from .simple_horizon.simple_horizon_classification_lstm_geotmp_mlp_tab_mlp import SimpleHorizonClassificationWithLSTMGeotempsMLPTabMLP
from .simple_horizon.simple_horizon_classification_lstm_embed_geotmp_mlp_tab_mlp_hybrid_loss import SimpleHorizonClassificationWithLSTMEmbeddingsGeotempsMLPTabMLPHybridLoss
from .simple_horizon.simple_horizon_classification_lstm_geotmp_mlp_tab_mlp_resnet import SimpleHorizonClassificationWithLSTMGeotempsMLPTabMLPResNet
from .simple_horizon.simple_horizon_classification_lstm_embed_geotmp_mlp_tab_mlp_resnet import SimpleHorizonClassificationWithLSTMEmbeddingsGeotempsMLPTabMLPResNet
from .simple_horizon.simple_horizon_classification_lstm_shortpath_geotmp_mlp_tab_mlp_resnet import SimpleHorizonClassificationWithLSTMShortPathGeotempsMLPTabMLPResNet
from .simple_tabulars.simple_tabulars_geotmp_resnet import SimpleTabularsGeotempsResNet
from .simple_tabulars.simple_tabulars_geotmp import SimpleTabularsGeotemps
from .end2end.end2end_lstm_embed import End2EndLSTMEmbed
from .end2end.end2end_lstm_resnet_embed import End2EndLSTMResNetEmbed
from .end2end.end2end_lstm import End2EndLSTM
from .end2end.end2end_lstm_resnet import End2EndLSTMResNet

def get_experiment(experiment_type, training_args, target, dataprocessor) -> Experiment:    
    if experiment_type == "simple_depths_geotmp":
        return SimpleDepthsGeotemps(training_args, target, dataprocessor)
    elif experiment_type == "simple_depths_geotmp_resnet":
        return SimpleDepthsGeotempsResNet(training_args, target, dataprocessor)
    elif experiment_type == "simple_depths_geotmp_cross":
        return SimpleDepthsGeotempsCrossAttention(training_args, target, dataprocessor)
    elif experiment_type == "simple_depths_geotmp_resnet_cross":
        return SimpleDepthsGeotempsResNetCrossAttention(training_args, target, dataprocessor)
    elif experiment_type == "simple_depths_maskedresnet_lstm":
        return SimpleDepthsMaskedResNetLSTM(training_args, target, dataprocessor)
    elif experiment_type == "simple_depths_maskedresnet_cross":
        return SimpleDepthsMaskedResNetCrossAttention(training_args, target, dataprocessor)
    elif experiment_type == "simple_horizon_classification_embed_geotmp_mlp":
        return SimpleHorizonClassificationEmbeddingsGeotempMLP(training_args, target, dataprocessor)
    elif experiment_type == "simple_horizon_classification_embed_geotmp_mlp_tab_mlp":
        return SimpleHorizonClassificationWithEmbeddingsGeotempsMLPTabMLP(training_args, target, dataprocessor)
    elif experiment_type == "simple_horizon_classification_lstm_embed_geotmp_mlp_tab_mlp":
        return SimpleHorizonClassificationWithLSTMEmbeddingsGeotempsMLPTabMLP(training_args, target, dataprocessor)
    elif experiment_type == "simple_horizon_classification_lstm_geotmp_mlp_tab_mlp":
        return SimpleHorizonClassificationWithLSTMGeotempsMLPTabMLP(training_args, target, dataprocessor)
    elif experiment_type == "simple_horizon_classification_lstm_embed_geotmp_mlp_tab_mlp_hybrid_loss":
        return SimpleHorizonClassificationWithLSTMEmbeddingsGeotempsMLPTabMLPHybridLoss(training_args, target, dataprocessor)
    elif experiment_type == "simple_horizon_classification":
        return SimpleHorizonClassification(training_args, target, dataprocessor)
    elif experiment_type == "simple_horizon_classification_embed_geotmp":
        return SimpleHorizonClassificationEmbeddingsGeotemp(training_args, target, dataprocessor)
    elif experiment_type == "simple_horizon_classification_embed":
        return SimpleHorizonClassificationEmbeddings(training_args, target, dataprocessor)
    elif experiment_type == "simple_horizon_classification_lstm_geotmp_mlp_tab_mlp_resnet":
        return SimpleHorizonClassificationWithLSTMGeotempsMLPTabMLPResNet(training_args, target, dataprocessor)
    elif experiment_type == "simple_horizon_classification_lstm_embed_geotmp_mlp_tab_mlp_resnet":
        return SimpleHorizonClassificationWithLSTMEmbeddingsGeotempsMLPTabMLPResNet(training_args, target, dataprocessor)
    elif experiment_type == "simple_horizon_classification_lstm_shortpath_geotmp_mlp_tab_mlp_resnet":
        return SimpleHorizonClassificationWithLSTMShortPathGeotempsMLPTabMLPResNet(training_args, target, dataprocessor)
    elif experiment_type == "simple_tabulars_geotmp_resnet":
        return SimpleTabularsGeotempsResNet(training_args, target, dataprocessor)
    elif experiment_type == "simple_tabulars_geotmp":
        return SimpleTabularsGeotemps(training_args, target, dataprocessor)
    elif experiment_type == "end2end_lstm_embed":
        return End2EndLSTMEmbed(training_args, target, dataprocessor)
    elif experiment_type == "end2end_lstm_resnet_embed":
        return End2EndLSTMResNetEmbed(training_args, target, dataprocessor)
    elif experiment_type == "end2end_lstm":
        return End2EndLSTM(training_args, target, dataprocessor)
    elif experiment_type == "end2end_lstm_resnet":
        return End2EndLSTMResNet(training_args, target, dataprocessor)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
def get_experiment_hyperparameters(experiment_type) -> dict:
    if experiment_type == "simple_depths_geotmp":
        return SimpleDepthsGeotemps.get_experiment_hyperparameters()
    elif experiment_type == "simple_depths_geotmp_resnet":
        return SimpleDepthsGeotempsResNet.get_experiment_hyperparameters()
    elif experiment_type == "simple_depths_geotmp_cross":
        return SimpleDepthsGeotempsCrossAttention.get_experiment_hyperparameters()
    elif experiment_type == "simple_depths_geotmp_resnet_cross":
        return SimpleDepthsGeotempsResNetCrossAttention.get_experiment_hyperparameters()
    elif experiment_type == "simple_depths_maskedresnet_lstm":
        return SimpleDepthsMaskedResNetLSTM.get_experiment_hyperparameters()
    elif experiment_type == "simple_depths_maskedresnet_cross":
        return SimpleDepthsMaskedResNetCrossAttention.get_experiment_hyperparameters()
    elif experiment_type == "simple_horizon_classification_embed_geotmp_mlp":
        return SimpleHorizonClassificationEmbeddingsGeotempMLP.get_experiment_hyperparameters()
    elif experiment_type == "simple_horizon_classification_embed_geotmp_mlp_tab_mlp":
        return SimpleHorizonClassificationWithEmbeddingsGeotempsMLPTabMLP.get_experiment_hyperparameters()
    elif experiment_type == "simple_horizon_classification_lstm_embed_geotmp_mlp_tab_mlp":
        return SimpleHorizonClassificationWithLSTMEmbeddingsGeotempsMLPTabMLP.get_experiment_hyperparameters()
    elif experiment_type == "simple_horizon_classification_lstm_geotmp_mlp_tab_mlp":
        return SimpleHorizonClassificationWithLSTMGeotempsMLPTabMLP.get_experiment_hyperparameters()
    elif experiment_type == "simple_horizon_classification_lstm_embed_geotmp_mlp_tab_mlp_hybrid_loss":
        return SimpleHorizonClassificationWithLSTMEmbeddingsGeotempsMLPTabMLPHybridLoss.get_experiment_hyperparameters()
    elif experiment_type == "simple_horizon_classification":
        return SimpleHorizonClassification.get_experiment_hyperparameters()
    elif experiment_type == "simple_horizon_classification_embed_geotmp":
        return SimpleHorizonClassificationEmbeddingsGeotemp.get_experiment_hyperparameters()
    elif experiment_type == "simple_horizon_classification_embed":
        return SimpleHorizonClassificationEmbeddings.get_experiment_hyperparameters()
    elif experiment_type == "simple_horizon_classification_lstm_geotmp_mlp_tab_mlp_resnet":
        return SimpleHorizonClassificationWithLSTMGeotempsMLPTabMLPResNet.get_experiment_hyperparameters()
    elif experiment_type == "simple_horizon_classification_lstm_embed_geotmp_mlp_tab_mlp_resnet":
        return SimpleHorizonClassificationWithLSTMEmbeddingsGeotempsMLPTabMLPResNet.get_experiment_hyperparameters()
    elif experiment_type == "simple_horizon_classification_lstm_shortpath_geotmp_mlp_tab_mlp_resnet":
        return SimpleHorizonClassificationWithLSTMShortPathGeotempsMLPTabMLPResNet.get_experiment_hyperparameters()
    elif experiment_type == "simple_tabulars_geotmp_resnet":
        return SimpleTabularsGeotempsResNet.get_experiment_hyperparameters()
    elif experiment_type == "simple_tabulars_geotmp":
        return SimpleTabularsGeotemps.get_experiment_hyperparameters()
    elif experiment_type == "end2end_lstm_embed":
        return End2EndLSTMEmbed.get_experiment_hyperparameters()
    elif experiment_type == "end2end_lstm_resnet_embed":
        return End2EndLSTMResNetEmbed.get_experiment_hyperparameters()
    elif experiment_type == "end2end_lstm":
        return End2EndLSTM.get_experiment_hyperparameters()
    elif experiment_type == "end2end_lstm_resnet":
        return End2EndLSTMResNet.get_experiment_hyperparameters()
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")