"""
End2End SoilNet experiment with NoGeoTemp using softcropping with SoftCroppedSegmentEncoder.
Tests different visual backbones (ResNet18, ResNet50, DINOv2 variants).
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bgr.soil.training_args import TrainingArgs

from bgr.soil.experiments.end2end.end2end_lstm_resnet_embed_nogeotemps import (
    End2EndLSTMResNetEmbed_NoGeoTemps
)

logger = logging.getLogger(__name__)


class End2EndSoftcropNoGeoTemp(End2EndLSTMResNetEmbed_NoGeoTemps):
    """
    No-geotemp end-to-end SoilNet experiment using softcropping with SoftCroppedSegmentEncoder.

    Supports multiple visual backbones:
    - resnet18, resnet50 (CNN)
    - dinov2_vits14, dinov2_vitb14, dinov2_vitl14 (ViT self-supervised)

    Uses bilinear pooling over masked feature maps for segment representation.

    Inherits all training/evaluation logic from End2EndLSTMResNetEmbed_NoGeoTemp.
    """

    def __init__(
        self,
        training_args: 'TrainingArgs',
        target: str,
        dataprocessor
    ):
        super().__init__(training_args, target, dataprocessor)

        self.hyperparameters = self.get_experiment_hyperparameters()
        self.hyperparameters.update(training_args.hyperparameters)

    def get_model(self):
        h = self.hyperparameters

        return self._create_model(
            h=h,
            image_encoder_output_dim=h.get('image_encoder_output_dim', 512),
            max_seq_len=h.get('max_seq_len', 8),
            stop_token=h.get('stop_token', 1.0),
            depth_rnn_hidden_dim=h.get('depth_rnn_hidden_dim', 256),
            img_patch_size=h.get('img_patch_size', 512),
            segment_encoder_output_dim=h.get('segment_encoder_output_dim', 512),
            tabular_output_dim_dict=self.tabulars_output_dim_dict,
            tab_rnn_hidden_dim=h.get('tab_rnn_hidden_dim', 1024),
            tab_num_lstm_layers=h.get('tab_num_lstm_layers', 2),
            segments_tabular_output_dim=h.get('segments_tabular_output_dim', 256),
            embedding_dim=self.label_embeddings_tensor.size(1),
            teacher_forcing_stop_epoch=h.get('teacher_forcing_stop_epoch', 5),
            teacher_forcing_approach=h.get('teacher_forcing_approach', 'linear'),
        )

    def _create_model(self, h, **model_kwargs):
        from bgr.soil.modelling.soilnet import SoilNet_NoGeoTemp_LSTM

        return SoilNet_NoGeoTemp_LSTM(
            image_encoder_output_dim=model_kwargs['image_encoder_output_dim'],
            max_seq_len=model_kwargs['max_seq_len'],
            stop_token=model_kwargs['stop_token'],
            depth_rnn_hidden_dim=model_kwargs['depth_rnn_hidden_dim'],
            img_patch_size=model_kwargs['img_patch_size'],
            segment_encoding_mode=h.get('segment_encoding_mode', 'softcropping'),
            segment_encoder_output_dim=model_kwargs['segment_encoder_output_dim'],
            segment_encoder_backbone=h.get('segment_encoder_backbone', 'resnet18'),
            segment_encoder_train_backbone=h.get('segment_encoder_train_backbone', False),
            segment_encoder_feed_mask_to_backbone=h.get('segment_encoder_feed_mask_to_backbone', False),
            tabular_output_dim_dict=model_kwargs['tabular_output_dim_dict'],
            tab_rnn_hidden_dim=model_kwargs['tab_rnn_hidden_dim'],
            tab_num_lstm_layers=model_kwargs['tab_num_lstm_layers'],
            segments_tabular_output_dim=model_kwargs['segments_tabular_output_dim'],
            embedding_dim=model_kwargs['embedding_dim'],
            teacher_forcing_stop_epoch=model_kwargs['teacher_forcing_stop_epoch'],
            teacher_forcing_approach=model_kwargs['teacher_forcing_approach'],
        )

    @staticmethod
    def get_experiment_hyperparameters() -> dict:
        base_h = End2EndLSTMResNetEmbed_NoGeoTemps.get_experiment_hyperparameters()

        softcrop_h = {
            'segment_encoding_mode': 'softcropping',
            'segment_encoder_backbone': 'resnet18',
            'segment_encoder_train_backbone': False,
            'segment_encoder_feed_mask_to_backbone': False,
        }

        base_h.update(softcrop_h)

        keys_to_remove = [
            'num_patches_per_segment',
            'segment_random_patch_size',
        ]
        for key in keys_to_remove:
            base_h.pop(key, None)

        return base_h