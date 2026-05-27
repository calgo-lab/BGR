from __future__ import annotations

"""End-to-end SAM experiment without geotemporal inputs."""

import os

import torchvision.transforms as transforms

from bgr.soil.experiments.end2end.end2end_lstm_resnet_embed_nogeotemps import (
    End2EndLSTMResNetEmbed_NoGeoTemps,
)
from bgr.soil.modelling.soilnet_sam import SoilNet_NoGeoTemp_SAM


class End2EndSAMEmbed_NoGeoTemps(End2EndLSTMResNetEmbed_NoGeoTemps):
    """No-geotemp end-to-end SoilNet experiment using SAM + soft masks."""

    def __init__(self, training_args, target, dataprocessor):
        super().__init__(training_args, target, dataprocessor)
        self.image_normalization = transforms.Compose([transforms.ToTensor()])
        self.hyperparameters = self.get_experiment_hyperparameters()
        self.hyperparameters.update(training_args.hyperparameters)

    def get_model(self):
        sam_checkpoint = self.hyperparameters.get("sam_checkpoint") or os.environ.get("SAM_CHECKPOINT")
        if not sam_checkpoint:
            raise ValueError(
                "SAM checkpoint not configured. Set hyperparameters['sam_checkpoint'] or the SAM_CHECKPOINT environment variable."
            )

        return SoilNet_NoGeoTemp_SAM(
            image_encoder_output_dim=self.hyperparameters["image_encoder_output_dim"],
            max_seq_len=self.hyperparameters["max_seq_len"],
            stop_token=self.hyperparameters["stop_token"],
            depth_rnn_hidden_dim=self.hyperparameters["depth_rnn_hidden_dim"],
            tabular_output_dim_dict=self.tabulars_output_dim_dict,
            segment_encoder_output_dim=self.hyperparameters["segment_encoder_output_dim"],
            tab_rnn_hidden_dim=self.hyperparameters["tab_rnn_hidden_dim"],
            tab_num_lstm_layers=self.hyperparameters["tab_num_lstm_layers"],
            segments_tabular_output_dim=self.hyperparameters["segments_tabular_output_dim"],
            embedding_dim=self.dataprocessor.embeddings_dict["embedding"].shape[1],
            teacher_forcing_stop_epoch=self.hyperparameters["teacher_forcing_stop_epoch"],
            teacher_forcing_approach=self.hyperparameters["teacher_forcing_approach"],
            sam_model_type=self.hyperparameters["sam_model_type"],
            sam_checkpoint=sam_checkpoint,
            sam_trainable_layers=self.hyperparameters["sam_trainable_layers"],
            sam_input_size=self.hyperparameters["sam_input_size"],
            segment_overlap_pct=self.hyperparameters["segment_overlap_pct"],
            boundary_smoothing=self.hyperparameters["boundary_smoothing"],
        )

    @staticmethod
    def get_experiment_hyperparameters() -> dict:
        return {
            "sam_model_type": "vit_b",
            "sam_checkpoint": "",
            "sam_trainable_layers": 4,
            "sam_input_size": 1024,
            "image_encoder_output_dim": 256,
            "max_seq_len": 10,
            "stop_token": 1.0,
            "depth_rnn_hidden_dim": 256,
            "segment_overlap_pct": 0.10,
            "boundary_smoothing": "cosine",
            "segment_encoder_output_dim": 256,
            "tab_rnn_hidden_dim": 1024,
            "tab_num_lstm_layers": 2,
            "segments_tabular_output_dim": 64,
            "teacher_forcing_stop_epoch": 5,
            "teacher_forcing_approach": "linear",
        }
