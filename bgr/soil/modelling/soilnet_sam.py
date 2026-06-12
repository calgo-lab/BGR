from __future__ import annotations

"""SAM-backed SoilNet variants.

This module keeps the original SoilNet output contract while replacing the
ResNet image backbone and hard segment cropping with a SAM encoder plus soft
vertical segment masks.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bgr.soil.modelling.depth.depth_modules import LSTMDepthMarkerPredictorWithGuardrails
from bgr.soil.modelling.horizon.horizon_modules import HorizonLSTMEmbedder
from bgr.soil.modelling.tabulars.tabular_modules import LSTMTabularPredictor
from bgr.soil.utils import unpad_image_using_mask

try:
    from segment_anything import sam_model_registry
except Exception:  # pragma: no cover - optional dependency
    sam_model_registry = None


def _smooth_step(values: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "linear":
        return values.clamp(0.0, 1.0)
    if mode == "gaussian":
        return torch.exp(-0.5 * ((values - 1.0) / 0.35) ** 2).clamp(0.0, 1.0)

    return 0.5 * (1.0 - torch.cos(torch.pi * values.clamp(0.0, 1.0)))


class LowRankBilinearSegmentPool(nn.Module):
    """Compact bilinear pooling over masked segment regions."""

    def __init__(self, input_dim: int, output_dim: int, rank: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank

        self.left_projection = nn.Conv2d(input_dim, rank, kernel_size=1)
        self.right_projection = nn.Conv2d(input_dim, rank, kernel_size=1)
        self.output_projection = nn.Linear(rank, output_dim) if rank != output_dim else nn.Identity()

    def forward(self, feature_map: torch.Tensor, segment_masks: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = feature_map.shape
        segment_masks_2d = segment_masks.unsqueeze(-1).expand(-1, -1, -1, width)

        masked_feature_map = feature_map.unsqueeze(1) * segment_masks_2d.unsqueeze(2)
        flat_feature_map = masked_feature_map.view(batch_size * segment_masks.size(1), channels, height, width)
        flat_mask = segment_masks_2d.view(batch_size * segment_masks.size(1), 1, height, width)
        flat_mask_sum = flat_mask.sum(dim=(-2, -1)).clamp(min=1e-6)

        left_features = self.left_projection(flat_feature_map)
        right_features = self.right_projection(flat_feature_map)

        left_pooled = (left_features * flat_mask).sum(dim=(-2, -1)) / flat_mask_sum
        right_pooled = (right_features * flat_mask).sum(dim=(-2, -1)) / flat_mask_sum

        bilinear_features = left_pooled * right_pooled
        bilinear_features = self.output_projection(bilinear_features)
        return bilinear_features.view(batch_size, segment_masks.size(1), -1)


class SegmentContextEncoder(nn.Module):
    """Lightweight transformer encoder for cross-segment context."""

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_seq_len, input_dim))

    def forward(self, segment_features: torch.Tensor) -> torch.Tensor:
        sequence_length = segment_features.size(1)
        positional_embedding = self.positional_embedding[:, :sequence_length]
        return self.encoder(segment_features + positional_embedding)


def create_soft_segment_masks(
    depth_markers: torch.Tensor,
    feature_height: int,
    num_segments: int,
    overlap_pct: float = 0.10,
    stop_token: float = 1.0,
    smoothing: str = "cosine",
) -> torch.Tensor:
    """Create smooth vertical masks for each horizon segment."""
    if depth_markers.dim() != 2:
        raise ValueError(f"depth_markers must have shape (B, S), got {tuple(depth_markers.shape)}")

    batch_size = depth_markers.size(0)
    device = depth_markers.device
    row_positions = torch.arange(feature_height, device=device, dtype=depth_markers.dtype)

    segment_masks = []
    for batch_index in range(batch_size):
        row_depths = depth_markers[batch_index]
        valid_depths = row_depths[row_depths != stop_token].clamp(0.0, stop_token)

        if valid_depths.numel() == 0:
            segment_masks.append(torch.zeros((num_segments, feature_height), device=device, dtype=depth_markers.dtype))
            continue

        bounds = torch.cat([torch.zeros(1, device=device, dtype=depth_markers.dtype), valid_depths])
        valid_segments = min(bounds.numel() - 1, num_segments)

        sample_masks = torch.zeros((num_segments, feature_height), device=device, dtype=depth_markers.dtype)
        for segment_index in range(valid_segments):
            start = bounds[segment_index] * (feature_height - 1)
            end = bounds[segment_index + 1] * (feature_height - 1)
            if end <= start:
                continue

            segment_length = (end - start).clamp(min=1.0)
            overlap = torch.clamp(segment_length * overlap_pct, min=1.0)
            overlap = torch.minimum(overlap, segment_length / 2)
            overlap = overlap.clamp(min=1e-6)

            inside = (row_positions >= start) & (row_positions <= end)
            weights = torch.zeros_like(row_positions)

            lower_zone = inside & (row_positions <= start + overlap)
            if lower_zone.any():
                lower_t = (row_positions[lower_zone] - start) / overlap
                weights[lower_zone] = _smooth_step(lower_t, smoothing)

            upper_zone = inside & (row_positions >= end - overlap)
            if upper_zone.any():
                upper_t = (end - row_positions[upper_zone]) / overlap
                weights[upper_zone] = _smooth_step(upper_t, smoothing)

            middle_zone = inside & ~lower_zone & ~upper_zone
            weights[middle_zone] = 1.0

            sample_masks[segment_index] = weights

        mask_sum = sample_masks.sum(dim=0, keepdim=True).clamp(min=1e-6)
        sample_masks = sample_masks / mask_sum
        segment_masks.append(sample_masks)

    return torch.stack(segment_masks, dim=0)


class SAMImageEncoder(nn.Module):
    """Encodes a soil profile image with a pretrained SAM image encoder."""

    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint: Optional[str] = None,
        trainable_layers: int = 4,
        output_dim: int = 256,
        sam_input_size: int = 1024,
    ):
        super().__init__()

        if sam_model_registry is None:
            raise ImportError(
                "segment_anything is not installed. Install the SAM package to use SoilNetSAM."
            )
        if checkpoint is None:
            raise ValueError("A SAM checkpoint path is required to instantiate SAMImageEncoder.")

        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam_input_size = sam_input_size
        self.raw_feature_dim = 256
        self.output_dim = output_dim

        for parameter in self.sam.parameters():
            parameter.requires_grad = False

        image_encoder = self.sam.image_encoder
        if hasattr(image_encoder, "blocks"):
            for block in image_encoder.blocks[-trainable_layers:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

        if hasattr(image_encoder, "neck"):
            for parameter in image_encoder.neck.parameters():
                parameter.requires_grad = True

        if output_dim != self.raw_feature_dim:
            self.projection = nn.Conv2d(self.raw_feature_dim, output_dim, kernel_size=1)
        else:
            self.projection = nn.Identity()

    def _prepare_single_image(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        image = unpad_image_using_mask(image, mask).float()
        if image.numel() == 0:
            image = torch.zeros((3, self.sam_input_size, self.sam_input_size), device=mask.device)

        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.sam_input_size, self.sam_input_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if image.max() <= 1.5:
            image = image * 255.0

        return image

    def forward(self, padded_images: torch.Tensor, image_mask: torch.Tensor) -> torch.Tensor:
        prepared_images = [self._prepare_single_image(image, mask) for image, mask in zip(padded_images, image_mask)]
        prepared_images = torch.stack(prepared_images, dim=0)
        prepared_images = self.sam.preprocess(prepared_images)

        feature_map = self.sam.image_encoder(prepared_images)
        feature_map = self.projection(feature_map)
        return feature_map


class SoilNet_NoGeoTemp_SAM(nn.Module):
    """End-to-end model predicting depths, tabular features and horizon embeddings.

    Optional soft-cropping and simplified segment context (disabled by default)
    to make behavior comparable to non-SAM variants.
    """

    def __init__(
        self,
        image_encoder_output_dim: int = 256,
        max_seq_len: int = 8,
        stop_token: float = 1.0,
        depth_rnn_hidden_dim: int = 256,
        tabular_output_dim_dict: dict[str, int] | None = None,
        segment_encoder_output_dim: int = 256,
        tab_rnn_hidden_dim: int = 1024,
        tab_num_lstm_layers: int = 2,
        segments_tabular_output_dim: int = 64,
        embedding_dim: int = 61,
        teacher_forcing_stop_epoch: int = 5,
        teacher_forcing_approach: str = "linear",
        sam_model_type: str = "vit_b",
        sam_checkpoint: Optional[str] = None,
        sam_trainable_layers: int = 4,
        sam_input_size: int = 1024,
        use_soft_crop: bool = True,
        segment_overlap_pct: float = 0.10,
        boundary_smoothing: str = "cosine",
        segment_pooling_mode: str = "masked_avg",
        bilinear_rank: int = 128,
        segment_attention_layers: int = 2,
        segment_attention_heads: int = 4,
        segment_attention_dropout: float = 0.1,
    ):
        super().__init__()

        self.image_encoder_output_dim = image_encoder_output_dim
        self.max_seq_len = max_seq_len
        self.stop_token = stop_token
        self.depth_rnn_hidden_dim = depth_rnn_hidden_dim
        self.tabular_output_dim_dict = tabular_output_dim_dict or {}
        self.segment_encoder_output_dim = segment_encoder_output_dim
        self.tab_rnn_hidden_dim = tab_rnn_hidden_dim
        self.tab_num_lstm_layers = tab_num_lstm_layers
        self.segments_tabular_output_dim = segments_tabular_output_dim
        self.embedding_dim = embedding_dim
        self.sam_input_size = sam_input_size
        self.segment_overlap_pct = segment_overlap_pct
        self.boundary_smoothing = boundary_smoothing
        self.segment_pooling_mode = segment_pooling_mode
        self.bilinear_rank = bilinear_rank
        self.segment_attention_layers = segment_attention_layers
        self.segment_attention_heads = segment_attention_heads
        self.segment_attention_dropout = segment_attention_dropout

        self.image_encoder = SAMImageEncoder(
            model_type=sam_model_type,
            checkpoint=sam_checkpoint,
            trainable_layers=sam_trainable_layers,
            output_dim=self.image_encoder_output_dim,
            sam_input_size=sam_input_size,
        )

        # Whether to use soft (smooth, overlapping) segment masks or hard/non-overlapping masks
        self.use_soft_crop = use_soft_crop

        self.depth_marker_predictor = LSTMDepthMarkerPredictorWithGuardrails(
            self.image_encoder_output_dim,
            self.depth_rnn_hidden_dim,
            self.max_seq_len,
            self.stop_token,
        )

        if self.segment_pooling_mode == "bilinear":
            self.segment_pool = LowRankBilinearSegmentPool(
                input_dim=self.image_encoder_output_dim,
                output_dim=self.segment_encoder_output_dim,
                rank=self.bilinear_rank,
            )
        else:
            self.segment_pool = None

        # Cross-attention / transformer-based segment context removed for
        # ablation simplicity — use identity to keep features comparable.
        self.segment_context_encoder = nn.Identity()

        self.tabular_predictors = nn.ModuleDict()
        predicted_tabular_input_dim = 0
        for key, output_dim in self.tabular_output_dim_dict.items():
            self.tabular_predictors[key] = LSTMTabularPredictor(
                input_dim=self.segment_encoder_output_dim,
                output_dim=output_dim,
                hidden_dim=self.tab_rnn_hidden_dim,
                num_lstm_layers=self.tab_num_lstm_layers,
            )
            predicted_tabular_input_dim += output_dim

        self.predicted_segments_tabular_encoder = nn.Sequential(
            nn.Linear(predicted_tabular_input_dim, self.segments_tabular_output_dim),
            nn.ReLU(),
        )

        self.true_segments_tabular_encoder = nn.Sequential(
            nn.LazyLinear(self.segments_tabular_output_dim),
            nn.ReLU(),
        )

        self.horizon_embedder = HorizonLSTMEmbedder(
            input_dim=self.segment_encoder_output_dim + self.segments_tabular_output_dim,
            output_dim=self.embedding_dim,
            hidden_dim=256,
        )

        self.epoch = 0
        self.teacher_forcing_probs = {
            epoch: 1 - ((epoch - 1) / teacher_forcing_stop_epoch) if teacher_forcing_approach == "linear" else 1.0
            for epoch in range(1, teacher_forcing_stop_epoch + 1)
        }
        self.teacher_forcing_stop_epoch = teacher_forcing_stop_epoch

    def train(self, mode: bool = True):
        if mode:
            self.epoch += 1
        return super().train(mode)

    def teacher_forcing_decision(self, probability: float) -> bool:
        if self.training:
            return torch.rand(1).item() < probability
        return False

    def _segment_pool(self, feature_map: torch.Tensor, segment_masks: torch.Tensor) -> torch.Tensor:
        if self.segment_pool is not None:
            return self.segment_pool(feature_map, segment_masks)

        batch_size, channels, height, width = feature_map.shape
        segment_masks_2d = segment_masks.unsqueeze(-1).expand(-1, -1, -1, width)
        masked_features = feature_map.unsqueeze(1) * segment_masks_2d.unsqueeze(2)
        feature_sum = masked_features.sum(dim=(-2, -1))
        mask_sum = segment_masks_2d.sum(dim=(-2, -1)).unsqueeze(-1).clamp(min=1e-6)
        return feature_sum / mask_sum

    def forward(
        self,
        padded_image: torch.Tensor,
        image_mask: torch.Tensor,
        true_padded_depths: Optional[torch.Tensor] = None,
        true_tabular_features: Optional[torch.Tensor] = None,
        use_trues_during_inference: bool = False,
    ):
        effective_num_segments = self.max_seq_len
        if true_tabular_features is not None and true_tabular_features.numel() > 0:
            effective_num_segments = true_tabular_features.size(1)
        elif true_padded_depths is not None and true_padded_depths.numel() > 0:
            effective_num_segments = true_padded_depths.size(1)

        if not self.training and use_trues_during_inference:
            teacher_forcing_decision = True
        elif self.epoch < self.teacher_forcing_stop_epoch + 1:
            teacher_forcing_decision = self.teacher_forcing_decision(self.teacher_forcing_probs[self.epoch])
        else:
            teacher_forcing_decision = False

        feature_map = self.image_encoder(padded_image, image_mask)
        image_features = feature_map.mean(dim=(2, 3))
        depth_markers = self.depth_marker_predictor(image_features)[:, :effective_num_segments]

        if teacher_forcing_decision and true_padded_depths is not None:
            processed_depth_markers = true_padded_depths[:, :effective_num_segments]
        else:
            processed_depth_markers = depth_markers

        # Create segment masks. If soft cropping is disabled, create a
        # non-overlapping hard mask (derived from the soft masks) so pooling
        # behavior is comparable but without smooth overlaps.
        segment_masks = create_soft_segment_masks(
            processed_depth_markers,
            feature_height=feature_map.shape[-2],
            num_segments=effective_num_segments,
            overlap_pct=self.segment_overlap_pct if self.use_soft_crop else 0.0,
            stop_token=self.stop_token,
            smoothing=self.boundary_smoothing if self.use_soft_crop else "linear",
        )
        if not self.use_soft_crop:
            # Convert to hard one-hot masks per row to avoid overlaps
            # segment_masks: (B, S, H) -> argmax over S -> one_hot -> (B, S, H)
            arg = segment_masks.argmax(dim=1)  # (B, H)
            one_hot = F.one_hot(arg, num_classes=segment_masks.size(1)).permute(0, 2, 1).float()
            mask_sum = one_hot.sum(dim=1, keepdim=True).clamp(min=1e-6)
            segment_masks = one_hot / mask_sum
        segment_features = self._segment_pool(feature_map, segment_masks)
        segment_features = self.segment_context_encoder(segment_features)

        tabular_predictions = {}
        for key, predictor in self.tabular_predictors.items():
            tabular_predictions[key] = predictor(segment_features)[:, :effective_num_segments]

        if teacher_forcing_decision and true_tabular_features is not None:
            processed_tabular_features = true_tabular_features[:, :effective_num_segments].view(padded_image.size(0), effective_num_segments, -1)
            tabular_embeddings = self.true_segments_tabular_encoder(processed_tabular_features)
        else:
            processed_tabular_features = torch.cat([tabular_predictions[key] for key in self.tabular_predictors.keys()], dim=-1)
            tabular_embeddings = self.predicted_segments_tabular_encoder(processed_tabular_features)
        segment_tabular_features = torch.cat([segment_features, tabular_embeddings], dim=-1)

        batch_size, num_segments, _ = segment_tabular_features.shape
        horizon_embeddings = self.horizon_embedder(segment_tabular_features.view(batch_size * num_segments, -1), num_segments)

        return depth_markers, tabular_predictions, horizon_embeddings
