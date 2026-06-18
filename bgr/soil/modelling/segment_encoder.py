import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class ResNetSpatialBackbone(nn.Module):
    """
    Wraps a ResNet so self.backbone(x) returns spatial feature maps (B, D, H', W').
    Unwraps the dict returned by create_feature_extractor internally.
    """
    def __init__(self, base_model):
        super().__init__()
        self.feature_extractor = create_feature_extractor(
            base_model, return_nodes={'layer4': 'features'}
        )

    def forward(self, x):
        return self.feature_extractor(x)['features']  # (B, D, H', W')


class DINOv2SpatialBackbone(nn.Module):
    """
    Wraps DINOv2 so self.backbone(x) returns spatial feature maps (B, D, H', W'),
    consistent with ResNetSpatialBackbone output format.
    Reshapes patch tokens (B, N, D) → (B, D, h, w) using input spatial dimensions.
    Handles non-divisible inputs by zero-pad to the next multiple of patch_size.
    """
    def __init__(self, dino_model):
        super().__init__()
        self.dino = dino_model
        self.patch_size = dino_model.patch_size

    def forward(self, x):
        B, _, H, W = x.shape

        H_padded = ((H + self.patch_size - 1) // self.patch_size) * self.patch_size
        W_padded = ((W + self.patch_size - 1) // self.patch_size) * self.patch_size

        if H_padded != H or W_padded != W:
            assert H_padded >= self.patch_size and W_padded >= self.patch_size, (
                f"Input image ({H}x{W}) is smaller than the patch size "
                f"({self.patch_size}). DinoV2 requires H and W >= patch_size."
            )
            rgb = x[:, :3]
            rgb_padded = F.pad(rgb, (0, W_padded - W, 0, H_padded - H), mode='constant', value=0)

            mask = x[:, 3:4]
            mask_zeros = torch.zeros(B, 1, H_padded, W_padded, dtype=x.dtype, device=x.device)
            mask_zeros[:, :, :H, :W] = mask

            x = torch.cat([rgb_padded, mask_zeros], dim=1)

        out = self.dino.forward_features(x)
        tokens = out['x_norm_patchtokens']
        h = H_padded // self.patch_size
        w = W_padded // self.patch_size
        return tokens.permute(0, 2, 1).view(B, -1, h, w)


class SoftCroppedSegmentEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        output_dim: int = 512,
        bilinear_dim: int = 128,
        train_backbone: bool = False,
        feed_mask_to_backbone: bool = False,
    ):
        """
        Parameters
        ----------
        backbone : str
            'resnet18' | 'resnet50' | 'dinov2_vits14' | 'dinov2_vitb14' | 'dinov2_vitl14'
        pretrained : bool
            Load pretrained weights.
        output_dim : int
            Final embedding dimension.
        bilinear_dim : int
            Projection dim before bilinear pool — output is bilinear_dim²
            before the final projection. 128 → 16384 intermediate dim.
        train_backbone : bool
            If True, unfreeze backbone weights for fine-tuning.
            If False (default), backbone is frozen and only head layers train.
        feed_mask_to_backbone : bool
            If True (default), append mask as 4th input channel (Option A).
            The backbone processes the mask alongside RGB during feature extraction.
            If False, mask is only used in the bilinear pool (Option C).
            When False, the backbone uses standard 3-channel pretrained weights
            (no 4th-channel expansion). Safe to combine with train_backbone=True.
        """
        super().__init__()
        self.bilinear_dim = bilinear_dim
        self.feed_mask_to_backbone = feed_mask_to_backbone

        if backbone == 'resnet18':
            base = models.resnet18(
                weights=models.resnet.ResNet18_Weights.DEFAULT if pretrained else None
            )
            backbone_out_dim = base.fc.in_features           # 512
            # Expand conv1 to 4 channels only when the mask is passed as 4th input.
            # When feed_mask_to_backbone=False, conv1 stays at 3 channels (standard
            # pretrained weights) and the mask is only used in the bilinear pool.
            if feed_mask_to_backbone:
                base.conv1 = self._expand_conv_channels(base.conv1)
            base.fc = nn.Identity()
            self.backbone = ResNetSpatialBackbone(base)

        elif backbone == 'resnet50':
            base = models.resnet50(
                weights=models.resnet.ResNet50_Weights.DEFAULT if pretrained else None
            )
            backbone_out_dim = base.fc.in_features           # 2048
            if feed_mask_to_backbone:
                base.conv1 = self._expand_conv_channels(base.conv1)
            base.fc = nn.Identity()
            self.backbone = ResNetSpatialBackbone(base)

        elif backbone == 'dinov2_vits14':
            dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=pretrained)
            backbone_out_dim = dino.embed_dim                # 384
            # Expand patch_embed to 4 channels only when the mask is passed as 4th input.
            # When feed_mask_to_backbone=False, the standard 3-channel patch_embed is used
            # and DINOv2SpatialBackbone.forward will never try to slice a 4th channel.
            if feed_mask_to_backbone:
                dino.patch_embed.proj = self._expand_conv_channels(dino.patch_embed.proj)
            self.backbone = DINOv2SpatialBackbone(dino)

        elif backbone == 'dinov2_vitb14':
            dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=pretrained)
            backbone_out_dim = dino.embed_dim                # 768
            if feed_mask_to_backbone:
                dino.patch_embed.proj = self._expand_conv_channels(dino.patch_embed.proj)
            self.backbone = DINOv2SpatialBackbone(dino)

        elif backbone == 'dinov2_vitl14':
            dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=pretrained)
            backbone_out_dim = dino.embed_dim                # 1024
            if feed_mask_to_backbone:
                dino.patch_embed.proj = self._expand_conv_channels(dino.patch_embed.proj)
            self.backbone = DINOv2SpatialBackbone(dino)

        else:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                "Choose 'resnet18', 'resnet50', 'dinov2_vits14', "
                "'dinov2_vitb14', or 'dinov2_vitl14'."
            )

        self.proj_bilinear = nn.Linear(backbone_out_dim, bilinear_dim)
        self.projection = nn.Linear(bilinear_dim ** 2, output_dim)

        if train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False

    @staticmethod
    def _expand_conv_channels(conv):
        """
        Expand a Conv2d from 3 → 4 input channels.
        Works for both ResNet conv1 and DINOv2 patch_embed.proj since both are Conv2d.
        Pretrained RGB weights are preserved; mask channel is initialised small
        so it does not dominate RGB features early in training.
        """
        new_conv = nn.Conv2d(
            4, conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = conv.weight
            nn.init.constant_(new_conv.weight[:, 3:], 0.01)
            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
        return new_conv

    def _masked_bilinear_pool(self, feature_maps, soft_mask, eps=1e-6):
        """
        Combines Option C (mask applied to feature maps) with reduced bilinear pooling.

        - Downsample soft mask to feature map resolution
        - Weight spatial locations by mask — removes padding and non-segment regions
        - Normalise by total mask weight — correctly handles variable segment sizes
        - Project to bilinear_dim, compute covariance F^T F
        - Signed sqrt + L2 normalise for training stability

        For ResNet: captures local texture co-activations across segment positions.
        For DINOv2: captures semantic pattern co-occurrences across segment patches.
        The implementation is identical; only the interpretation shifts.

        feature_maps : (B, D, H', W')
        soft_mask    : (B, H, W)  values in [0, 1]
        Returns      : (B, bilinear_dim²)
        """
        B, D, Hf, Wf = feature_maps.shape

        # Downsample mask to match feature map spatial resolution (Option C)
        mask = F.interpolate(
            soft_mask.unsqueeze(1),   # (B, 1, H, W)
            size=(Hf, Wf),
            mode='bilinear',
            align_corners=False
        )  # (B, 1, Hf, Wf)

        # Early-return for all-zero masks: the bilinear pool would otherwise compute
        # proj_bilinear(all_zeros) = bias, then bias @ bias.T — non-zero noise with
        # no image information. This guard makes the method self-contained safe and
        # ensures the caller does not need to pre-check for empty segments.
        if (mask == 0).all():
            return torch.zeros(B, self.bilinear_dim ** 2, device=feature_maps.device)

        # Flatten spatial dims → sequence format
        f = feature_maps.view(B, D, -1).permute(0, 2, 1)  # (B, N, D)
        m = mask.view(B, -1, 1)                            # (B, N, 1)

        # Normalise by mask weight sum — a narrow segment and a wide segment
        # produce comparably scaled representations before the outer product
        mask_sum = m.sum(dim=1, keepdim=True).clamp(min=eps)
        f_weighted = f * m / mask_sum                      # (B, N, D)

        # Project to bilinear_dim before outer product
        f_proj = self.proj_bilinear(f_weighted)            # (B, N, bilinear_dim)

        # Bilinear pool: F^T F
        # Entry [d, k] = how much feature d and feature k co-activate
        # across the segment's spatial positions
        bp = torch.bmm(
            f_proj.transpose(1, 2),  # (B, bilinear_dim, N)
            f_proj                    # (B, N, bilinear_dim)
        )  # (B, bilinear_dim, bilinear_dim)

        # Signed sqrt normalisation — standard practice for bilinear pools,
        # prevents large entries from dominating and stabilises gradients
        bp = torch.sign(bp) * torch.sqrt(torch.abs(bp) + eps)
        bp = F.normalize(bp.view(B, -1), p=2, dim=1)      # (B, bilinear_dim²)

        return bp

    def forward(self, image, soft_mask):
        """
        Parameters
        ----------
        image : torch.Tensor
            (B, 3, H, W)
        soft_mask : torch.Tensor
            (B, H, W)  values in [0, 1], one soft mask per segment

        Returns
        -------
        torch.Tensor
            (B, output_dim)
        """
        if self.feed_mask_to_backbone:
            x = torch.cat([image, soft_mask.unsqueeze(1)], dim=1)  # (B, 4, H, W)
        else:
            x = image  # (B, 3, H, W) — backbone never sees the mask

        feature_maps = self.backbone(x)

        # Option C + bilinear: masked second-order aggregation
        pooled = self._masked_bilinear_pool(feature_maps, soft_mask)  # (B, bilinear_dim²)

        return self.projection(pooled)  # (B, output_dim)