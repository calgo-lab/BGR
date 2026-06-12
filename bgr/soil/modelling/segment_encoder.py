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
    def __init__(self, backbone='resnet18', pretrained=True, output_dim=512, bilinear_dim=64):
        """
        backbone    : 'resnet18' | 'resnet50' | 'dinov2_vits14'
        pretrained  : load pretrained weights
        output_dim  : final embedding dimension
        bilinear_dim: projection dim before bilinear pool — output is bilinear_dim²
                      before the final projection. 64 → 4096 intermediate dim.
        """
        super().__init__()
        self.bilinear_dim = bilinear_dim

        if backbone == 'resnet18':
            base = models.resnet18(
                weights=models.resnet.ResNet18_Weights.DEFAULT if pretrained else None
            )
            backbone_out_dim = base.fc.in_features           # 512
            base.conv1 = self._expand_conv_channels(base.conv1)
            base.fc = nn.Identity()
            self.backbone = ResNetSpatialBackbone(base)

        elif backbone == 'resnet50':
            base = models.resnet50(
                weights=models.resnet.ResNet50_Weights.DEFAULT if pretrained else None
            )
            backbone_out_dim = base.fc.in_features           # 2048
            base.conv1 = self._expand_conv_channels(base.conv1)
            base.fc = nn.Identity()
            self.backbone = ResNetSpatialBackbone(base)

        elif backbone == 'dinov2_vits14':
            dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=pretrained)
            backbone_out_dim = dino.embed_dim                # 384 for ViT-S/14
            dino.patch_embed.proj = self._expand_conv_channels(dino.patch_embed.proj)
            self.backbone = DINOv2SpatialBackbone(dino)

        else:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                "Choose 'resnet18', 'resnet50', or 'dinov2_vits14'."
            )

        # Project to smaller space before bilinear pool to avoid D² explosion.
        # bilinear_dim=64 → 4096-dim covariance, tractable for any backbone.
        self.proj_bilinear = nn.Linear(backbone_out_dim, bilinear_dim)

        # Learned projection from flattened covariance matrix to output_dim
        self.projection = nn.Linear(bilinear_dim ** 2, output_dim)

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
        image     : (B, 3, H, W)
        soft_mask : (B, H, W)  values in [0, 1], one soft mask per segment
        Returns   : (B, output_dim)
        """
        # Option A: soft mask as 4th input channel — backbone sees where the
        # segment is and can condition its representations accordingly
        x = torch.cat([image, soft_mask.unsqueeze(1)], dim=1)  # (B, 4, H, W)

        # Unified call — returns (B, D, H', W') for all three backbone types
        feature_maps = self.backbone(x)

        # Option C + bilinear: masked second-order aggregation
        pooled = self._masked_bilinear_pool(feature_maps, soft_mask)  # (B, bilinear_dim²)

        return self.projection(pooled)  # (B, output_dim)