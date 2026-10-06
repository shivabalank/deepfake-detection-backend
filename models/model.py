"""
models/model.py
-----------------
EfficientNet-B4 backbone (ImageNet-pretrained) with a custom binary
classification head:

    Dropout -> Linear -> ReLU -> Dropout -> Linear -> (Sigmoid via BCEWithLogitsLoss)

Note: we output a single raw logit and use BCEWithLogitsLoss for
numerical stability; sigmoid is applied only at inference time.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

from config import cfg
from utils.logger import get_logger

logger = get_logger(__name__)


class DeepfakeClassifier(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.4):
        super().__init__()

        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b4(weights=weights)

        in_features = backbone.classifier[1].in_features  # 1792 for b4
        backbone.classifier = nn.Identity()  # strip original head

        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, cfg.NUM_CLASSES),  # single logit
        )

        logger.info(f"Initialized EfficientNet-B4 (pretrained={pretrained}), "
                    f"feature dim={in_features}")

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        logger.info("Backbone frozen.")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        logger.info("Backbone unfrozen.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)          # (B, in_features)
        logits = self.head(features)          # (B, 1)
        return logits.squeeze(1)              # (B,)

    def get_last_conv_layer(self):
        """Returns the last convolutional layer, used as the Grad-CAM target."""
        # torchvision efficientnet: backbone.features is a Sequential of blocks
        return self.backbone.features[-1]
