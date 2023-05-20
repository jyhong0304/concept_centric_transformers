import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
from ctc.model_utils import SlotAttention, Tokenizer, TransformerLayer, CrossAttention
import timm
from .vit import CVIT, SlotCVIT
from timm.models.vision_transformer import VisionTransformer

# Pre-defined CTC Models
__all__ = ["cifar100superclass_cvit"]


def cifar100superclass_cvit(backbone_name='vit_tiny_patch16_224', baseline=False, *args, **kwargs):
    """
    Args:
        baseline (bool): If true it returns the baseline model, which in this case it just the vit backbone without concept transformer
    """
    if baseline:
        return ExplVITTiny(num_classes=20, model_name=backbone_name)
    else:
        return CVIT(
            model_name=backbone_name,
            num_classes=20,
            n_unsup_concepts=0,
            n_concepts=100,
            n_spatial_concepts=0,
            num_heads=12,
            attention_dropout=0.1,
            projection_dropout=0.1,
            *args,
            **kwargs,
        )


def cifar100superclass_slotcvit(backbone_name='vit_tiny_patch16_224', *args, **kwargs):
    """
    Args:
        baseline (bool): If true it returns the baseline model, which in this case it just the vit backbone without concept transformer
    """
    return SlotCVIT(
        model_name=backbone_name,
        num_classes=20,
        n_unsup_concepts=0,
        n_concepts=100,
        n_spatial_concepts=0,
        num_heads=12,
        attention_dropout=0.1,
        projection_dropout=0.1,
        *args,
        **kwargs,
    )


class ExplVITTiny(VisionTransformer):
    """VIT modified to return dummy concept attentions"""

    def __init__(self, num_classes, model_name, embed_dim=192):
        super().__init__(num_classes=num_classes, embed_dim=embed_dim)

        loaded_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.load_state_dict(loaded_model.state_dict())

    def forward(self, x):
        out = super().forward(x)
        return out, None, None, None
