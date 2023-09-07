from .loss import concepts_cost, concepts_sparsity_cost, spatial_concepts_cost
from .vit import cifar100superclass_cvit, cifar100superclass_slotcvit_sa, cifar100superclass_slotcvit_isa, cifar100superclass_slotcvit_qsa
from .vit import cub_cvit, cub_slotcvit_sa, cub_slotcvit_isa, cub_slotcvit_qsa
from .vit import imagenet_vit_small, imagenet_cvit_small, imagenet_slotcvit_small_sa, imagenet_slotcvit_small_isa, imagenet_slotcvit_small_qsa
from .swin import cub_slotcswin_sa, cub_slotcswin_isa, cub_slotcswin_qsa
from .convnext import cub_slotc_convnext_sa, cub_slotc_convnext_isa, cub_slotc_convnext_qsa
from .ctc_model import CTCModel, load_exp, run_exp

__all__ = [
    # basic modules to execute models.
    "CTCModel",
    "run_exp",
    "load_exp",

    # losses
    "concepts_cost",
    "spatial_concepts_cost",
    "concepts_sparsity_cost",

    # models for CIFAR100Superclass experiments.
    "cifar100superclass_cvit",
    "cifar100superclass_slotcvit_sa",
    "cifar100superclass_slotcvit_isa",
    "cifar100superclass_slotcvit_qsa",

    # models for CUB-200-2011 experiments.
    "cub_cvit",
    "cub_slotcvit_sa",
    "cub_slotcvit_isa",
    "cub_slotcvit_qsa",
    "cub_slotcswin_sa",
    "cub_slotcswin_isa",
    "cub_slotcswin_qsa",
    "cub_slotc_convnext_sa",
    "cub_slotc_convnext_isa",
    "cub_slotc_convnext_qsa",

    # models for ImageNet experiments.
    "imagenet_vit_small",
    "imagenet_cvit_small",
    "imagenet_slotcvit_small_sa",
    "imagenet_slotcvit_small_isa",
    "imagenet_slotcvit_small_qsa",
]
