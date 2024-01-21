import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def ent_loss(probs, eps=1e-8):
    ent = -probs * torch.log(probs + eps)
    return ent.mean()


def concepts_sparsity_cost(concept_attn, spatial_concept_attn):
    """
    Sparsity loss based on entropy.
    :param concept_attn: predicted concept scores for global concepts.
    :param spatial_concept_attn: predicted concept scores for spatial/patch-level concepts.
    :return: sparsity loss.
    """
    cost = ent_loss(concept_attn) if concept_attn is not None else 0.0
    if spatial_concept_attn is not None:
        cost = cost + ent_loss(spatial_concept_attn)
    return cost


def concepts_cost(concept_attn, attn_targets):
    """
    Global concepts cost.
    Attention targets are normalized to sum to 1, but cost is normalized to be O(1)

    Args:
        attn_targets, torch.tensor of size (batch_size, n_concepts): one-hot attention targets
    """
    if concept_attn is None:
        return 0.0
    if attn_targets.dim() < 3:
        attn_targets = attn_targets.unsqueeze(1)
    norm = attn_targets.sum(-1, keepdims=True)
    idx = ~torch.isnan(norm).squeeze()
    if not torch.any(idx):
        return 0.0
    # MSE requires both floats to be of the same type
    norm_attn_targets = (attn_targets[idx] / norm[idx]).float()
    n_concepts = norm_attn_targets.shape[-1]
    return n_concepts * F.mse_loss(concept_attn[idx], norm_attn_targets, reduction="mean")


def spatial_concepts_cost(spatial_concept_attn, attn_targets):
    """
    Spatial concepts cost.
    Attention targets are normalized to sum to 1

    Args:
        attn_targets, torch.tensor of size (batch_size, n_patches, n_concepts):
            one-hot attention targets

    Note:
        If one patch contains a `np.nan` the whole patch is ignored
    """
    if spatial_concept_attn is None:
        return 0.0
    norm = attn_targets.sum(-1, keepdims=True)
    # Correctly shape idx for indexing
    if attn_targets.size(0) == 1:  # Handling for last batch
        idx = ~torch.isnan(norm) # remove squeeze to avoid empty tensor for idx
        idx = idx.squeeze(-1)  # Squeeze the last dimension to match with attn_targets[idx] and norm[idx]
    else:
        idx = ~torch.isnan(norm).squeeze()
    if not torch.any(idx):
        return 0.0
    norm_attn_targets = (attn_targets[idx] / norm[idx]).float()
    n_concepts = norm_attn_targets.shape[-1]
    return n_concepts * F.mse_loss(spatial_concept_attn[idx], norm_attn_targets, reduction="mean")
