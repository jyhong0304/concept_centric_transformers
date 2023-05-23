import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from ctc.model_utils import ConceptSlotAttention, TransformerLayer, CrossAttention

# Pre-defined CTC Models
__all__ = ["mnist_ctc"]


def mnist_ctc(*args, **kwargs):
    return _cct(
        num_layers=2,
        num_heads=2,
        mlp_ratio=1,
        embedding_dim=128,
        img_size=28,
        n_input_channels=1,
        num_classes=2,
        n_unsup_concepts=0,
        n_concepts=10,
        n_spatial_concepts=0,
        *args,
        **kwargs,
    )


def mnist_slotctc(*args, **kwargs):
    return _cct(
        num_layers=2,
        num_heads=2,
        mlp_ratio=1,
        embedding_dim=128,
        img_size=28,
        n_input_channels=1,
        num_classes=2,
        n_unsup_concepts=0,
        n_concepts=10,
        n_spatial_concepts=0,
        use_slot=True,
        *args,
        **kwargs,
    )


def _cct(
        num_layers,
        num_heads,
        mlp_ratio,
        embedding_dim,
        kernel_size=3,
        stride=None,
        padding=None,
        use_slot=False,
        *args,
        **kwargs,
):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))

    if use_slot:
        return SlotCTC(
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            embedding_dim=embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            *args,
            **kwargs,
        )
    else:
        return CTC(
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            embedding_dim=embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            *args,
            **kwargs,
        )


class PretrainedTokenizer(nn.Module):
    def __init__(
            self, name="resnet50", embedding_dim=768, flatten=False, freeze=True, *args, **kwargs
    ):
        """
        Note: if img_size=448, number of tokens is 14 x 14
        """
        super().__init__()

        self.flatten = flatten
        self.embedding_dim = embedding_dim

        ver = torchvision.__version__.split("a")[0]
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model = torch.hub.load("pytorch/vision:v" + ver, name, pretrained=True)
        model.eval()

        # Remove final pooler and classifier
        self.n_features = model.fc.in_features
        model.avgpool = nn.Sequential()
        model.fc = nn.Sequential()

        # Freeze model
        self.model = model
        if freeze:
            self.model.requires_grad_(False)

        self.fc = nn.Conv2d(self.n_features, embedding_dim, kernel_size=1)

    def forward(self, x):
        out = self.model(x)

        width = int(math.sqrt(out.shape[1] / self.n_features))
        out = out.unflatten(-1, (self.n_features, width, width))
        out = self.fc(out)

        if self.flatten:
            return out.flatten(-2, -1).transpose(-2, -1)
        else:
            return out


class Tokenizer(nn.Module):
    """Applies strided convolutions to the input and then tokenizes (creates patches)"""

    def __init__(
            self,
            kernel_size,
            stride,
            padding,
            pooling_kernel_size=3,
            pooling_stride=2,
            pooling_padding=1,
            n_conv_layers=1,
            n_input_channels=3,
            n_output_channels=64,
            in_planes=64,
            activation=None,
            max_pool=True,
            conv_bias=False,
            flatten=True,
    ):
        super().__init__()

        n_filter_list = (
                [n_input_channels]
                + [in_planes for _ in range(n_conv_layers - 1)]
                + [n_output_channels]
        )

        self.conv_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        n_filter_list[i],
                        n_filter_list[i + 1],
                        kernel_size=(kernel_size, kernel_size),
                        stride=(stride, stride),
                        padding=(padding, padding),
                        bias=conv_bias,
                    ),
                    nn.Identity() if activation is None else activation(),
                    nn.MaxPool2d(
                        kernel_size=pooling_kernel_size,
                        stride=pooling_stride,
                        padding=pooling_padding,
                    )
                    if max_pool
                    else nn.Identity(),
                )
                for i in range(n_conv_layers)
            ]
        )

        self.flatten = flatten
        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        out = self.forward(torch.zeros((1, n_channels, height, width)))
        if self.flatten:
            return out.shape[1]
        else:
            return self.flattener(out).transpose(-2, -1).shape[1]

    def forward(self, x):
        out = self.conv_layers(x)
        if self.flatten:
            return self.flattener(out).transpose(-2, -1)
        else:
            return out

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class FactorizedPositionEncoding(nn.Module):
    def __init__(self, max_seqlen, dim, embedding_dim):
        super().__init__()
        self.dim = dim
        self.max_seqlen = max_seqlen

        # Create factorized position embeddings
        self.positional_emb = nn.ParameterList()
        for i in range(dim):
            self.positional_emb.append(
                nn.Parameter(
                    torch.zeros(1, self.max_seqlen, *(dim - 1) * [1], embedding_dim).transpose(
                        1, i + 1
                    ),
                    requires_grad=True,
                )
            )

    def forward(self, x):
        x = F.pad(
            x,
            [j for i in x.shape[-1:1:-1] for j in [0, self.max_seqlen - i]],
            mode="constant",
            value=0,
        )
        x = x.transpose(-1, 1)
        x = x + sum(self.positional_emb)
        return x


class ConceptTransformer(nn.Module):
    """
    Processes spatial and non-spatial concepts in parallel and aggregates the log-probabilities at the end
    """

    def __init__(
            self,
            embedding_dim=768,
            num_classes=10,
            num_heads=2,
            attention_dropout=0.1,
            projection_dropout=0.1,
            n_unsup_concepts=0,
            n_concepts=10,
            n_spatial_concepts=0,
            *args,
            **kwargs,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        # Unsupervised concepts
        self.n_unsup_concepts = n_unsup_concepts
        self.unsup_concepts = nn.Parameter(
            torch.zeros(1, n_unsup_concepts, embedding_dim), requires_grad=True
        )
        nn.init.trunc_normal_(self.unsup_concepts, std=1.0 / math.sqrt(embedding_dim))
        if n_unsup_concepts > 0:
            self.unsup_concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

        # Non-spatial concepts
        self.n_concepts = n_concepts
        self.concepts = nn.Parameter(torch.zeros(1, n_concepts, embedding_dim), requires_grad=True)
        nn.init.trunc_normal_(self.concepts, std=1.0 / math.sqrt(embedding_dim))

        if n_concepts > 0:
            self.concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

        # Sequence pooling for both non-spatial and unsupervised concepts
        if n_concepts > 0 or n_unsup_concepts > 0:
            self.token_attention_pool = nn.Linear(embedding_dim, 1)

        # Spatial Concepts
        self.n_spatial_concepts = n_spatial_concepts
        self.spatial_concepts = nn.Parameter(
            torch.zeros(1, n_spatial_concepts, embedding_dim), requires_grad=True
        )
        nn.init.trunc_normal_(self.spatial_concepts, std=1.0 / math.sqrt(embedding_dim))
        if n_spatial_concepts > 0:
            self.spatial_concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

    def forward(self, x):
        unsup_concept_attn, concept_attn, spatial_concept_attn = None, None, None

        out = 0
        if self.n_unsup_concepts > 0 or self.n_concepts > 0:
            token_attn = F.softmax(self.token_attention_pool(x), dim=1).transpose(-1, -2)
            x_pooled = torch.matmul(token_attn, x)

        if self.n_unsup_concepts > 0:  # unsupervised stream
            out_unsup, unsup_concept_attn = self.concept_tranformer(x_pooled, self.unsup_concepts)
            unsup_concept_attn = unsup_concept_attn.mean(1)  # average over heads
            out = out + out_unsup.squeeze(1)  # squeeze token dimension

        if self.n_concepts > 0:  # Non-spatial stream
            out_n, concept_attn = self.concept_tranformer(x_pooled, self.concepts)
            concept_attn = concept_attn.mean(1)  # average over heads
            out = out + out_n.squeeze(1)  # squeeze token dimension

        if self.n_spatial_concepts > 0:  # Spatial stream
            out_s, spatial_concept_attn = self.spatial_concept_tranformer(x, self.spatial_concepts)
            spatial_concept_attn = spatial_concept_attn.mean(1)  # average over heads
            out = out + out_s.mean(1)  # pool tokens sequence

        return out, unsup_concept_attn, concept_attn, spatial_concept_attn


class ConceptCentricTransformer(nn.Module):
    def __init__(
            self,
            embedding_dim=768,
            num_classes=10,
            num_heads=2,
            attention_dropout=0.1,
            projection_dropout=0.1,
            n_unsup_concepts=0,
            n_concepts=10,
            n_spatial_concepts=0,
            *args,
            **kwargs,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        # Unsupervised concepts
        self.n_unsup_concepts = n_unsup_concepts
        if n_unsup_concepts > 0:
            self.unsup_concept_slot_attention = ConceptSlotAttention(num_iterations=1, num_slots=n_unsup_concepts,
                                                                     slot_size=embedding_dim,
                                                                     mlp_hidden_size=embedding_dim,
                                                                     input_size=embedding_dim)
            self.unsup_concept_slot_pos = nn.Parameter(torch.zeros(1, 1, n_concepts * embedding_dim),
                                                       requires_grad=True)
            self.unsup_concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )
        # Non-spatial concepts
        self.n_concepts = n_concepts
        if n_concepts > 0:
            # JINYUNG HONG
            self.concept_slot_attention = ConceptSlotAttention(num_iterations=1, num_slots=n_concepts,
                                                               slot_size=embedding_dim,
                                                               mlp_hidden_size=embedding_dim, input_size=embedding_dim)
            self.concept_slot_pos = nn.Parameter(torch.zeros(1, 1, n_concepts * embedding_dim), requires_grad=True)
            self.concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

        # Sequence pooling for both non-spatial and unsupervised concepts
        if n_concepts > 0 or n_unsup_concepts > 0:
            self.token_attention_pool = nn.Linear(embedding_dim, 1)

        # Spatial Concepts
        self.n_spatial_concepts = n_spatial_concepts
        if n_spatial_concepts > 0:
            # JINYUNG HONG
            self.spatial_concept_slot_attention = ConceptSlotAttention(num_iterations=1, num_slots=n_spatial_concepts,
                                                                       slot_size=embedding_dim,
                                                                       mlp_hidden_size=embedding_dim,
                                                                       input_size=embedding_dim)
            self.spatial_concept_slot_pos = nn.Parameter(torch.zeros(1, 1, n_spatial_concepts * embedding_dim),
                                                         requires_grad=True)
            self.spatial_concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

    def forward(self, x):
        unsup_concept_attn, concept_attn, spatial_concept_attn = None, None, None

        out = 0
        if self.n_unsup_concepts > 0 or self.n_concepts > 0:
            token_attn = F.softmax(self.token_attention_pool(x), dim=1).transpose(-1, -2)
            x_pooled = torch.matmul(token_attn, x)

        if self.n_unsup_concepts > 0:  # unsupervised stream
            unsup_concepts, _ = self.unsup_concept_slot_attention(x_pooled)  # [B, num_concepts, embedding_dim]
            unsup_concepts += self.unsup_concept_slot_pos.view(-1, self.n_unsup_concepts, self.embedding_dim)
            out_unsup, unsup_concept_attn = self.concept_tranformer(x_pooled, unsup_concepts)
            unsup_concept_attn = unsup_concept_attn.mean(1)  # average over heads
            out = out + out_unsup.squeeze(1)  # squeeze token dimension

        if self.n_concepts > 0:  # Non-spatial stream
            # Use slot-attention to extract concepts before applying it to concept transformer.
            concepts, _ = self.concept_slot_attention(x_pooled)  # [B, num_concepts, embedding_dim]
            # Make slot concepts have an order.
            concepts += self.concept_slot_pos.view(-1, self.n_concepts, self.embedding_dim)
            out_n, concept_attn = self.concept_tranformer(x_pooled, concepts)
            concept_attn = concept_attn.mean(1)  # average over heads
            out = out + out_n.squeeze(1)  # squeeze token dimension

        if self.n_spatial_concepts > 0:  # Spatial stream
            spatial_concepts, _ = self.spatial_concept_slot_attention(x)
            spatial_concepts += self.spatial_concept_slot_pos.view(-1, self.n_spatial_concepts, self.embedding_dim)
            out_s, spatial_concept_attn = self.spatial_concept_tranformer(x, spatial_concepts)
            spatial_concept_attn = spatial_concept_attn.mean(1)  # average over heads
            out = out + out_s.mean(1)  # pool tokens sequence

        return out, unsup_concept_attn, concept_attn, spatial_concept_attn


# Shared methods
def ent_loss(probs):
    """Entropy loss"""
    ent = -probs * torch.log(probs + 1e-8)
    return ent.mean()


def concepts_sparsity_cost(concept_attn, spatial_concept_attn):
    cost = ent_loss(concept_attn) if concept_attn is not None else 0.0
    if spatial_concept_attn is not None:
        cost = cost + ent_loss(spatial_concept_attn)
    return cost


def concepts_cost(concept_attn, attn_targets):
    """Non-spatial concepts cost
        Attention targets are normalized to sum to 1,
        but cost is normalized to be O(1)

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
    """Spatial concepts cost
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
    idx = ~torch.isnan(norm).squeeze()
    if not torch.any(idx):
        return 0.0
    norm_attn_targets = (attn_targets[idx] / norm[idx]).float()
    n_concepts = norm_attn_targets.shape[-1]
    return n_concepts * F.mse_loss(spatial_concept_attn[idx], norm_attn_targets, reduction="mean")


class CTC(nn.Module):
    """Concept Transformer Classifier"""

    def __init__(
            self,
            img_size=224,
            embedding_dim=768,
            n_input_channels=3,
            n_conv_layers=1,
            kernel_size=7,
            stride=2,
            padding=3,
            pooling_kernel_size=3,
            pooling_stride=2,
            pooling_padding=1,
            n_concepts=0,
            n_spatial_concepts=0,
            *args,
            **kwargs,
    ):
        super().__init__()

        self.n_concepts = n_concepts
        self.n_spatial_concepts = n_spatial_concepts

        self.tokenizer = Tokenizer(
            n_input_channels=n_input_channels,
            n_output_channels=embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            pooling_padding=pooling_padding,
            max_pool=True,
            activation=nn.ReLU,
            n_conv_layers=n_conv_layers,
            conv_bias=False,
        )

        sequence_length = self.tokenizer.sequence_length(
            n_channels=n_input_channels, height=img_size, width=img_size
        )
        self.transformer_classifier = TransformerLayer(
            embedding_dim=embedding_dim, sequence_length=sequence_length, *args, **kwargs
        )
        self.norm = nn.LayerNorm(embedding_dim)

        self.concept_transformer = ConceptTransformer(
            embedding_dim=embedding_dim,
            attention_dropout=0.1,
            projection_dropout=0.1,
            n_concepts=n_concepts,
            n_spatial_concepts=n_spatial_concepts,
            *args,
            **kwargs,
        )

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.transformer_classifier(x)
        x = self.norm(x)
        out, unsup_concept_attn, concept_attn, spatial_concept_attn = self.concept_transformer(x)
        return out.squeeze(-2), unsup_concept_attn, concept_attn, spatial_concept_attn


class SlotCTC(nn.Module):
    """Slot Concept Transformer Classifier"""

    def __init__(
            self,
            img_size=224,
            embedding_dim=768,
            n_input_channels=3,
            n_conv_layers=1,
            kernel_size=7,
            stride=2,
            padding=3,
            pooling_kernel_size=3,
            pooling_stride=2,
            pooling_padding=1,
            n_concepts=0,
            n_spatial_concepts=0,
            *args,
            **kwargs,
    ):
        super().__init__()

        self.n_concepts = n_concepts
        self.n_spatial_concepts = n_spatial_concepts

        self.tokenizer = Tokenizer(
            n_input_channels=n_input_channels,
            n_output_channels=embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            pooling_padding=pooling_padding,
            max_pool=True,
            activation=nn.ReLU,
            n_conv_layers=n_conv_layers,
            conv_bias=False,
        )

        sequence_length = self.tokenizer.sequence_length(
            n_channels=n_input_channels, height=img_size, width=img_size
        )
        self.transformer_classifier = TransformerLayer(
            embedding_dim=embedding_dim, sequence_length=sequence_length, *args, **kwargs
        )
        self.norm = nn.LayerNorm(embedding_dim)

        self.concept_transformer = ConceptCentricTransformer(
            embedding_dim=embedding_dim,
            attention_dropout=0.1,
            projection_dropout=0.1,
            n_concepts=n_concepts,
            n_spatial_concepts=n_spatial_concepts,
            *args,
            **kwargs,
        )

    def forward(self, x):
        """
        x: [B, 1, 28, 28]
        """
        x = self.tokenizer(x)
        # x: [B, 196, 128 (embed_size)]
        x = self.transformer_classifier(x)
        # x: [B, 196, 128 (embed_size)]
        x = self.norm(x)
        out, unsup_concept_attn, concept_attn, spatial_concept_attn = self.concept_transformer(x)
        # slot_attn: [B, 100, 196] expected, but [B, 1, 10]???
        return out.squeeze(-2), unsup_concept_attn, concept_attn, spatial_concept_attn


class CTC_ADAPT(nn.Module):
    """Concept Transformer Classifier with pretrained tokenizer and adaptive number of tokens

    Args:
        max_seqlen (int): Maximal sequence length per dimension, which means that if max_seqlen=20 and
            inputs are 2D (images) then the number of tokens is 20*20=400
    """

    def __init__(
            self, embedding_dim=768, n_concepts=0, n_spatial_concepts=0, max_seqlen=14, *args, **kwargs
    ):
        super().__init__()

        self.n_concepts = n_concepts
        self.n_spatial_concepts = n_spatial_concepts

        self.tokenizer = PretrainedTokenizer(
            embedding_dim=embedding_dim, flatten=False, *args, **kwargs
        )
        self.position_embedding = FactorizedPositionEncoding(max_seqlen, 2, embedding_dim)
        self.flattener = nn.Flatten(1, 2)
        self.transformer_classifier = TransformerLayer(
            embedding_dim=embedding_dim, positional_embedding="none", *args, **kwargs
        )
        self.norm = nn.LayerNorm(embedding_dim)

        self.concept_transformer = ConceptTransformer(
            embedding_dim=embedding_dim,
            attention_dropout=0.1,
            projection_dropout=0.1,
            n_concepts=n_concepts,
            n_spatial_concepts=n_spatial_concepts,
            *args,
            **kwargs,
        )

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.position_embedding(x)
        x = self.flattener(x)
        x = self.transformer_classifier(x)
        x = self.norm(x)
        out, unsup_concept_attn, concept_attn, spatial_concept_attn = self.concept_transformer(x)
        return out.squeeze(-2), unsup_concept_attn, concept_attn, spatial_concept_attn
