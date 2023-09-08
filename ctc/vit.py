import math
import timm
import torch
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from ctc.model_utils import ConceptSlotAttention, CrossAttention, ConceptQuerySlotAttention, ConceptISA


# For CIFAR100
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


class ExplVITTiny(VisionTransformer):
    """VIT modified to return dummy concept attentions"""

    def __init__(self, num_classes, model_name, embed_dim=192):
        super().__init__(num_classes=num_classes, embed_dim=embed_dim)

        loaded_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.load_state_dict(loaded_model.state_dict())

    def forward(self, x):
        out = super().forward(x)
        return out, None, None, None


def cifar100superclass_slotcvit_sa(backbone_name='vit_tiny_patch16_224', *args, **kwargs):
    return SlotCVITSA(
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


def cifar100superclass_slotcvit_isa(backbone_name='vit_tiny_patch16_224', *args, **kwargs):
    return SlotCVITISA(
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


def cifar100superclass_slotcvit_qsa(backbone_name='vit_tiny_patch16_224', *args, **kwargs):
    return SlotCVITQSA(
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


# For CUB
def cub_cvit(backbone_name="vit_large_patch16_224", baseline=False, *args, **kwargs):
    """
    Args:
        baseline (bool): If true it returns the baseline model, which in this case it just the vit backbone without concept transformer
    """
    if not baseline:
        return CVIT(
            model_name=backbone_name,
            num_classes=200,
            n_unsup_concepts=0,
            n_concepts=13,
            n_spatial_concepts=95,
            num_heads=12,
            attention_dropout=0.1,
            projection_dropout=0.1,
            *args,
            **kwargs,
        )
    else:
        return ExplVIT(model_name=backbone_name, num_classes=200)


def cub_slotcvit_sa(backbone_name="vit_large_patch16_224", *args, **kwargs):
    return SlotCVITSA(
        model_name=backbone_name,
        num_classes=200,
        n_unsup_concepts=0,
        n_concepts=13,
        n_spatial_concepts=95,
        num_heads=12,
        attention_dropout=0.1,
        projection_dropout=0.1,
        *args,
        **kwargs,
    )


def cub_slotcvit_isa(backbone_name="vit_large_patch16_224", *args, **kwargs):
    return SlotCVITISA(
        model_name=backbone_name,
        num_classes=200,
        n_unsup_concepts=0,
        n_concepts=13,
        n_spatial_concepts=95,
        num_heads=12,
        attention_dropout=0.1,
        projection_dropout=0.1,
        *args,
        **kwargs,
    )


def cub_slotcvit_qsa(backbone_name="vit_large_patch16_224", *args, **kwargs):
    return SlotCVITQSA(
        model_name=backbone_name,
        num_classes=200,
        n_unsup_concepts=0,
        n_concepts=13,
        n_spatial_concepts=95,
        num_heads=12,
        attention_dropout=0.1,
        projection_dropout=0.1,
        *args,
        **kwargs,
    )


# For ImageNet
def imagenet_cvit_small(backbone_name="vit_small_patch16_224", *args, **kwargs):
    return CVIT(
        model_name=backbone_name,
        num_classes=200,
        n_unsup_concepts=0,
        n_concepts=0,
        n_spatial_concepts=50,
        num_heads=12,
        attention_dropout=0.1,
        projection_dropout=0.1,
        *args,
        **kwargs,
    )


def imagenet_vit_small(backbone_name="vit_small_patch16_224", *args, **kwargs):
    return ExplVITTiny(model_name=backbone_name, num_classes=200, embed_dim=384)


def imagenet_slotcvit_small_sa(backbone_name="vit_small_patch16_224", *args, **kwargs):
    return SlotCVITSA(
        model_name=backbone_name,
        num_classes=200,
        n_unsup_concepts=0,
        n_concepts=0,
        n_spatial_concepts=50,
        num_heads=12,
        attention_dropout=0.1,
        projection_dropout=0.1,
        *args,
        **kwargs,
    )


def imagenet_slotcvit_small_isa(backbone_name="vit_small_patch16_224", *args, **kwargs):
    return SlotCVITISA(
        model_name=backbone_name,
        num_classes=200,
        n_unsup_concepts=0,
        n_concepts=0,
        n_spatial_concepts=50,
        num_heads=12,
        attention_dropout=0.1,
        projection_dropout=0.1,
        *args,
        **kwargs,
    )


def imagenet_slotcvit_small_qsa(backbone_name="vit_small_patch16_224", *args, **kwargs):
    return SlotCVITQSA(
        model_name=backbone_name,
        num_classes=200,
        n_unsup_concepts=0,
        n_concepts=0,
        n_spatial_concepts=50,
        num_heads=12,
        attention_dropout=0.1,
        projection_dropout=0.1,
        *args,
        **kwargs,
    )


class ExplVIT(VisionTransformer):
    """VIT modified to return dummy concept attentions"""

    def __init__(self, num_classes=200, model_name="vit_base_patch16_224", *args, **kwargs):
        super().__init__(num_classes=num_classes)

        loaded_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.load_state_dict(loaded_model.state_dict())

    def forward(self, x):
        out = super().forward(x)
        return out, None, None, None


class CVIT(nn.Module):
    """Concept Vision Transformer"""

    def __init__(
            self,
            num_classes=200,
            model_name="vit_base_patch16_224",
            pretrained=True,
            n_unsup_concepts=50,
            n_concepts=13,
            n_spatial_concepts=95,
            num_heads=12,
            attention_dropout=0.1,
            projection_dropout=0.1,
            *args,
            **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.feature_extractor = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        del self.feature_extractor.head

        self.classifier = ConceptTransformerVIT(
            embedding_dim=self.feature_extractor.embed_dim,
            num_classes=num_classes,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            n_unsup_concepts=n_unsup_concepts,
            n_concepts=n_concepts,
            n_spatial_concepts=n_spatial_concepts,
            *args,
            **kwargs,
        )

    def forward(self, x):
        x = self.feature_extractor.patch_embed(x)
        cls_token = self.feature_extractor.cls_token.expand(
            x.shape[0], -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
        x = self.feature_extractor.blocks(x)
        x = self.feature_extractor.norm(x)

        out, unsup_concept_attn, concept_attn, spatial_concept_attn = self.classifier(
            x[:, 0].unsqueeze(1), x[:, 1:]
        )
        return out, unsup_concept_attn, concept_attn, spatial_concept_attn


class SlotCVITSA(nn.Module):
    """Concept-Centric Transformer with ViT"""

    def __init__(
            self,
            num_classes=200,
            model_name="vit_base_patch16_224",
            pretrained=True,
            n_unsup_concepts=50,
            n_concepts=13,
            n_spatial_concepts=95,
            num_heads=12,
            attention_dropout=0.1,
            projection_dropout=0.1,
            *args,
            **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.feature_extractor = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        del self.feature_extractor.head

        self.classifier = ConceptCentricTransformerVITSA(
            embedding_dim=self.feature_extractor.embed_dim,
            num_classes=num_classes,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            n_unsup_concepts=n_unsup_concepts,
            n_concepts=n_concepts,
            n_spatial_concepts=n_spatial_concepts,
            *args,
            **kwargs,
        )

    def forward(self, x):
        x = self.feature_extractor.patch_embed(x)
        cls_token = self.feature_extractor.cls_token.expand(
            x.shape[0], -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
        x = self.feature_extractor.blocks(x)
        x = self.feature_extractor.norm(x)

        out, unsup_concept_attn, concept_attn, spatial_concept_attn = self.classifier(
            x[:, 0].unsqueeze(1), x[:, 1:]
        )
        return out, unsup_concept_attn, concept_attn, spatial_concept_attn


class ConceptTransformerVIT(nn.Module):
    """Processes spatial and non-spatial concepts in parallel and aggregates the log-probabilities at the end.
    The difference with the version in ctc.py is that instead of using sequence pooling for global concepts it
    uses the embedding of the cls token of the VIT
    """

    def __init__(
            self,
            embedding_dim=768,
            num_classes=10,
            num_heads=2,
            attention_dropout=0.1,
            projection_dropout=0.1,
            n_unsup_concepts=10,
            n_concepts=10,
            n_spatial_concepts=10,
            *args,
            **kwargs,
    ):
        super().__init__()

        # Unsupervised concepts
        self.n_unsup_concepts = n_unsup_concepts
        if n_unsup_concepts > 0:
            self.unsup_concepts = nn.Parameter(
                torch.zeros(1, n_unsup_concepts, embedding_dim), requires_grad=True
            )
            nn.init.trunc_normal_(self.unsup_concepts, std=1.0 / math.sqrt(embedding_dim))
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
            self.concepts = nn.Parameter(torch.zeros(1, n_concepts, embedding_dim), requires_grad=True)
            nn.init.trunc_normal_(self.concepts, std=1.0 / math.sqrt(embedding_dim))
            self.concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

        # Spatial Concepts
        self.n_spatial_concepts = n_spatial_concepts
        if n_spatial_concepts > 0:
            self.spatial_concepts = nn.Parameter(
                torch.zeros(1, n_spatial_concepts, embedding_dim), requires_grad=True
            )
            nn.init.trunc_normal_(self.spatial_concepts, std=1.0 / math.sqrt(embedding_dim))
            self.spatial_concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

    def forward(self, x_cls, x):
        unsup_concept_attn, concept_attn, spatial_concept_attn = None, None, None

        out = 0.0
        if self.n_unsup_concepts > 0:  # unsupervised stream
            out_unsup, unsup_concept_attn = self.concept_tranformer(x_cls, self.unsup_concepts)
            unsup_concept_attn = unsup_concept_attn.mean(1)  # average over heads
            out = out + out_unsup.squeeze(1)  # squeeze token dimension

        if self.n_concepts > 0:  # Non-spatial stream
            out_n, concept_attn = self.concept_tranformer(x_cls, self.concepts)
            concept_attn = concept_attn.mean(1)  # average over heads
            out = out + out_n.squeeze(1)  # squeeze token dimension

        if self.n_spatial_concepts > 0:  # Spatial stream
            out_s, spatial_concept_attn = self.spatial_concept_tranformer(x, self.spatial_concepts)
            spatial_concept_attn = spatial_concept_attn.mean(1)  # average over heads
            out = out + out_s.mean(1)  # pool tokens sequence

        return out, unsup_concept_attn, concept_attn, spatial_concept_attn


class ConceptCentricTransformerVITSA(nn.Module):
    """Processes spatial and non-spatial concepts in parallel and aggregates the log-probabilities at the end.
    The difference with the version in ctc.py is that instead of using sequence pooling for global concepts it
    uses the embedding of the cls token of the VIT
    """

    def __init__(
            self,
            embedding_dim=768,
            num_classes=10,
            num_heads=2,
            attention_dropout=0.1,
            projection_dropout=0.1,
            n_unsup_concepts=10,
            n_concepts=10,
            n_spatial_concepts=10,
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
            self.unsup_concept_slot_pos = nn.Parameter(torch.zeros(1, 1, n_unsup_concepts * embedding_dim),
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

        # Spatial Concepts
        self.n_spatial_concepts = n_spatial_concepts
        if n_spatial_concepts > 0:
            self.spatial_concepts = nn.Parameter(
                torch.zeros(1, n_spatial_concepts, embedding_dim), requires_grad=True
            )
            nn.init.trunc_normal_(self.spatial_concepts, std=1.0 / math.sqrt(embedding_dim))
            self.spatial_concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

    def forward(self, x_cls, x):
        unsup_concept_attn, concept_attn, spatial_concept_attn = None, None, None

        out = 0.0
        if self.n_unsup_concepts > 0:  # unsupervised stream
            unsup_concepts, unsup_concepts_slot_attn = self.unsup_concept_slot_attention(x_cls)
            unsup_concepts += self.unsup_concept_slot_pos.view(-1, self.n_unsup_concepts, self.embedding_dim)
            out_unsup, unsup_concept_attn = self.concept_tranformer(x_cls, unsup_concepts)
            unsup_concept_attn = unsup_concept_attn.mean(1)  # average over heads
            out = out + out_unsup.squeeze(1)  # squeeze token dimension

        if self.n_concepts > 0:  # Non-spatial stream
            concepts, concept_slot_attn = self.concept_slot_attention(x_cls)  # [B, num_concepts, embedding_dim]
            # Make slot concepts have an order.
            concepts += self.concept_slot_pos.view(-1, self.n_concepts,
                                                   self.embedding_dim)  # [1, num_concepts, embedding_dim]
            out_n, concept_attn = self.concept_tranformer(x_cls, concepts)
            concept_attn = concept_attn.mean(1)  # average over heads
            out = out + out_n.squeeze(1)  # squeeze token dimension

        if self.n_spatial_concepts > 0:  # Spatial stream
            out_s, spatial_concept_attn = self.spatial_concept_tranformer(x, self.spatial_concepts)
            spatial_concept_attn = spatial_concept_attn.mean(1)  # average over heads
            out = out + out_s.mean(1)  # pool tokens sequence

        return out, unsup_concept_attn, concept_attn, spatial_concept_attn


class SlotCVITISA(nn.Module):
    """Neuro-Symbolic Concept-Centric Transformer with ViT"""

    def __init__(
            self,
            num_classes=200,
            model_name="vit_base_patch16_224",
            pretrained=True,
            n_unsup_concepts=50,
            n_concepts=13,
            n_spatial_concepts=95,
            num_heads=12,
            attention_dropout=0.1,
            projection_dropout=0.1,
            expansion_size=1,
            *args,
            **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.feature_extractor = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        del self.feature_extractor.head

        self.n_global_concepts = n_concepts
        self.n_spatial_concepts = n_spatial_concepts

        self.classifier = ConceptCentricTransformerISA(
            embedding_dim=self.feature_extractor.embed_dim,
            num_classes=num_classes,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            n_unsup_concepts=n_unsup_concepts,
            n_concepts=n_concepts,
            n_spatial_concepts=n_spatial_concepts,
            *args,
            **kwargs,
        )

    def forward(self, x):
        x = self.feature_extractor.patch_embed(x)
        cls_token = self.feature_extractor.cls_token.expand(
            x.shape[0], -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_token, x), dim=1)
        x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
        x = self.feature_extractor.blocks(x)
        x = self.feature_extractor.norm(x)

        out, unsup_concept_attn, concept_attn, spatial_concept_attn = self.classifier(
            x[:, 0].unsqueeze(1), x[:, 1:]
        )
        return out, unsup_concept_attn, concept_attn, spatial_concept_attn


class ConceptCentricTransformerISA(nn.Module):
    def __init__(
            self,
            embedding_dim=768,
            num_classes=10,
            num_heads=2,
            attention_dropout=0.1,
            projection_dropout=0.1,
            n_unsup_concepts=10,
            n_concepts=10,
            n_spatial_concepts=10,
            num_iterations=3,
            *args,
            **kwargs,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        # Unsupervised concepts
        self.n_unsup_concepts = n_unsup_concepts
        if n_unsup_concepts > 0:
            self.unsup_concept_slot_attention = ConceptISA(num_iterations=num_iterations,
                                                           slot_size=embedding_dim,
                                                           mlp_hidden_size=embedding_dim,
                                                           )
            self.unsup_concept_slot_pos = nn.Parameter(torch.zeros(1, 1, n_unsup_concepts * embedding_dim),
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
            self.concept_slots_init = nn.Embedding(self.n_concepts, embedding_dim)
            nn.init.xavier_uniform_(self.concept_slots_init.weight)
            self.concept_slot_attention = ConceptISA(num_iterations=num_iterations,
                                                     slot_size=embedding_dim,
                                                     mlp_hidden_size=embedding_dim)
            self.concept_slot_pos = nn.Parameter(torch.zeros(1, 1, n_concepts * embedding_dim), requires_grad=True)
            self.concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

        # Spatial Concepts
        self.n_spatial_concepts = n_spatial_concepts
        if n_spatial_concepts > 0:
            self.spatial_concept_slots_init = nn.Embedding(self.n_spatial_concepts, embedding_dim)
            nn.init.xavier_uniform_(self.spatial_concept_slots_init.weight)
            self.spatial_concept_slot_attention = ConceptISA(num_iterations=num_iterations,
                                                             slot_size=embedding_dim,
                                                             mlp_hidden_size=embedding_dim)
            self.spatial_concept_slot_pos = nn.Parameter(torch.zeros(1, 1, n_spatial_concepts * embedding_dim),
                                                         requires_grad=True)
            self.spatial_concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

    def forward(self, x_cls, x, sigma=0):
        unsup_concept_attn, concept_attn, spatial_concept_attn = None, None, None

        out = 0.0
        if self.n_unsup_concepts > 0:  # unsupervised stream
            unsup_concepts, unsup_concepts_slot_attn = self.unsup_concept_slot_attention(x_cls)
            unsup_concepts += self.unsup_concept_slot_pos.view(-1, self.n_unsup_concepts, self.embedding_dim)
            out_unsup, unsup_concept_attn = self.concept_tranformer(x_cls, unsup_concepts)
            unsup_concept_attn = unsup_concept_attn.mean(1)  # average over heads
            out = out + out_unsup.squeeze(1)  # squeeze token dimension

        if self.n_concepts > 0:  # Non-spatial stream
            mu = self.concept_slots_init.weight.expand(x.size(0), -1, -1)
            z = torch.randn_like(mu).type_as(x)
            concept_slots_init = mu + z * sigma * mu.detach()
            concepts, concept_slot_attn = self.concept_slot_attention(x_cls,
                                                                      concept_slots_init)  # [B, num_concepts, embedding_dim]
            # Make slot concepts have an order.
            concepts += self.concept_slot_pos.view(-1, self.n_concepts,
                                                   self.embedding_dim)  # [1, num_concepts, embedding_dim]
            out_n, concept_attn = self.concept_tranformer(x_cls, concepts)
            concept_attn = concept_attn.mean(1)  # average over heads
            out = out + out_n.squeeze(1)  # squeeze token dimension

        if self.n_spatial_concepts > 0:  # Spatial stream
            mu = self.spatial_concept_slots_init.weight.expand(x.size(0), -1, -1)
            z = torch.randn_like(mu).type_as(x)
            spatial_concept_slots_init = mu + z * sigma * mu.detach()
            spatial_concepts, _ = self.spatial_concept_slot_attention(x,
                                                                      spatial_concept_slots_init)  # [B, num_concepts, embedding_dim]
            out_s, spatial_concept_attn = self.spatial_concept_tranformer(x, spatial_concepts)
            spatial_concept_attn = spatial_concept_attn.mean(1)  # average over heads
            out = out + out_s.mean(1)  # pool tokens sequence

        return out, unsup_concept_attn, concept_attn, spatial_concept_attn


class SlotCVITQSA(nn.Module):
    """Neuro-Symbolic Concept-Centric Transformer with ViT"""

    def __init__(
            self,
            num_classes=200,
            model_name="vit_base_patch16_224",
            pretrained=True,
            n_unsup_concepts=50,
            n_concepts=13,
            n_spatial_concepts=95,
            num_heads=12,
            attention_dropout=0.1,
            projection_dropout=0.1,
            expansion_size=1,
            *args,
            **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.feature_extractor = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        del self.feature_extractor.head

        self.n_global_concepts = n_concepts
        self.n_spatial_concepts = n_spatial_concepts

        self.classifier = ConceptCentricTransformerQSA(
            embedding_dim=self.feature_extractor.embed_dim,
            num_classes=num_classes,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            n_unsup_concepts=n_unsup_concepts,
            n_concepts=n_concepts,
            n_spatial_concepts=n_spatial_concepts,
            *args,
            **kwargs,
        )

    def forward(self, x):
        x = self.feature_extractor.patch_embed(x)
        cls_token = self.feature_extractor.cls_token.expand(
            x.shape[0], -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_token, x), dim=1)
        x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
        x = self.feature_extractor.blocks(x)
        x = self.feature_extractor.norm(x)

        out, unsup_concept_attn, concept_attn, spatial_concept_attn = self.classifier(
            x[:, 0].unsqueeze(1), x[:, 1:]
        )
        return out, unsup_concept_attn, concept_attn, spatial_concept_attn


class ConceptCentricTransformerQSA(nn.Module):
    def __init__(
            self,
            embedding_dim=768,
            num_classes=10,
            num_heads=2,
            attention_dropout=0.1,
            projection_dropout=0.1,
            n_unsup_concepts=10,
            n_concepts=10,
            n_spatial_concepts=10,
            num_iterations=2,
            *args,
            **kwargs,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        # Unsupervised concepts
        self.n_unsup_concepts = n_unsup_concepts
        if n_unsup_concepts > 0:
            self.unsup_concept_slot_attention = ConceptQuerySlotAttention(num_iterations=num_iterations,
                                                                          slot_size=embedding_dim,
                                                                          mlp_hidden_size=embedding_dim,
                                                                          )
            self.unsup_concept_slot_pos = nn.Parameter(torch.zeros(1, 1, n_unsup_concepts * embedding_dim),
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
            self.concept_slots_init = nn.Embedding(self.n_concepts, embedding_dim)
            nn.init.xavier_uniform_(self.concept_slots_init.weight)
            self.concept_slot_attention = ConceptQuerySlotAttention(num_iterations=num_iterations,
                                                                    slot_size=embedding_dim,
                                                                    mlp_hidden_size=embedding_dim)
            self.concept_slot_pos = nn.Parameter(torch.zeros(1, 1, n_concepts * embedding_dim), requires_grad=True)
            self.concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

        # Spatial Concepts
        self.n_spatial_concepts = n_spatial_concepts
        if n_spatial_concepts > 0:
            self.spatial_concept_slots_init = nn.Embedding(self.n_spatial_concepts, embedding_dim)
            nn.init.xavier_uniform_(self.spatial_concept_slots_init.weight)
            self.spatial_concept_slot_attention = ConceptQuerySlotAttention(num_iterations=num_iterations,
                                                                            slot_size=embedding_dim,
                                                                            mlp_hidden_size=embedding_dim)
            self.spatial_concept_slot_pos = nn.Parameter(torch.zeros(1, 1, n_spatial_concepts * embedding_dim),
                                                         requires_grad=True)
            self.spatial_concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

    def forward(self, x_cls, x, sigma=0):
        unsup_concept_attn, concept_attn, spatial_concept_attn = None, None, None

        out = 0.0
        if self.n_unsup_concepts > 0:  # unsupervised stream
            unsup_concepts, unsup_concepts_slot_attn = self.unsup_concept_slot_attention(x_cls)
            unsup_concepts += self.unsup_concept_slot_pos.view(-1, self.n_unsup_concepts, self.embedding_dim)
            out_unsup, unsup_concept_attn = self.concept_tranformer(x_cls, unsup_concepts)
            unsup_concept_attn = unsup_concept_attn.mean(1)  # average over heads
            out = out + out_unsup.squeeze(1)  # squeeze token dimension

        if self.n_concepts > 0:  # Non-spatial stream
            mu = self.concept_slots_init.weight.expand(x.size(0), -1, -1)
            z = torch.randn_like(mu).type_as(x)
            concept_slots_init = mu + z * sigma * mu.detach()
            concepts, concept_slot_attn = self.concept_slot_attention(x_cls,
                                                                      concept_slots_init)  # [B, num_concepts, embedding_dim]
            # Make slot concepts have an order.
            concepts += self.concept_slot_pos.view(-1, self.n_concepts,
                                                   self.embedding_dim)  # [1, num_concepts, embedding_dim]
            out_n, concept_attn = self.concept_tranformer(x_cls, concepts)
            concept_attn = concept_attn.mean(1)  # average over heads
            out = out + out_n.squeeze(1)  # squeeze token dimension

        if self.n_spatial_concepts > 0:  # Spatial stream
            mu = self.spatial_concept_slots_init.weight.expand(x.size(0), -1, -1)
            z = torch.randn_like(mu).type_as(x)
            spatial_concept_slots_init = mu + z * sigma * mu.detach()
            spatial_concepts, _ = self.spatial_concept_slot_attention(x,
                                                                      spatial_concept_slots_init)  # [B, num_concepts, embedding_dim]
            out_s, spatial_concept_attn = self.spatial_concept_tranformer(x, spatial_concepts)
            spatial_concept_attn = spatial_concept_attn.mean(1)  # average over heads
            out = out + out_s.mean(1)  # pool tokens sequence

        return out, unsup_concept_attn, concept_attn, spatial_concept_attn
