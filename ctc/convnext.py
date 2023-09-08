import timm
import torch
from torch import nn
from ctc.model_utils import ConceptSlotAttention, CrossAttention, ConceptISA, ConceptQuerySlotAttention


# For CUB
def cub_slotc_convnext_sa(backbone_name="convnext_large.fb_in22k_ft_in1k", *args, **kwargs):
    return SlotCConvNeXtSA(
        model_name=backbone_name,
        num_classes=200,
        n_unsup_concepts=0,
        n_concepts=13,
        n_spatial_concepts=95,
        num_heads=12,
        attention_dropout=0.1,
        *args,
        **kwargs,
    )


def cub_slotc_convnext_isa(backbone_name="convnext_large.fb_in22k_ft_in1k", *args, **kwargs):
    return SlotCConvNeXtISA(
        model_name=backbone_name,
        num_classes=200,
        n_unsup_concepts=0,
        n_concepts=13,
        n_spatial_concepts=95,
        num_heads=12,
        attention_dropout=0.1,
        *args,
        **kwargs,
    )


def cub_slotc_convnext_qsa(backbone_name="convnext_large.fb_in22k_ft_in1k", *args, **kwargs):
    return SlotCConvNeXtQSA(
        model_name=backbone_name,
        num_classes=200,
        n_unsup_concepts=0,
        n_concepts=13,
        n_spatial_concepts=95,
        num_heads=12,
        attention_dropout=0.1,
        *args,
        **kwargs,
    )


class SlotCConvNeXtSA(nn.Module):
    """Neuro-Symbolic Concept-Centric Transformer with ViT"""

    def __init__(
            self,
            img_size=224,
            n_input_channels=3,
            embedding_dim=192,
            n_conv_layers=1,
            kernel_size=7,
            stride=2,
            padding=3,
            pooling_kernel_size=3,
            pooling_stride=2,
            pooling_padding=1,
            num_classes=200,
            model_name="convnext_base.fb_in22k_ft_in1k",
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

        self.embedding_dim = self.feature_extractor.num_features
        self.classifier = CoceptCentricTransformerConvNextSA(
            embedding_dim=self.embedding_dim,
            num_classes=num_classes,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            n_unsup_concepts=n_unsup_concepts,
            n_concepts=n_concepts,
            n_spatial_concepts=n_spatial_concepts,
            *args,
            **kwargs,
        )

        self.token_size = 14 * 14
        self.input_nn = nn.Sequential(
            nn.Linear(7 * 7, self.token_size)
        )

    def forward(self, x):
        x = self.feature_extractor.stem(x)
        x = self.feature_extractor.stages(x)
        x = self.feature_extractor.norm_pre(
            x)  # x (base): [B (16), num_features (1024), 7, 7], x (large): [B (16), num_features (1536), 7, 7], (small): [B (16, 768, 7, 7]
        x = x.reshape(x.size(0), -1, self.embedding_dim)  # x (large): [B (16), 7*7, num_features]
        x = self.input_nn(x.transpose(-1, -2)).transpose(-1, -2)  # x: [B, 14*14, num_features]
        out, unsup_concept_attn, concept_attn, spatial_concept_attn = self.classifier(x)

        return out, unsup_concept_attn, concept_attn, spatial_concept_attn


class CoceptCentricTransformerConvNextSA(nn.Module):
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

        out = 0.0
        if self.n_unsup_concepts > 0:  # unsupervised stream
            unsup_concepts, unsup_concepts_slot_attn = self.unsup_concept_slot_attention(x)
            unsup_concepts += self.unsup_concept_slot_pos.view(-1, self.n_unsup_concepts, self.embedding_dim)
            out_unsup, unsup_concept_attn = self.concept_tranformer(x, unsup_concepts)
            unsup_concept_attn = unsup_concept_attn.mean(1)  # average over heads
            out = out + out_unsup.mean(1)  # squeeze token dimension

        if self.n_concepts > 0:  # Non-spatial stream
            concepts, concept_slot_attn = self.concept_slot_attention(x)  # [B, num_concepts, embedding_dim]
            # Make slot concepts have an order.
            concepts += self.concept_slot_pos.view(-1, self.n_concepts,
                                                   self.embedding_dim)  # [1, num_concepts, embedding_dim]
            out_n, concept_attn = self.concept_tranformer(x, concepts)
            concept_attn = concept_attn.mean(1)  # average over heads
            out = out + out_n.mean(1)  # squeeze token dimension

        if self.n_spatial_concepts > 0:  # Spatial stream
            spatial_concepts, spatial_concept_slot_attn = self.spatial_concept_slot_attention(
                x)  # [B, num_concepts, embedding_dim]
            # Make slot concepts have an order.
            spatial_concepts += self.spatial_concept_slot_pos.view(-1, self.n_spatial_concepts,
                                                                   self.embedding_dim)  # [1, num_concepts, embedding_dim]
            out_s, spatial_concept_attn = self.spatial_concept_tranformer(x, spatial_concepts)
            spatial_concept_attn = spatial_concept_attn.mean(1)  # average over heads
            out = out + out_s.mean(1)  # pool tokens sequence

        if not concept_attn is None:
            concept_attn = concept_attn.mean(1)

        return out, unsup_concept_attn, concept_attn, spatial_concept_attn


class SlotCConvNeXtISA(nn.Module):
    """Neuro-Symbolic Concept-Centric Transformer with ViT"""

    def __init__(
            self,
            img_size=224,
            n_input_channels=3,
            embedding_dim=192,
            n_conv_layers=1,
            kernel_size=7,
            stride=2,
            padding=3,
            pooling_kernel_size=3,
            pooling_stride=2,
            pooling_padding=1,
            num_classes=200,
            model_name="convnext_base.fb_in22k_ft_in1k",
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

        self.embedding_dim = self.feature_extractor.num_features
        self.classifier = CoceptCentricTransformerConvNextISA(
            embedding_dim=self.embedding_dim,
            num_classes=num_classes,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            n_unsup_concepts=n_unsup_concepts,
            n_concepts=n_concepts,
            n_spatial_concepts=n_spatial_concepts,
            *args,
            **kwargs,
        )

        self.token_size = 14 * 14
        self.input_nn = nn.Sequential(
            nn.Linear(7 * 7, self.token_size)
        )

    def forward(self, x):
        x = self.feature_extractor.stem(x)
        x = self.feature_extractor.stages(x)
        x = self.feature_extractor.norm_pre(
            x)  # x (base): [B (16), num_features (1024), 7, 7], x (large): [B (16), num_features (1536), 7, 7], (small): [B (16, 768, 7, 7]
        x = x.reshape(x.size(0), -1, self.embedding_dim)  # x (large): [B (16), 7*7, num_features]
        x = self.input_nn(x.transpose(-1, -2)).transpose(-1, -2)  # x: [B, 14*14, num_features]
        out, unsup_concept_attn, concept_attn, spatial_concept_attn = self.classifier(x)

        return out, unsup_concept_attn, concept_attn, spatial_concept_attn


class CoceptCentricTransformerConvNextISA(nn.Module):
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
            self.unsup_concept_slot_attention = ConceptISA(num_iterations=1,
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
            self.concept_slot_attention = ConceptISA(num_iterations=1,
                                                     slot_size=embedding_dim,
                                                     mlp_hidden_size=embedding_dim, )
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
            self.spatial_concept_slot_attention = ConceptISA(num_iterations=1,
                                                             slot_size=embedding_dim,
                                                             mlp_hidden_size=embedding_dim, )
            self.spatial_concept_slot_pos = nn.Parameter(torch.zeros(1, 1, n_spatial_concepts * embedding_dim),
                                                         requires_grad=True)
            self.spatial_concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

    def forward(self, x, sigma=0):
        unsup_concept_attn, concept_attn, spatial_concept_attn = None, None, None

        out = 0.0
        if self.n_unsup_concepts > 0:  # unsupervised stream
            unsup_concepts, unsup_concepts_slot_attn = self.unsup_concept_slot_attention(x)
            unsup_concepts += self.unsup_concept_slot_pos.view(-1, self.n_unsup_concepts, self.embedding_dim)
            out_unsup, unsup_concept_attn = self.concept_tranformer(x, unsup_concepts)
            unsup_concept_attn = unsup_concept_attn.mean(1)  # average over heads
            out = out + out_unsup.mean(1)  # squeeze token dimension

        if self.n_concepts > 0:  # Non-spatial stream
            mu = self.concept_slots_init.weight.expand(x.size(0), -1, -1)
            z = torch.randn_like(mu).type_as(x)
            concept_slots_init = mu + z * sigma * mu.detach()
            concepts, concept_slot_attn = self.concept_slot_attention(x,
                                                                      concept_slots_init)  # [B, num_concepts, embedding_dim]
            # Make slot concepts have an order.
            concepts += self.concept_slot_pos.view(-1, self.n_concepts,
                                                   self.embedding_dim)  # [1, num_concepts, embedding_dim]
            out_n, concept_attn = self.concept_tranformer(x, concepts)
            concept_attn = concept_attn.mean(1)  # average over heads
            out = out + out_n.mean(1)  # squeeze token dimension

        if self.n_spatial_concepts > 0:  # Spatial stream
            mu = self.spatial_concept_slots_init.weight.expand(x.size(0), -1, -1)
            z = torch.randn_like(mu).type_as(x)
            spatial_concept_slots_init = mu + z * sigma * mu.detach()
            spatial_concepts, _ = self.spatial_concept_slot_attention(x,
                                                                      spatial_concept_slots_init)  # [B, num_concepts, embedding_dim]
            out_s, spatial_concept_attn = self.spatial_concept_tranformer(x, spatial_concepts)
            spatial_concept_attn = spatial_concept_attn.mean(1)  # average over heads
            out = out + out_s.mean(1)  # pool tokens sequence

        if not concept_attn is None:
            concept_attn = concept_attn.mean(1)

        return out, unsup_concept_attn, concept_attn, spatial_concept_attn


class SlotCConvNeXtQSA(nn.Module):
    """Neuro-Symbolic Concept-Centric Transformer with ViT"""

    def __init__(
            self,
            img_size=224,
            n_input_channels=3,
            embedding_dim=192,
            n_conv_layers=1,
            kernel_size=7,
            stride=2,
            padding=3,
            pooling_kernel_size=3,
            pooling_stride=2,
            pooling_padding=1,
            num_classes=200,
            model_name="convnext_base.fb_in22k_ft_in1k",
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

        self.embedding_dim = self.feature_extractor.num_features
        self.classifier = CoceptCentricTransformerConvNextQSA(
            embedding_dim=self.embedding_dim,
            num_classes=num_classes,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            n_unsup_concepts=n_unsup_concepts,
            n_concepts=n_concepts,
            n_spatial_concepts=n_spatial_concepts,
            *args,
            **kwargs,
        )

        self.token_size = 14 * 14
        self.input_nn = nn.Sequential(
            nn.Linear(7 * 7, self.token_size)
        )

    def forward(self, x):
        x = self.feature_extractor.stem(x)
        x = self.feature_extractor.stages(x)
        x = self.feature_extractor.norm_pre(
            x)  # x (base): [B (16), num_features (1024), 7, 7], x (large): [B (16), num_features (1536), 7, 7], (small): [B (16, 768, 7, 7]
        x = x.reshape(x.size(0), -1, self.embedding_dim)  # x (large): [B (16), 7*7, num_features]
        x = self.input_nn(x.transpose(-1, -2)).transpose(-1, -2)  # x: [B, 14*14, num_features]
        out, unsup_concept_attn, concept_attn, spatial_concept_attn = self.classifier(x)

        return out, unsup_concept_attn, concept_attn, spatial_concept_attn


class CoceptCentricTransformerConvNextQSA(nn.Module):
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
            self.unsup_concept_slot_attention = ConceptQuerySlotAttention(num_iterations=1,
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
            self.concept_slot_attention = ConceptQuerySlotAttention(num_iterations=1,
                                                                    slot_size=embedding_dim,
                                                                    mlp_hidden_size=embedding_dim, )
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
            self.spatial_concept_slot_attention = ConceptQuerySlotAttention(num_iterations=1,
                                                                            slot_size=embedding_dim,
                                                                            mlp_hidden_size=embedding_dim, )
            self.spatial_concept_slot_pos = nn.Parameter(torch.zeros(1, 1, n_spatial_concepts * embedding_dim),
                                                         requires_grad=True)
            self.spatial_concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

    def forward(self, x, sigma=0):
        unsup_concept_attn, concept_attn, spatial_concept_attn = None, None, None

        out = 0.0
        if self.n_unsup_concepts > 0:  # unsupervised stream
            unsup_concepts, unsup_concepts_slot_attn = self.unsup_concept_slot_attention(x)
            unsup_concepts += self.unsup_concept_slot_pos.view(-1, self.n_unsup_concepts, self.embedding_dim)
            out_unsup, unsup_concept_attn = self.concept_tranformer(x, unsup_concepts)
            unsup_concept_attn = unsup_concept_attn.mean(1)  # average over heads
            out = out + out_unsup.mean(1)  # squeeze token dimension

        if self.n_concepts > 0:  # Non-spatial stream
            mu = self.concept_slots_init.weight.expand(x.size(0), -1, -1)
            z = torch.randn_like(mu).type_as(x)
            concept_slots_init = mu + z * sigma * mu.detach()
            concepts, concept_slot_attn = self.concept_slot_attention(x,
                                                                      concept_slots_init)  # [B, num_concepts, embedding_dim]
            # Make slot concepts have an order.
            concepts += self.concept_slot_pos.view(-1, self.n_concepts,
                                                   self.embedding_dim)  # [1, num_concepts, embedding_dim]
            out_n, concept_attn = self.concept_tranformer(x, concepts)
            concept_attn = concept_attn.mean(1)  # average over heads
            out = out + out_n.mean(1)  # squeeze token dimension

        if self.n_spatial_concepts > 0:  # Spatial stream
            mu = self.spatial_concept_slots_init.weight.expand(x.size(0), -1, -1)
            z = torch.randn_like(mu).type_as(x)
            spatial_concept_slots_init = mu + z * sigma * mu.detach()
            spatial_concepts, _ = self.spatial_concept_slot_attention(x,
                                                                      spatial_concept_slots_init)  # [B, num_concepts, embedding_dim]
            out_s, spatial_concept_attn = self.spatial_concept_tranformer(x, spatial_concepts)
            spatial_concept_attn = spatial_concept_attn.mean(1)  # average over heads
            out = out + out_s.mean(1)  # pool tokens sequence

        if not concept_attn is None:
            concept_attn = concept_attn.mean(1)

        return out, unsup_concept_attn, concept_attn, spatial_concept_attn
