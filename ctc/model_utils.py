import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TransformerLayer(nn.Module):
    def __init__(
            self,
            embedding_dim=768,
            num_layers=12,
            num_heads=12,
            mlp_ratio=4.0,
            dropout_rate=0.1,
            attention_dropout=0.1,
            stochastic_depth_rate=0.1,
            positional_embedding="sine",
            sequence_length=None,
            *args,
            **kwargs,
    ):
        super().__init__()

        positional_embedding = (
            positional_embedding
            if positional_embedding in ["sine", "learnable", "none"]
            else "sine"
        )
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length

        assert sequence_length is not None or positional_embedding == "none", (
            f"Positional embedding is set to {positional_embedding} and"
            f" the sequence length was not specified."
        )

        if positional_embedding != "none":
            if positional_embedding == "learnable":
                self.positional_emb = nn.Parameter(
                    torch.zeros(1, sequence_length, embedding_dim), requires_grad=True
                )
                nn.init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = nn.Parameter(
                    self.sinusoidal_embedding(sequence_length, embedding_dim), requires_grad=False
                )
        else:
            self.positional_emb = None

        self.dropout = nn.Dropout(p=dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout_rate,
                    attention_dropout=attention_dropout,
                    drop_path_rate=dpr[i],
                )
                for i in range(num_layers)
            ]
        )

        self.apply(self.init_weight)

    def forward(self, x):
        if self.positional_emb is not None:
            x += self.positional_emb
        elif self.sequence_length is not None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode="constant", value=0)

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor(
            [[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)] for p in range(n_channels)]
        )
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


class CrossAttention(nn.Module):
    def __init__(
            self, dim, n_outputs=None, num_heads=8, attention_dropout=0.1, projection_dropout=0.0
    ):
        super().__init__()
        n_outputs = n_outputs if n_outputs else dim
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)

        self.proj = nn.Linear(dim, n_outputs)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x, y):
        B, Nx, C = x.shape
        By, Ny, Cy = y.shape

        assert C == Cy, "Feature size of x and y must be the same"

        q = self.q(x).reshape(B, Nx, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = (
            self.kv(y)
            .reshape(By, Ny, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q = q[0]
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nx, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class TransformerEncoderLayer(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """

    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            attention_dropout=0.1,
            drop_path_rate=0.1,
    ):
        super().__init__()
        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = Attention(
            dim=d_model,
            num_heads=nhead,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConceptSlotAttention(nn.Module):
    def __init__(self, num_iterations, num_slots,
                 input_size, slot_size, mlp_hidden_size,
                 epsilon=1e-8):
        super().__init__()

        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        # parameters for Gaussian initialization (shared by all slots).
        self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
        self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        # norms
        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)

        # linear maps for the attention module.
        self.project_q = nn.Linear(slot_size, slot_size, bias=False)
        self.project_k = nn.Linear(input_size, slot_size, bias=False)
        self.project_v = nn.Linear(input_size, slot_size, bias=False)

        # slot update functions.
        self.gru = nn.GRUCell(slot_size, slot_size)

    def forward(self, inputs):
        B, num_inputs, input_size = inputs.size()

        # initialize slots
        slots = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
        slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots

        # setup key and value
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        k = (self.slot_size ** (-0.5)) * k

        for i in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            attn_logits = torch.bmm(k, q.transpose(-1, -2))
            attn_vis = F.softmax(attn_logits, dim=-1)
            # `attn_vis` has shape: [batch_size, num_inputs, num_slots].

            # Weighted mean.
            attn = attn_vis + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.bmm(attn.transpose(-1, -2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            slots = self.gru(
                updates.view(-1, self.slot_size),
                slots_prev.view(-1, self.slot_size)
            )
            slots = slots.view(-1, self.num_slots, self.slot_size)

        return slots, attn_vis


class ConceptISA(nn.Module):
    """
    ISA module for extracting concepts.
    We refer to "Object Representations as Fixed Points" to implement it.
    [https://arxiv.org/abs/2207.00787]
    We simplify this by removing the last LayerNorm and MLP.
    """

    def __init__(
            self,
            num_iterations,
            slot_size,
            mlp_hidden_size,
            epsilon=1e-8,
            drop_path=0.2,
    ):
        super().__init__()
        self.slot_size = slot_size
        self.epsilon = epsilon
        self.num_iterations = num_iterations

        self.norm_feature = nn.LayerNorm(slot_size)
        self.norm_slots = nn.LayerNorm(slot_size)

        self.project_k = linear(slot_size, slot_size, bias=False)
        self.project_v = linear(slot_size, slot_size, bias=False)
        self.project_q = linear(slot_size, slot_size, bias=False)

        self.gru = gru_cell(slot_size, slot_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, features, slots_init):
        # `feature` has shape [batch_size, num_feature, inputs_size].
        features = self.norm_feature(features)
        k = self.project_k(features)  # Shape: [B, num_features, slot_size]
        v = self.project_v(features)  # Shape: [B, num_features, slot_size]

        B, N, D = v.shape
        slots = slots_init
        slots = self._iterate(lambda z: self._step(z, k, v), slots, self.num_iterations)
        slots = self._step(slots.detach(), k, v)

        return slots, None

    def _iterate(self, f, x, num_iters):
        for _ in range(num_iters):
            x = f(x)
        return x

    def _step(self, slots, k, v):
        B, N, D = v.shape
        q = self.project_q(slots)
        # Attention
        scale = D ** -0.5
        attn_logits = torch.einsum('bid,bjd->bij', q, k) * scale
        attn = F.softmax(attn_logits, dim=1)
        # Weighted mean
        attn_sum = torch.sum(attn, dim=-1, keepdim=True) + self.epsilon
        attn_wm = attn / attn_sum
        updates = torch.einsum('bij, bjd->bid', attn_wm, v)
        # Update slots
        slots = self.gru(
            updates.reshape(-1, D),
            slots.reshape(-1, D)
        )
        slots = slots.reshape(B, -1, D)

        return slots


class ConceptQuerySlotAttention(nn.Module):
    """
    BO-QSA module for extracting concepts.
    We refer to "Improving Object-centric Learning with Query Optimization" to implement it.
    [https://arxiv.org/abs/2210.08990]
    We simplify this by removing the last LayerNorm and MLP.
    """

    def __init__(
            self,
            num_iterations,
            slot_size,
            mlp_hidden_size,
            truncate='bi-level',
            epsilon=1e-8,
            drop_path=0.2,
    ):
        super().__init__()
        self.slot_size = slot_size
        self.epsilon = epsilon
        self.truncate = truncate
        self.num_iterations = num_iterations

        self.norm_feature = nn.LayerNorm(slot_size)
        self.norm_slots = nn.LayerNorm(slot_size)

        self.project_k = linear(slot_size, slot_size, bias=False)
        self.project_v = linear(slot_size, slot_size, bias=False)
        self.project_q = linear(slot_size, slot_size, bias=False)

        self.gru = gru_cell(slot_size, slot_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, features, slots_init):
        # `feature` has shape [batch_size, num_feature, inputs_size].
        features = self.norm_feature(features)
        k = self.project_k(features)  # Shape: [B, num_features, slot_size]
        v = self.project_v(features)  # Shape: [B, num_features, slot_size]

        B, N, D = v.shape
        slots = slots_init
        # Multiple rounds of attention.
        for i in range(self.num_iterations):
            if i == self.num_iterations - 1:
                slots = slots.detach() + slots_init - slots_init.detach()
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.project_q(slots)
            # Attention
            scale = D ** -0.5
            attn_logits = torch.einsum('bid,bjd->bij', q, k) * scale
            attn = F.softmax(attn_logits, dim=1)

            # Weighted mean
            attn_sum = torch.sum(attn, dim=-1, keepdim=True) + self.epsilon
            attn_wm = attn / attn_sum
            updates = torch.einsum('bij, bjd->bid', attn_wm, v)

            # Update slots
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            )
            slots = slots.reshape(B, -1, D)

        return slots, attn


def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    m = nn.Linear(in_features, out_features, bias)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)

    if bias:
        nn.init.zeros_(m.bias)

    return m


def gru_cell(input_size, hidden_size, bias=True):
    m = nn.GRUCell(input_size, hidden_size, bias)

    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)

    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)

    return m
