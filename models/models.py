import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttentionMIL(nn.Module):
    """Interpretable attention-based MIL (gated attention).

    This is the classic MIL aggregator where each instance (patch feature) receives
    an attention weight, and the slide embedding is the weighted sum.

    Inputs:
        x:    Tensor [B, L, D]  (B=batch, L=instances/patches, D=feature dim)
        mask: Bool   [B, L]     True for padded/invalid instances to ignore

    Returns:
        logits: Tensor [B, output]
        (optional) attn: Tensor [B, L] with attention weights (sum to 1 over valid instances)
    """

    def __init__(
        self,
        input_dim: int,
        layers_nodes,
        output: int,
        activation: str = "gelu",
        dropout: float = 0.3,
        attn_dim: int = 128,
    ):
        super().__init__()

        hidden_dim = int(layers_nodes[0]) if layers_nodes and len(layers_nodes) > 0 else int(input_dim)

        act_dict = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
            "prelu": nn.PReLU(),
        }
        act = act_dict.get(activation, nn.GELU())

        # Per-instance encoder (lightweight MLP)
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
        )

        # Gated attention:
        #   a_i = w^T( tanh(V h_i) ⊙ sigmoid(U h_i) )
        self.attn_V = nn.Linear(hidden_dim, attn_dim)
        self.attn_U = nn.Linear(hidden_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1, bias=False)

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _safe_softmax(attn_logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # attn_logits: [B, L]
        if mask is None:
            return F.softmax(attn_logits, dim=1)

        # Mask padded positions with -inf then softmax
        attn_logits = attn_logits.masked_fill(mask, torch.finfo(attn_logits.dtype).min)
        attn = F.softmax(attn_logits, dim=1)

        # If a row is fully masked, softmax can return NaNs. Force masked to 0 and renormalize.
        attn = attn.masked_fill(mask, 0.0)
        denom = attn.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return attn / denom

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, *, return_attn: bool = False):
        h = self.instance_encoder(x)  # [B, L, H]

        v = torch.tanh(self.attn_V(h))
        u = torch.sigmoid(self.attn_U(h))
        logits_attn = self.attn_w(v * u).squeeze(-1)  # [B, L]

        attn = self._safe_softmax(logits_attn, mask)  # [B, L]
        z = (attn.unsqueeze(-1) * h).sum(dim=1)       # [B, H]
        z = self.out_norm(z)
        logits = self.head(z)

        if return_attn:
            return logits, attn
        return logits


class AttentionMILTwoHead(nn.Module):
    """AttentionMIL with two output heads.

    - activities head outputs logits [B, K] (use softmax externally)
    - total head outputs a scalar [B, 1] (intended as log-total)
    """

    def __init__(
        self,
        input_dim: int,
        layers_nodes,
        output_act: int,
        activation: str = "gelu",
        dropout: float = 0.3,
        attn_dim: int = 128,
    ):
        super().__init__()

        hidden_dim = int(layers_nodes[0]) if layers_nodes and len(layers_nodes) > 0 else int(input_dim)

        act_dict = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
            "prelu": nn.PReLU(),
        }
        act = act_dict.get(activation, nn.GELU())

        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
        )

        self.attn_V = nn.Linear(hidden_dim, attn_dim)
        self.attn_U = nn.Linear(hidden_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1, bias=False)

        self.out_norm = nn.LayerNorm(hidden_dim)

        self.head_act = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, int(output_act)),
        )

        self.head_total = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _safe_softmax(attn_logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return F.softmax(attn_logits, dim=1)

        attn_logits = attn_logits.masked_fill(mask, torch.finfo(attn_logits.dtype).min)
        attn = F.softmax(attn_logits, dim=1)
        attn = attn.masked_fill(mask, 0.0)
        denom = attn.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return attn / denom

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, *, return_attn: bool = False):
        h = self.instance_encoder(x)  # [B, L, H]

        v = torch.tanh(self.attn_V(h))
        u = torch.sigmoid(self.attn_U(h))
        logits_attn = self.attn_w(v * u).squeeze(-1)  # [B, L]

        attn = self._safe_softmax(logits_attn, mask)
        z = (attn.unsqueeze(-1) * h).sum(dim=1)
        z = self.out_norm(z)

        act_logits = self.head_act(z)
        log_total = self.head_total(z)

        if return_attn:
            return act_logits, log_total, attn
        return act_logits, log_total


class Hist2SigPerceiverPool(nn.Module):
    """Perceiver-style latent cross-attention pooling for set/bag inputs.

    Designed for very long token sequences (e.g., L=3k-10k) where full self-attention
    over tokens would be too expensive, and where token order is not meaningful.

    Input:
        x: [B, L, D]
        mask: [B, L] bool, True for padded tokens (ignored)
    Output:
        [B, output]
    """

    def __init__(
        self,
        input_dim: int,
        layers_nodes,
        output: int,
        activation: str = "gelu",
        dropout: float = 0.3,
        num_latents: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()

        hidden_dim = int(layers_nodes[0])
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")

        num_latents = int(num_latents)
        if num_latents < 1:
            raise ValueError(f"num_latents must be >= 1, got {num_latents}")

        num_layers = int(num_layers)
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        num_heads = int(num_heads)
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        act_dict = {
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
            "relu": nn.ReLU(),
        }
        self.activation = act_dict.get(activation, nn.GELU())

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable latent array [M, H]
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim) * 0.02)

        # Cross-attention + FF blocks operating on latents.
        self.attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.attn_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.ff_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        ff_dim = int(layers_nodes[1]) if len(layers_nodes) > 1 else hidden_dim * 2
        self.ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, ff_dim),
                    self.activation,
                    nn.Dropout(dropout),
                    nn.Linear(ff_dim, hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(hidden_dim)

        head_dim = int(layers_nodes[2]) if len(layers_nodes) > 2 else hidden_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.LayerNorm(head_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(head_dim, output),
        )

    def forward(self, x, mask=None):
        # Project tokens
        x = self.input_proj(x)

        # Expand latents for batch: [B, M, H]
        bsz = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(bsz, -1, -1)

        # Latent cross-attention blocks: Q=latents, K/V=tokens
        for attn, attn_norm, ff, ff_norm in zip(self.attn, self.attn_norm, self.ff, self.ff_norm, strict=False):
            attn_out, _ = attn(
                latents,
                x,
                x,
                key_padding_mask=mask,
                need_weights=False,
            )
            latents = attn_norm(latents + self.dropout(attn_out))

            ff_out = ff(latents)
            latents = ff_norm(latents + ff_out)

        # Pool latents -> slide embedding
        pooled = latents.mean(dim=1)
        pooled = self.out_norm(pooled)
        return self.head(pooled)


class Hist2SigFirst_Res(nn.Module):
    def __init__(self, input_dim, layers_nodes, output, activation='gelu', dropout=0.3):
        super(Hist2SigFirst_Res, self).__init__()

        self.first = nn.Linear(input_dim, layers_nodes[0])

        self.attention_net = nn.MultiheadAttention(
            embed_dim=layers_nodes[0],
            num_heads=2,
            dropout=dropout,
            batch_first=True
        )

        self.layers_dims = layers_nodes
        self.layers = nn.ModuleList()
        self.norm = nn.ModuleList()

        self.dropout = nn.Dropout(dropout)

        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()

        for i in range(len(self.layers_dims) - 1):
            self.layers.append(nn.Linear(self.layers_dims[i], self.layers_dims[i + 1]))
            self.norm.append(nn.LayerNorm(self.layers_dims[i + 1]))

        # Pooling: concat(mean, max) => 2 * hidden_dim
        self.last = nn.Linear(layers_nodes[-1] * 2, output)

    def fully(self, x_concat):
        for i, layer in enumerate(self.layers):
            x_concat = layer(x_concat)
            x_concat = self.norm[i](x_concat)
            x_concat = self.activation(x_concat)
            if i != len(self.layers) - 1:
                x_concat = self.dropout(x_concat)
        return x_concat

    def forward(self, x, mask=None):
        x = self.first(x)
        attn_out, _ = self.attention_net(
            x,
            x,
            x,
            key_padding_mask=mask,
            need_weights=False,
        )
        x = x + attn_out

        x = self.fully(x)
        if mask is not None:
            # mask: True for padded positions.
            valid = (~mask).unsqueeze(-1).to(dtype=x.dtype)
            denom = valid.sum(dim=1).clamp(min=1.0)
            mean_pool = (x * valid).sum(dim=1) / denom

            min_val = torch.finfo(x.dtype).min
            max_pool = x.masked_fill(mask.unsqueeze(-1), min_val).max(dim=1).values
        else:
            mean_pool = x.mean(dim=1)
            max_pool = x.max(dim=1).values

        pooled = torch.cat([mean_pool, max_pool], dim=1)
        return self.last(pooled)


class Hist2SigConvMasked(nn.Module):
    """Fast baseline: Conv1d over patch sequence + masked pooling.

    - No attention (faster, often more stable)
    - Uses the same mask convention as MultiheadAttention: mask=True for padded tokens
    """

    def __init__(self, input_dim, layers_nodes, output, activation='gelu', dropout=0.3, conv_kernel=3):
        super().__init__()

        act_dict = {
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(),
            'relu': nn.ReLU(),
        }
        self.activation = act_dict.get(activation, nn.ReLU())

        hidden = layers_nodes[0]
        self.first = nn.Linear(input_dim, hidden)

        pad = conv_kernel // 2
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=conv_kernel, padding=pad)
        self.bn = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        self.norm = nn.ModuleList()
        for i in range(len(layers_nodes) - 1):
            self.layers.append(nn.Linear(layers_nodes[i], layers_nodes[i + 1]))
            self.norm.append(nn.LayerNorm(layers_nodes[i + 1]))

        # Pooling: concat(mean, max, var)
        self.last = nn.Linear(layers_nodes[-1] * 3, output)

    def _token_mlp(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.norm[i](x)
            x = self.activation(x)
            if i != len(self.layers) - 1:
                x = self.dropout(x)
        return x

    def forward(self, x, mask=None):
        # x: [B, L, D]
        x = self.first(x)

        # Conv1d expects [B, C, L]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = x.transpose(1, 2)

        x = self._token_mlp(x)

        if mask is not None:
            valid = (~mask).unsqueeze(-1).to(dtype=x.dtype)
            denom = valid.sum(dim=1).clamp(min=1.0)
            mean_pool = (x * valid).sum(dim=1) / denom

            second_moment = (x * x * valid).sum(dim=1) / denom
            var_pool = (second_moment - mean_pool * mean_pool).clamp(min=0.0)

            min_val = torch.finfo(x.dtype).min
            max_pool = x.masked_fill(mask.unsqueeze(-1), min_val).max(dim=1).values
        else:
            mean_pool = x.mean(dim=1)
            max_pool = x.max(dim=1).values

            var_pool = x.to(torch.float32).var(dim=1, unbiased=False).to(dtype=x.dtype)

        pooled = torch.cat([mean_pool, max_pool, var_pool], dim=1)
        return self.last(pooled)


class Hist2SigConvResMasked(nn.Module):
    """Conv1d backbone with a simple residual conv block + masked pooling.

    This keeps the same I/O contract as Hist2SigConvMasked but adds a skip connection
    around the conv block (often improves optimization stability).
    """

    def __init__(self, input_dim, layers_nodes, output, activation='gelu', dropout=0.3, conv_kernel=3, conv_blocks=1):
        super().__init__()

        act_dict = {
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(),
            'relu': nn.ReLU(),
        }
        self.activation = act_dict.get(activation, nn.ReLU())

        hidden = layers_nodes[0]
        self.first = nn.Linear(input_dim, hidden)

        conv_blocks = int(conv_blocks)
        if conv_blocks < 1:
            raise ValueError(f"conv_blocks must be >= 1, got {conv_blocks}")

        self.conv_blocks = conv_blocks
        pad = conv_kernel // 2
        self.convs = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, kernel_size=conv_kernel, padding=pad) for _ in range(conv_blocks)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(conv_blocks)])
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        self.norm = nn.ModuleList()
        for i in range(len(layers_nodes) - 1):
            self.layers.append(nn.Linear(layers_nodes[i], layers_nodes[i + 1]))
            self.norm.append(nn.LayerNorm(layers_nodes[i + 1]))

        # Pooling: concat(mean, max, var)
        self.last = nn.Linear(layers_nodes[-1] * 3, output)

    def _token_mlp(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.norm[i](x)
            x = self.activation(x)
            if i != len(self.layers) - 1:
                x = self.dropout(x)
        return x

    def forward(self, x, mask=None):
        # x: [B, L, D]
        x = self.first(x)

        # Conv1d expects [B, C, L]
        x_t = x.transpose(1, 2)
        for conv, bn in zip(self.convs, self.bns, strict=False):
            conv_out = conv(x_t)
            conv_out = bn(conv_out)
            conv_out = self.activation(conv_out)
            conv_out = self.dropout(conv_out)
            x_t = x_t + conv_out

        x = x_t.transpose(1, 2)
        x = self._token_mlp(x)

        if mask is not None:
            valid = (~mask).unsqueeze(-1).to(dtype=x.dtype)
            denom = valid.sum(dim=1).clamp(min=1.0)
            mean_pool = (x * valid).sum(dim=1) / denom

            second_moment = (x * x * valid).sum(dim=1) / denom
            var_pool = (second_moment - mean_pool * mean_pool).clamp(min=0.0)

            min_val = torch.finfo(x.dtype).min
            max_pool = x.masked_fill(mask.unsqueeze(-1), min_val).max(dim=1).values
        else:
            mean_pool = x.mean(dim=1)
            max_pool = x.max(dim=1).values
            var_pool = x.to(torch.float32).var(dim=1, unbiased=False).to(dtype=x.dtype)

        pooled = torch.cat([mean_pool, max_pool, var_pool], dim=1)
        return self.last(pooled)


def _as_list_int(x) -> list[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]


def build_model(args, input_dim: int, k: int, device: torch.device) -> nn.Module:
    """Instantiate the architecture described by a checkpoint's saved args."""
    if args.model == "attn_res":
        return Hist2SigFirst_Res(
            input_dim,
            args.layers_nodes,
            k,
            args.activation,
            args.dropout,
        ).to(device, dtype=torch.bfloat16)
    if args.model == "perceiver_pool":
        return Hist2SigPerceiverPool(
            input_dim=input_dim,
            layers_nodes=args.layers_nodes,
            output=k,
            activation=args.activation,
            dropout=args.dropout,
            num_latents=int(getattr(args, "perceiver_latents", 32) or 32),
            num_layers=int(getattr(args, "perceiver_layers", 2) or 2),
            num_heads=int(getattr(args, "perceiver_heads", 4) or 4),
        ).to(device, dtype=torch.bfloat16)
    if args.model == "attention_mil":
        return AttentionMIL(
            input_dim=input_dim,
            layers_nodes=_as_list_int(getattr(args, "layers_nodes", None)) or [256],
            output=k,
            activation=str(getattr(args, "activation", "gelu")),
            dropout=float(getattr(args, "dropout", 0.0) or 0.0),
            attn_dim=int(getattr(args, "attn_dim", 128) or 128),
        ).to(device, dtype=torch.bfloat16)
    raise ValueError(f"Unknown model={args.model!r}")
