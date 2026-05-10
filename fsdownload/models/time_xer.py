import math

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_CONFIG_KEYS = (
    "input_dim_temp",
    "input_dim_param",
    "pred_steps",
    "patch_len",
    "stride",
    "long_patch_len",
    "long_stride",
    "hidden_dim",
    "dropout",
    "num_layers",
    "num_heads",
    "d_ff",
    "max_patches",
)


def model_config_from_args(args):
    return {key: getattr(args, key) for key in MODEL_CONFIG_KEYS}


class CombustionVariableSelector(nn.Module):
    def __init__(self, input_dim_param, hidden_dim, dropout):
        super().__init__()
        self.context_mlp = nn.Sequential(
            nn.Linear(input_dim_param, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.local_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.score_proj = nn.Linear(hidden_dim, 1)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, param_tokens, raw_params):
        context = self.context_mlp(raw_params).unsqueeze(1)
        local_features = self.local_mlp(param_tokens)
        scores = self.score_proj(torch.tanh(local_features + context)).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        selected_tokens = self.output_norm(param_tokens * (1.0 + weights.unsqueeze(-1)))
        return selected_tokens, weights


class TemperatureGraphCoupling(nn.Module):
    def __init__(self, input_dim_temp, hidden_dim, dropout):
        super().__init__()
        self.src_embed = nn.Parameter(torch.randn(input_dim_temp, hidden_dim) / math.sqrt(hidden_dim))
        self.dst_embed = nn.Parameter(torch.randn(input_dim_temp, hidden_dim) / math.sqrt(hidden_dim))
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.last_adjacency = None

    def _build_adjacency(self, device, dtype):
        logits = torch.matmul(self.src_embed, self.dst_embed.transpose(0, 1)) / math.sqrt(self.src_embed.size(-1))
        logits = logits.to(device=device, dtype=dtype)
        logits = logits + torch.eye(logits.size(0), device=device, dtype=dtype)
        adjacency = torch.softmax(logits, dim=-1)
        return adjacency

    def forward(self, tokens):
        adjacency = self._build_adjacency(tokens.device, tokens.dtype)
        graph_context = torch.einsum("ij,bjph->biph", adjacency, tokens)
        graph_context = self.graph_proj(graph_context)
        fusion_gate = self.gate(torch.cat([tokens, graph_context], dim=-1))
        output = self.output_norm(tokens + fusion_gate * graph_context)
        self.last_adjacency = adjacency.detach()
        return output, adjacency


class MultiScaleTemperatureEncoder(nn.Module):
    def __init__(
        self,
        input_dim_temp,
        patch_len,
        stride,
        long_patch_len,
        long_stride,
        hidden_dim,
        max_patches,
        dropout,
    ):
        super().__init__()
        self.input_dim_temp = input_dim_temp
        self.patch_len = patch_len
        self.stride = stride
        self.long_patch_len = long_patch_len
        self.long_stride = long_stride
        self.hidden_dim = hidden_dim
        self.max_patches = max_patches

        self.short_patch_embed = nn.Linear(patch_len, hidden_dim)
        self.long_patch_embed = nn.Linear(long_patch_len, hidden_dim)
        self.temp_variate_embed = nn.Embedding(input_dim_temp, hidden_dim)
        self.short_pos_embed = nn.Parameter(torch.randn(max_patches, hidden_dim) / math.sqrt(hidden_dim))
        self.long_pos_embed = nn.Parameter(torch.randn(max_patches, hidden_dim) / math.sqrt(hidden_dim))

        self.short_graph = TemperatureGraphCoupling(input_dim_temp, hidden_dim, dropout)
        self.long_graph = TemperatureGraphCoupling(input_dim_temp, hidden_dim, dropout)
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

        self.last_short_adjacency = None
        self.last_long_adjacency = None

    def _build_scale_tokens(self, temp_seq, patch_len, stride, patch_embed, pos_embed):
        temp_seq = temp_seq.transpose(1, 2)
        patches = temp_seq.unfold(dimension=2, size=patch_len, step=stride)
        num_patches = patches.size(2)
        if num_patches > self.max_patches:
            raise ValueError(f"num_patches={num_patches} exceeds max_patches={self.max_patches}")

        patch_tokens = patch_embed(patches)
        temp_index = torch.arange(self.input_dim_temp, device=temp_seq.device)
        temp_bias = self.temp_variate_embed(temp_index).view(1, self.input_dim_temp, 1, self.hidden_dim)
        pos_bias = pos_embed[:num_patches].view(1, 1, num_patches, self.hidden_dim)
        return patch_tokens + temp_bias + pos_bias, num_patches

    @staticmethod
    def _align_long_tokens(long_tokens, target_patches):
        batch_size, num_vars, num_patches, hidden_dim = long_tokens.shape
        if num_patches == target_patches:
            return long_tokens

        pooled = long_tokens.permute(0, 1, 3, 2).reshape(batch_size * num_vars, hidden_dim, num_patches)
        pooled = F.adaptive_avg_pool1d(pooled, target_patches)
        pooled = pooled.reshape(batch_size, num_vars, hidden_dim, target_patches).permute(0, 1, 3, 2)
        return pooled

    def forward(self, temp_seq):
        short_tokens, num_short_patches = self._build_scale_tokens(
            temp_seq=temp_seq,
            patch_len=self.patch_len,
            stride=self.stride,
            patch_embed=self.short_patch_embed,
            pos_embed=self.short_pos_embed,
        )
        long_tokens, _ = self._build_scale_tokens(
            temp_seq=temp_seq,
            patch_len=self.long_patch_len,
            stride=self.long_stride,
            patch_embed=self.long_patch_embed,
            pos_embed=self.long_pos_embed,
        )

        short_tokens, short_adjacency = self.short_graph(short_tokens)
        long_tokens, long_adjacency = self.long_graph(long_tokens)
        long_tokens = self._align_long_tokens(long_tokens, num_short_patches)

        fused_tokens = self.scale_fusion(torch.cat([short_tokens, long_tokens], dim=-1))
        fused_tokens = self.output_norm(short_tokens + fused_tokens)
        flattened = fused_tokens.reshape(temp_seq.size(0), self.input_dim_temp * num_short_patches, self.hidden_dim)

        patch_lags = torch.arange(num_short_patches - 1, -1, -1, device=temp_seq.device, dtype=torch.long)
        patch_lags = patch_lags.repeat(self.input_dim_temp)

        self.last_short_adjacency = short_adjacency.detach()
        self.last_long_adjacency = long_adjacency.detach()
        return flattened, patch_lags


class LagAwareCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, num_exogenous, max_patches):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.lag_bias = nn.Parameter(torch.zeros(num_exogenous, num_heads, max_patches))
        nn.init.normal_(self.lag_bias, std=0.02)

    def _build_lag_bias(self, patch_indices, num_exogenous, device, dtype):
        bias = torch.zeros(
            1,
            self.num_heads,
            patch_indices.numel(),
            num_exogenous,
            device=device,
            dtype=dtype,
        )
        valid_mask = patch_indices >= 0
        if not torch.any(valid_mask):
            return bias

        selected = self.lag_bias[:num_exogenous, :, patch_indices[valid_mask]]
        selected = selected.permute(1, 2, 0)
        bias[:, :, valid_mask, :] = selected.unsqueeze(0)
        return bias

    def forward(self, query_tokens, exogenous_tokens, patch_indices):
        batch_size, num_query, _ = query_tokens.shape
        num_exogenous = exogenous_tokens.size(1)

        query = self.q_proj(query_tokens).view(batch_size, num_query, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(exogenous_tokens).view(batch_size, num_exogenous, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(exogenous_tokens).view(batch_size, num_exogenous, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        scores = scores + self._build_lag_bias(
            patch_indices=patch_indices,
            num_exogenous=num_exogenous,
            device=query_tokens.device,
            dtype=scores.dtype,
        )

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        context = torch.matmul(attn_weights, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_query, self.hidden_dim)
        return self.out_proj(context), attn_weights


class TimeXerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, d_ff, dropout, input_dim_param, max_patches):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = LagAwareCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_exogenous=input_dim_param,
            max_patches=max_patches,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, endogenous_tokens, exogenous_tokens, patch_indices):
        self_attended, _ = self.self_attn(endogenous_tokens, endogenous_tokens, endogenous_tokens)
        endogenous_tokens = self.norm1(endogenous_tokens + self.dropout(self_attended))

        cross_attended, cross_weights = self.cross_attn(endogenous_tokens, exogenous_tokens, patch_indices)
        endogenous_tokens = self.norm2(endogenous_tokens + self.dropout(cross_attended))

        ffn_out = self.ffn(endogenous_tokens)
        endogenous_tokens = self.norm3(endogenous_tokens + self.dropout(ffn_out))
        return endogenous_tokens, cross_weights


class TimeXerForecaster(nn.Module):
    def __init__(
        self,
        input_dim_temp=16,
        input_dim_param=27,
        pred_steps=200,
        patch_len=16,
        stride=8,
        long_patch_len=32,
        long_stride=16,
        hidden_dim=128,
        dropout=0.2,
        num_layers=3,
        num_heads=8,
        d_ff=256,
        max_patches=256,
    ):
        super().__init__()
        self.input_dim_temp = input_dim_temp
        self.input_dim_param = input_dim_param
        self.pred_steps = pred_steps
        self.hidden_dim = hidden_dim

        self.temperature_encoder = MultiScaleTemperatureEncoder(
            input_dim_temp=input_dim_temp,
            patch_len=patch_len,
            stride=stride,
            long_patch_len=long_patch_len,
            long_stride=long_stride,
            hidden_dim=hidden_dim,
            max_patches=max_patches,
            dropout=dropout,
        )

        self.param_value_embed = nn.Linear(1, hidden_dim)
        self.param_variate_embed = nn.Embedding(input_dim_param, hidden_dim)
        self.zone_embed = nn.Embedding(math.ceil(input_dim_param / 3), hidden_dim)
        self.channel_embed = nn.Embedding(3, hidden_dim)
        self.variable_selector = CombustionVariableSelector(input_dim_param, hidden_dim, dropout)

        self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim) / math.sqrt(hidden_dim))
        self.blocks = nn.ModuleList(
            [
                TimeXerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    input_dim_param=input_dim_param,
                    max_patches=max_patches,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_steps * input_dim_temp),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_steps * input_dim_temp),
        )

        self.last_selection_weights = None
        self.last_cross_attention = None
        self.last_short_graph = None
        self.last_long_graph = None

    def _build_exogenous_tokens(self, param_now):
        batch_size = param_now.size(0)
        param_tokens = self.param_value_embed(param_now.unsqueeze(-1))

        param_index = torch.arange(self.input_dim_param, device=param_now.device)
        zone_index = torch.div(param_index, 3, rounding_mode="floor")
        channel_index = torch.remainder(param_index, 3)

        param_tokens = (
            param_tokens
            + self.param_variate_embed(param_index).view(1, self.input_dim_param, self.hidden_dim)
            + self.zone_embed(zone_index).view(1, self.input_dim_param, self.hidden_dim)
            + self.channel_embed(channel_index).view(1, self.input_dim_param, self.hidden_dim)
        )
        param_tokens, selection_weights = self.variable_selector(param_tokens, param_now)
        return param_tokens.reshape(batch_size, self.input_dim_param, self.hidden_dim), selection_weights

    def predict_distribution(self, temp_seq, param_now, return_aux=False):
        endogenous_tokens, patch_lags = self.temperature_encoder(temp_seq)
        exogenous_tokens, selection_weights = self._build_exogenous_tokens(param_now)

        global_token = self.global_token.expand(temp_seq.size(0), -1, -1)
        endogenous_tokens = torch.cat([global_token, endogenous_tokens], dim=1)
        patch_indices = torch.cat(
            [
                torch.full((1,), -1, device=temp_seq.device, dtype=torch.long),
                patch_lags,
            ]
        )

        cross_attention = None
        for block in self.blocks:
            endogenous_tokens, cross_attention = block(endogenous_tokens, exogenous_tokens, patch_indices)

        summary = self.output_norm(endogenous_tokens[:, 0, :])
        output = self.head(summary).view(-1, self.pred_steps, self.input_dim_temp)
        logvar = self.logvar_head(summary).view(-1, self.pred_steps, self.input_dim_temp)
        logvar = torch.clamp(logvar, min=-6.0, max=4.0)

        self.last_selection_weights = selection_weights.detach()
        self.last_short_graph = self.temperature_encoder.last_short_adjacency
        self.last_long_graph = self.temperature_encoder.last_long_adjacency
        if cross_attention is not None:
            self.last_cross_attention = cross_attention.detach()

        if return_aux:
            return output, logvar, {
                "selection_weights": selection_weights,
                "cross_attention": cross_attention,
                "short_graph": self.temperature_encoder.last_short_adjacency,
                "long_graph": self.temperature_encoder.last_long_adjacency,
            }
        return output, logvar

    def forward(self, temp_seq, param_now, return_aux=False):
        if return_aux:
            output, _, aux = self.predict_distribution(temp_seq, param_now, return_aux=True)
            return output, aux
        output, _ = self.predict_distribution(temp_seq, param_now, return_aux=False)
        return output
