import math

import torch
import torch.nn as nn


class TimeXerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
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

    def forward(self, endogenous_tokens, exogenous_tokens):
        self_attended, _ = self.self_attn(endogenous_tokens, endogenous_tokens, endogenous_tokens)
        endogenous_tokens = self.norm1(endogenous_tokens + self.dropout(self_attended))

        cross_attended, _ = self.cross_attn(endogenous_tokens, exogenous_tokens, exogenous_tokens)
        endogenous_tokens = self.norm2(endogenous_tokens + self.dropout(cross_attended))

        ffn_out = self.ffn(endogenous_tokens)
        endogenous_tokens = self.norm3(endogenous_tokens + self.dropout(ffn_out))
        return endogenous_tokens


class TimeXerForecaster(nn.Module):
    def __init__(
        self,
        input_dim_temp=16,
        input_dim_param=27,
        pred_steps=200,
        patch_len=16,
        stride=8,
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
        self.patch_len = patch_len
        self.stride = stride
        self.hidden_dim = hidden_dim
        self.max_patches = max_patches

        self.temp_patch_embed = nn.Linear(patch_len, hidden_dim)
        self.temp_variate_embed = nn.Embedding(input_dim_temp, hidden_dim)
        self.patch_pos_embed = nn.Parameter(torch.randn(max_patches, hidden_dim) / math.sqrt(hidden_dim))

        self.param_value_embed = nn.Linear(1, hidden_dim)
        self.param_variate_embed = nn.Embedding(input_dim_param, hidden_dim)

        self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim) / math.sqrt(hidden_dim))
        self.blocks = nn.ModuleList(
            [TimeXerBlock(hidden_dim, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_steps * input_dim_temp),
        )

    def _build_endogenous_tokens(self, temp_seq):
        batch_size, seq_len, _ = temp_seq.shape
        temp_seq = temp_seq.transpose(1, 2)
        patches = temp_seq.unfold(dimension=2, size=self.patch_len, step=self.stride)
        num_patches = patches.size(2)
        if num_patches > self.max_patches:
            raise ValueError(f"num_patches={num_patches} exceeds max_patches={self.max_patches}")

        patch_tokens = self.temp_patch_embed(patches)

        temp_index = torch.arange(self.input_dim_temp, device=temp_seq.device)
        temp_bias = self.temp_variate_embed(temp_index).view(1, self.input_dim_temp, 1, self.hidden_dim)
        pos_bias = self.patch_pos_embed[:num_patches].view(1, 1, num_patches, self.hidden_dim)
        patch_tokens = patch_tokens + temp_bias + pos_bias
        patch_tokens = patch_tokens.reshape(batch_size, self.input_dim_temp * num_patches, self.hidden_dim)
        return patch_tokens

    def _build_exogenous_tokens(self, param_now):
        batch_size = param_now.size(0)
        param_tokens = self.param_value_embed(param_now.unsqueeze(-1))
        param_index = torch.arange(self.input_dim_param, device=param_now.device)
        param_bias = self.param_variate_embed(param_index).view(1, self.input_dim_param, self.hidden_dim)
        param_tokens = param_tokens + param_bias
        return param_tokens.reshape(batch_size, self.input_dim_param, self.hidden_dim)

    def forward(self, temp_seq, param_now):
        endogenous_tokens = self._build_endogenous_tokens(temp_seq)
        exogenous_tokens = self._build_exogenous_tokens(param_now)

        global_token = self.global_token.expand(temp_seq.size(0), -1, -1)
        endogenous_tokens = torch.cat([global_token, endogenous_tokens], dim=1)

        for block in self.blocks:
            endogenous_tokens = block(endogenous_tokens, exogenous_tokens)

        summary = self.output_norm(endogenous_tokens[:, 0, :])
        output = self.head(summary)
        return output.view(-1, self.pred_steps, self.input_dim_temp)
