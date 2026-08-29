"""
Arquitecturas para el estudio de ablación (ver Notas.md y Experimentos.md).

TextEncoder: bloque Transformer Encoder-only (embedding + positional encoding
senoidal + N `nn.TransformerEncoderLayer` + mean-pooling sobre tokens no-pad).
Usa los módulos estándar de PyTorch, que implementan exactamente lo visto en
`transformers.VTT` (Multi-Head Self-Attention + residual + LayerNorm, MLP
feed-forward + residual + LayerNorm) — no hay atención ni encoding posicional
"a mano" ni técnicas alternativas no vistas en clase.

TabularMLPBaseline: capas densas solo sobre las features tabulares (experimento 1).
EncoderFusionModel: fusión tardía (experimento 2) — TextEncoder + tabular concatenados.
"""
import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        max_len: int = 45,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_dim = d_model

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        padding_mask = positions[None, :] >= lengths[:, None]  # True = posición de padding

        x = self.embedding(token_ids)
        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        valid = (~padding_mask).unsqueeze(-1).float()
        pooled = (x * valid).sum(dim=1) / lengths.unsqueeze(-1).float().clamp(min=1)
        return pooled


class TabularMLPBaseline(nn.Module):
    def __init__(self, n_tabular_features: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_tabular_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, tabular: torch.Tensor, token_ids=None, lengths=None) -> torch.Tensor:
        return self.net(tabular).squeeze(-1)


class EncoderFusionModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_tabular_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        max_len: int = 45,
        hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, d_model, n_heads, n_layers, dim_feedforward, max_len, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model + n_tabular_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, tabular: torch.Tensor, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        text_vec = self.text_encoder(token_ids, lengths)
        combined = torch.cat([text_vec, tabular], dim=1)
        return self.head(combined).squeeze(-1)
