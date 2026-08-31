"""
Modelo del Experimento 1: Transformer Encoder-only "lo más básico posible"
sobre title+description, sin features tabulares todavía (ver Notas.md).

Arquitectura: embedding + positional encoding senoidal (Clase 1) + 1
nn.TransformerEncoderLayer (self-attention + residual + LayerNorm,
feed-forward + residual + LayerNorm -- módulos estándar de PyTorch, no
hechos a mano) + mean-pooling sobre tokens no-pad + Linear a la salida.

El mean-pooling no está enseñado tal cual en la cátedra como técnica de
pooling para clasificación (chequeado en transformers.VTT/embeddings_*.VTT:
no aparece). Se apoya en que Marina explica la atención misma como "un
promedio ponderado" de tokens (embeddings_1.VTT) -- mean-pooling es el caso
de pesos uniformes de esa misma idea. Ver Notas.md para la discusión
completa de por qué no se usó [CLS] en su lugar.
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


class TextTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 16,
        n_heads: int = 1,
        n_layers: int = 1,
        dim_feedforward: int = 64,
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
        self.output = nn.Linear(d_model, 1)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        padding_mask = positions[None, :] >= lengths[:, None]  # True = posición de padding

        x = self.embedding(token_ids)
        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        valid = (~padding_mask).unsqueeze(-1).float()
        pooled = (x * valid).sum(dim=1) / lengths.unsqueeze(-1).float().clamp(min=1)
        return self.output(pooled).squeeze(-1)
