"""
Arquitecturas del Ejercicio 2 (ver Notas.md y Experimentos.md).

`TextEncoder`: bloque Transformer Encoder-only sobre title+description --
embedding + positional encoding senoidal (Clase 1) + N
`nn.TransformerEncoderLayer` (self-attention + residual + LayerNorm,
feed-forward + residual + LayerNorm -- módulos estándar de PyTorch) +
mean-pooling sobre tokens no-pad. Devuelve un solo vector por fila.

El mean-pooling no está enseñado tal cual en la cátedra como técnica de
pooling para clasificación (chequeado en transformers.VTT/embeddings_*.VTT:
no aparece). Se apoya en que Marina explica la atención misma como "un
promedio ponderado" de tokens (embeddings_1.VTT) -- mean-pooling es el caso
de pesos uniformes de esa misma idea. Ver Notas.md para la discusión
completa de por qué no se usó [CLS] en su lugar.

`TextTransformerClassifier`: solo texto (Experimentos 1 a 6) -- el vector
de `TextEncoder` va directo a una salida lineal, sin usar las features
tabulares.

`CombinedModel`: junta la salida de `TextEncoder` con el vector de
features tabulares ya encodeadas (concatenación simple) y sigue por una
salida lineal -- el texto no pasa por ninguna capa que vea lo tabular, y
lo tabular nunca entra al Transformer. Ver Notas.md para la discusión de
por qué se combinan así y no de otra forma.
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
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        max_len: int,
        dropout: float,
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
        return (x * valid).sum(dim=1) / lengths.unsqueeze(-1).float().clamp(min=1)


class TextTransformerClassifier(nn.Module):
    """Experimentos 1 a 6 -- solo texto, sin features tabulares."""

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
        self.text_encoder = TextEncoder(vocab_size, d_model, n_heads, n_layers, dim_feedforward, max_len, dropout)
        self.output = nn.Linear(d_model, 1)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.output(self.text_encoder(token_ids, lengths)).squeeze(-1)


class CombinedModel(nn.Module):
    """Texto (vía TextEncoder) + features tabulares, concatenados antes de la salida.

    Defaults = configuración ganadora de los Experimentos 1 a 6
    (n_heads=1, n_layers=2, d_model=64, dim_feedforward=64).

    `hidden`: si es `None` (default, usado en el Experimento 7), la salida
    es un `Linear` directo sobre el vector combinado -- no hay forma de
    aprender una interacción entre lo que dice el texto y lo que dicen las
    tabulares, solo pesar cada una por separado. Si se pasa un entero
    (Experimento 8 en adelante), se agrega una capa oculta no-lineal antes
    de la salida (`Linear → ReLU → Dropout → Linear`) para poder cruzar
    ambas ramas.
    """

    def __init__(
        self,
        vocab_size: int,
        n_tabular_features: int,
        d_model: int = 64,
        n_heads: int = 1,
        n_layers: int = 2,
        dim_feedforward: int = 64,
        max_len: int = 45,
        dropout: float = 0.1,
        hidden: int = None,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, d_model, n_heads, n_layers, dim_feedforward, max_len, dropout)
        combined_dim = d_model + n_tabular_features
        if hidden is None:
            self.output = nn.Linear(combined_dim, 1)
        else:
            self.output = nn.Sequential(
                nn.Linear(combined_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

    def forward(self, tabular: torch.Tensor, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        text_vec = self.text_encoder(token_ids, lengths)
        combined = torch.cat([text_vec, tabular], dim=1)
        return self.output(combined).squeeze(-1)
