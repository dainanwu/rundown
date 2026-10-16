"""2-layer LSTM for sequence-to-sequence regression.

The model contract rundown assumes (see ``rundown.engine``):

    forward(x: Tensor[B, T, F], warm_start: int) -> Tensor[B, T']

Where T' = T - warm_start. The ``head`` Linear maps the LSTM's last hidden
state at each time step to a scalar output (here: OT degC).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SequenceLSTM(nn.Module):
    """Stacked LSTM applied across the whole input, head per time step.

    Input  ``x``: [B, T, F]  -- we drop the warm-start prefix.
    Output ``y`` : [B, T'] where T' = T - warm_start.
    """

    def __init__(
        self,
        in_dim: int = 6,
        hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, warm_start: int = 0) -> torch.Tensor:
        h, _ = self.lstm(x[:, warm_start:])  # [B, T', hidden]
        return self.head(h).squeeze(-1)       # [B, T']