"""The torch modules behind the sequence adapters (§10.3).

Importing this module imports torch, which is why nothing imports it at package
import time: `alphacore.models.sequence` pulls it in inside a function, and
`import alphacore` on a machine with no `dl` extra never reaches it. Keeping the
networks here rather than defining them inside a factory function means they
stay picklable — a class defined in a closure is not.

The networks are deliberately small. Two thousand daily bars of a single symbol
support a few thousand parameters, not a few million, and §19's warning about
fantasy results applies with more force to a model that can memorise its
training fold in ten epochs.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

__all__ = ["RecurrentClassifier", "TemporalConvNet"]


class RecurrentClassifier(nn.Module):
    """An LSTM or GRU over a window, read out at its last step.

    Bidirectionality is legal here and only here: the window contains bars up to
    and including the row being scored and nothing after it, so a backward pass
    over it reads the recent past in reverse, not the future. A bidirectional
    pass over a window *centred* on the row would be a leak, which is why the
    window builder only ever emits trailing windows (§10.3).
    """

    def __init__(
        self,
        n_features: int,
        *,
        cell: str,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
    ) -> None:
        """Build the recurrent stack and its linear head.

        Args:
            n_features: Number of input features per bar.
            cell: `lstm` or `gru`.
            hidden_size: Units per direction.
            num_layers: Stacked recurrent layers.
            bidirectional: Whether to read the window in both directions.
            dropout: Dropout applied between layers and before the head.
        """
        super().__init__()
        rnn_type = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_type(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a batch of windows.

        Args:
            x: `(batch, seq_len, n_features)` input.

        Returns:
            A `(batch,)` tensor of logits.
        """
        output, _ = self.rnn(x)
        return cast("torch.Tensor", self.head(self.dropout(output[:, -1, :])).squeeze(-1))


class _CausalConv1d(nn.Module):
    """A dilated 1-D convolution that cannot see forward in time."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ) -> None:
        """Build the convolution and record its left padding.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Kernel width in bars.
            dilation: Spacing between kernel taps.
        """
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve with left-only padding.

        Args:
            x: `(batch, channels, seq_len)` input.

        Returns:
            A `(batch, out_channels, seq_len)` tensor whose step `t` depends on
            steps `<= t` only.
        """
        return cast("torch.Tensor", self.conv(nn.functional.pad(x, (self.pad, 0))))


class TemporalConvNet(nn.Module):
    """A small dilated causal TCN read out at its last step."""

    def __init__(
        self,
        n_features: int,
        *,
        channels: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        """Build the dilated stack and its linear head.

        Args:
            n_features: Number of input features per bar.
            channels: Channels per convolutional layer.
            num_layers: Layers, whose dilation doubles at each one.
            kernel_size: Kernel width in bars.
            dropout: Dropout after each layer.
        """
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = n_features
        for layer in range(num_layers):
            layers += [
                _CausalConv1d(in_channels, channels, kernel_size, dilation=2**layer),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_channels = channels
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a batch of windows.

        Args:
            x: `(batch, seq_len, n_features)` input.

        Returns:
            A `(batch,)` tensor of logits.
        """
        features = self.body(x.transpose(1, 2))
        return cast("torch.Tensor", self.head(features[:, :, -1]).squeeze(-1))
