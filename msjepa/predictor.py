from __future__ import annotations

from torch import Tensor, nn


class DensePredictor(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be at least 1.")

        layers: list[nn.Module] = []
        if depth == 1:
            layers.append(nn.Conv2d(feature_dim, feature_dim, kernel_size=1))
        else:
            layers.extend(
                [
                    nn.Conv2d(feature_dim, hidden_dim, kernel_size=1),
                    nn.GELU(),
                ]
            )
            for _ in range(depth - 2):
                layers.extend(
                    [
                        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                        nn.GELU(),
                    ]
                )
            layers.append(nn.Conv2d(hidden_dim, feature_dim, kernel_size=1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)
