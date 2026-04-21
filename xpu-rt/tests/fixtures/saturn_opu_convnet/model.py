"""ConvNet fixture for the Saturn OPU Zephyr/FireSim bring-up.

Architecture is a 1:1 copy of Merlin's ``opu_bench_convnet`` (ResNet-style
CNN with 3 stages). It is kept self-contained inside the fixture so the
CompGen test suite does not have to import from ``merlin`` at runtime.

The fixture exposes:

* :func:`build_model` — returns an eval-mode :class:`ConvNet` instance.
* :func:`default_inputs` — returns the canonical ``torch.randn`` input
  tuple (``(1, 3, 64, 64)``) with a fixed RNG seed so goldens stay
  deterministic across test runs.

Used by the Saturn OPU integration tests (`tests/integration/
test_saturn_opu_convnet_*.py`) and by the ``compgen generate`` CLI in
the Zephyr bring-up recipe.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_SEED = 0xC0FFEE


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.shortcut(x))


class ConvNet(nn.Module):
    """Small ResNet-style CNN: 3 stages × 2 blocks, channels 32→64→128.

    Mirrors ``merlin.models.opu_bench_suite.opu_bench_models.ConvNet``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.stage1 = nn.Sequential(_ConvBlock(32, 32), _ConvBlock(32, 32))
        self.stage2 = nn.Sequential(_ConvBlock(32, 64, stride=2), _ConvBlock(64, 64))
        self.stage3 = nn.Sequential(_ConvBlock(64, 128, stride=2), _ConvBlock(128, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def build_model() -> ConvNet:
    """Return an eval-mode ConvNet with deterministic parameters."""
    torch.manual_seed(_SEED)
    model = ConvNet()
    model.eval()
    return model


def default_inputs() -> tuple[torch.Tensor, ...]:
    """Return the canonical input tuple (seed-pinned so goldens are stable)."""
    gen = torch.Generator().manual_seed(_SEED)
    return (torch.randn((1, 3, 64, 64), generator=gen),)


__all__ = ["ConvNet", "build_model", "default_inputs"]
