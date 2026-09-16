#!/usr/bin/env python3
"""CPU smoke test for PE shapes, initialization, and parameter counts."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from projector_pe import (  # noqa: E402
    FourierPosEmbed,
    LearnedPosEmbed,
    LogRetinaPosEmbed,
    NoSpatialPosEmbed,
    PolarPosEmbed,
)


def count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def main() -> int:
    hidden = 3584
    h = w = 27
    selection = torch.zeros(2, 8, 8)
    selection[0, 1:3, 2:5] = 1
    selection[1, 5:8, 4:7] = 1

    modules = {
        "N": NoSpatialPosEmbed(hidden),
        "A": LearnedPosEmbed(729, hidden),
        "C": FourierPosEmbed(hidden, 32),
        "E": LogRetinaPosEmbed(hidden, alpha=1.0, num_freqs=32, dynamic_center=True),
        "F": PolarPosEmbed(hidden, 32),
    }
    expected = {"N": 0, "A": 2_612_736, "C": 462_336, "E": 462_336, "F": 462_336}

    for code, module in modules.items():
        actual = count(module)
        assert actual == expected[code], (code, actual)
        output = module.get_pe(h, w, selection if code == "E" else None)
        expected_batch = 2 if code == "E" else 1
        assert output.shape == (expected_batch, h * w, hidden), (code, output.shape)
        # LearnedPosEmbed is constructed at zero, then condition A is
        # reinitialized to Normal(0, 0.02) by the training hook. A0 leaves it
        # at zero. Functional projections and N deliberately start at zero.
        if code in {"N", "C", "E", "F"}:
            assert torch.count_nonzero(output) == 0, f"{code} must start at zero"
        print(f"PASS {code}: shape={tuple(output.shape)} parameters={actual:,}")

    print("PASS parameter ratio: {:.6f}".format(expected["A"] / expected["F"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
