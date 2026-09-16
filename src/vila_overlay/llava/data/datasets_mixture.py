"""Minimal legacy dataset registry for the ICASSP training mixture.

All paths derive from DATA_ROOT. Dataset files are not redistributed here;
obtain them under their original licenses and prepare the JSONL files named
below.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(default=None)
    meta_path: str = field(default=None)
    image_path: str = field(default=None)
    description: str = field(default=None)
    test_script: str = (None,)
    maintainer: str = (None,)
    caption_choice: str = field(default=None)
    caption_choice_2: str = field(default=None)
    start_idx: float = field(default=-1)
    end_idx: float = field(default=-1)


DATASETS_LEGACY = {}


def add_dataset(dataset):
    if dataset.dataset_name in DATASETS_LEGACY:
        warnings.warn(f"Dataset {dataset.dataset_name!r} already exists.")
    DATASETS_LEGACY[dataset.dataset_name] = dataset


def register_datasets_mixtures():
    raw_root = os.environ.get("DATA_ROOT")
    if not raw_root:
        raise RuntimeError("DATA_ROOT must point to the prepared dataset root")
    root = Path(raw_root).expanduser().resolve()

    definitions = (
        ("chartqa_train", "ChartQA/chartqa_train.jsonl", "ChartQA"),
        ("docvqa_train", "DocVQA/docvqa_train.jsonl", "DocVQA"),
        ("shareGPT4V", "ShareGPT4V/sharegpt4v_processed.jsonl", "ShareGPT4V/images"),
        ("svit_conversation", "SVIT/svit_conversation.jsonl", "SVIT/images"),
        ("svit_complex_reasoning", "SVIT/svit_complex_reasoning.jsonl", "SVIT/images"),
        ("tulu", "Tulu/tulu.jsonl", None),
    )
    for name, data, images in definitions:
        add_dataset(
            Dataset(
                dataset_name=name,
                dataset_type="torch",
                data_path=str(root / data),
                image_path=str(root / images) if images else None,
                description="ICASSP 2027 projector-PE training mixture",
            )
        )

