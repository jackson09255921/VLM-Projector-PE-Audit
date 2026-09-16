import hashlib
import os
import os.path as osp
from itertools import chain
from typing import Any, List, Optional

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from torch.utils.data import ConcatDataset, Dataset, Subset
from transformers import PreTrainedTokenizer

from llava.data.datasets_mixture import DATASETS_LEGACY
from llava.train.args import DataArguments, TrainingArguments
from llava.utils import io
from llava.utils.logging import logger

__all__ = ["DATASETS", "MIXTURES", "register_datasets", "register_mixtures", "parse_mixture", "build_dataset"]

def load_dataset_yaml(name):
    fname = f"{name}.yaml" if not name.endswith(".yaml") else name
    repo_path = osp.join(osp.dirname(__file__), "registry", "datasets", fname)
    if osp.exists(repo_path):
        return repo_path
    abs_path = osp.expanduser(fname)
    if osp.exists(abs_path):
        return abs_path
    raise FileNotFoundError(f"Dataset '{name}' is not found in the {repo_path} or {abs_path}.")

def register_datasets(name: Optional[str] = None):
    if name is None:
        name = os.environ.get("VILA_DATASETS", "default")
        logger.info(f"Registering datasets from environment: '{name}'.")
    dataset_meta = {}
    for _name in name.split(","):
        yamlpath = load_dataset_yaml(_name)
        logger.info(f"Registering datasets from: '{yamlpath}'.")
        meta = io.load(yamlpath)
        dataset_meta.update(meta)
    return dataset_meta

def register_mixtures():
    return io.load(os.path.join(os.path.dirname(__file__), "registry", "mixtures.yaml"))

DATASETS = register_datasets()
MIXTURES = register_mixtures()

def parse_mixture(mixture: str) -> List[str]:
    names = mixture.split("+") if "+" in mixture else [mixture]
    while any(name in MIXTURES for name in names):
        names = list(chain(*[MIXTURES.get(name, [name]) for name in names]))
    return sorted(names)

class RepeatedDataset(Dataset):
    def __init__(self, dataset: Dataset, times: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.times = times
    def __len__(self) -> int:
        return len(self.dataset) * self.times
    def __getitem__(self, index: int) -> Any:
        return self.dataset[index % len(self.dataset)]

def get_world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def apply_fixed_fraction_subset(dataset: Dataset, dataset_name: str) -> Dataset:
    """Apply an opt-in deterministic subset independently to each source.

    The default path is unchanged.  This is intentionally controlled through
    environment variables so archived training commands remain reproducible.
    """
    raw_fraction = os.environ.get("VILA_FIXED_SUBSET_FRACTION")
    if raw_fraction is None:
        return dataset

    fraction = float(raw_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("VILA_FIXED_SUBSET_FRACTION must be in (0, 1].")

    base_seed = int(os.environ.get("VILA_FIXED_SUBSET_SEED", "0"))
    source_offset = int.from_bytes(
        hashlib.sha256(dataset_name.encode("utf-8")).digest()[:8], "big"
    )
    generator = torch.Generator().manual_seed((base_seed + source_offset) % (2**63 - 1))
    subset_size = max(1, int(len(dataset) * fraction))
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size]
    indices = torch.sort(indices).values.tolist()
    fingerprint = hashlib.sha256(
        ",".join(str(index) for index in indices).encode("utf-8")
    ).hexdigest()[:16]
    logger.warning(
        f"Fixed subset: source={dataset_name} fraction={fraction:.6f} "
        f"seed={base_seed} size={subset_size}/{len(dataset)} "
        f"sha256[:16]={fingerprint}"
    )
    return Subset(dataset, indices)

def build_dataset(
    mixture: str,
    data_args: DataArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    # 🚀 在 build 階段先標註目前的 Stage
    current_stage = getattr(data_args, "training_stage", "unknown")
    logger.warning(f"🚀 Training VILA [Stage: {current_stage}] with mixture '{mixture}'.")
    
    datasets = []
    for name in parse_mixture(mixture):
        mixture_entry = name
        slice_subset = False
        subset_size = None
        slice_folder = None

        if "@" in name:
            name, subset_choice = name.split("@")
            slice_subset = True
            try:
                s = subset_choice.lower()
                multiplier = 1
                if s.endswith("k"):
                    multiplier = 1000
                    s = s[:-1]
                elif s.endswith("m"):
                    multiplier = 1000000
                    s = s[:-1]
                subset_size = int(float(s) * multiplier)
                subset_choice = None
            except ValueError:
                slice_folder = os.environ.get("VILA_SLICE_FOLDER")
                if not slice_folder:
                    raise RuntimeError(
                        "VILA_SLICE_FOLDER is required for named dataset slices"
                    )

        if "*" in name:
            name, times_str = name.split("*")
            times = float(times_str)
        else:
            times = 1.0

        if DATASETS is not None and name in DATASETS:
            if name in DATASETS_LEGACY:
                logger.warning(f"Dataset '{name}' exists in both new and legacy registries. Using the new one.")
            dataset = instantiate(DATASETS[name], _partial_=True)(
                tokenizer=tokenizer,
                data_args=data_args,
                global_batch_size=(
                    training_args.per_device_train_batch_size
                    * get_world_size()
                    * training_args.gradient_accumulation_steps
                ),
            )
        elif name in DATASETS_LEGACY:
            # 🚀 進入 Legacy 模式處理 (你的 qa_grounding 資料都在這)
            dataset = build_dataset_legacy(
                name,
                data_args=data_args,
                training_args=training_args,
                tokenizer=tokenizer,
            )
        else:
            raise ValueError(f"Dataset '{name}' is not found in the registries.")

        if slice_subset:
            if subset_size is not None:
                indices = list(range(min(subset_size, len(dataset))))
                dataset = Subset(dataset, indices)
            elif subset_choice is not None and slice_folder is not None:
                slice_json = osp.join(slice_folder, subset_choice, f"{name}.json")
                ignore_indices = io.load(slice_json)
                indices = sorted(list(set(range(len(dataset))) - set(ignore_indices)))
                dataset = Subset(dataset, indices)

        dataset = apply_fixed_fraction_subset(dataset, mixture_entry)

        if times <= 0: continue
        int_times = int(times)
        frac = times - int_times
        replicated = []
        if int_times > 0:
            replicated.append(dataset if int_times == 1 else RepeatedDataset(dataset, int_times))
        if frac > 0:
            frac_indices = list(range(max(1, int(len(dataset) * frac))))
            replicated.append(Subset(dataset, frac_indices))

        datasets.append(ConcatDataset(replicated))

    return ConcatDataset(datasets)

def build_dataset_legacy(
    name: str,
    data_args: DataArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    from llava.data.dataset import (
        LazyCCSWebDataset,
        LazyCoyoDataset,
        LazyCoyoWebDataset,
        LazyMMC4Dataset,
        LazyQAGroundingDataset,
        LazySupervisedDataset,
        LazyVideoWebDataset,
        LazyWDSDataset,
        LazyHighResQADataset,
        LazyCanvasVQADataset,
    )
    
    dataset_cfg = DATASETS_LEGACY[name]
    dataset_type = dataset_cfg.dataset_type

    # 🚀 核心映射邏輯
    if dataset_type == "torch":
        # Stage 1 / 1.5 專用
        dataset_cls = LazySupervisedDataset
    elif dataset_type == "qa_grounding":
        # Stage 2 / 3 專用 (PS3 機制核心)
        dataset_cls = LazyQAGroundingDataset
    elif dataset_type == "canvas_vqa":
        # Stage 3 專用：將低解析度 VQA 圖貼到大背景畫布上，
        # 迫使模型主動透過 top-down selection 定位感興趣區域。
        # Stage 1 / Stage 2 完全不使用此類型，不受影響。
        dataset_cls = LazyCanvasVQADataset
    elif dataset_type == "high_res_qa":
        # 特殊高解析 QA (如 ChartQA 專用類別)
        dataset_cls = LazyHighResQADataset
    elif dataset_type == "wds": dataset_cls = LazyWDSDataset
    elif dataset_type == "mmc4": dataset_cls = LazyMMC4Dataset
    elif dataset_type == "coyo": dataset_cls = LazyCoyoDataset
    elif dataset_type == "ccs-wds": dataset_cls = LazyCCSWebDataset
    elif dataset_type == "video-wds": dataset_cls = LazyVideoWebDataset
    else:
        raise NotImplementedError(f"{dataset_type} is not supported.")

    # 傳遞 meta 資訊
    data_args.meta_path = getattr(dataset_cfg, "meta_path", None)
    data_args.caption_choice = getattr(dataset_cfg, "caption_choice", None)
    data_args.caption_choice_2 = getattr(dataset_cfg, "caption_choice_2", None)

    return dataset_cls(
        tokenizer=tokenizer,
        data_path=dataset_cfg.data_path,
        image_folder=getattr(dataset_cfg, "image_path"),
        data_args=data_args,
        training_args=training_args,
    )
