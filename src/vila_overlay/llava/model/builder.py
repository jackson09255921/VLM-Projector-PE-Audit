# This file is modified from https://github.com/haotian-liu/LLaVA/
# Copyright 2023 Haotian Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PretrainedConfig,
)

from llava.model import LlavaLlamaModel, LlavaTopDownLlamaModel
from llava.model.utils import is_mm_model


def _guess_base_root_from_cfg(cfg: PretrainedConfig):
    """
    從多模態 config 裡的 llm_cfg._name_or_path 推回 base_root。
    例如:
      llm_cfg._name_or_path = runs/.../align-stage1.../model/llm
      => base_root = runs/.../align-stage1.../model
    """
    llm_cfg = getattr(cfg, "llm_cfg", None)
    llm_path = None
    if isinstance(llm_cfg, dict):
        llm_path = llm_cfg.get("_name_or_path", None)
    else:
        llm_path = getattr(llm_cfg, "_name_or_path", None)

    base_root = None
    if isinstance(llm_path, str) and "/llm" in llm_path:
        base_root = llm_path.split("/llm")[0]

    # Stale local paths baked into HF-uploaded configs won't exist on other machines.
    # HF Hub IDs have exactly one slash (namespace/repo); local paths have more.
    if base_root is not None and not os.path.exists(base_root):
        if base_root.count('/') != 1:
            base_root = None

    return base_root


def _maybe_swap_pe_module(model, raw_cfg):
    """
    If the SFT config uses a different PE type than the base model (LearnedPosEmbed),
    replace mm_projector.pos_embed_module before loading non_lora weights.
    This is necessary because the HuggingFace from_pretrained path re-reads the
    base model config and ignores pos_embed_type from the SFT checkpoint.
    """
    from llava.model.multimodal_projector.positional_embeddings import (
        LearnedPosEmbed, build_pos_embed,
    )

    pe_type = getattr(raw_cfg, "pos_embed_type", "learned")
    if pe_type == "learned":
        return  # base model PE matches — nothing to do

    # Navigate to mm_projector (VILA wraps the torch model)
    inner = getattr(model, "model", model)
    mm_proj = getattr(inner, "mm_projector", None)
    if mm_proj is None or not hasattr(mm_proj, "pos_embed_module"):
        return

    if not isinstance(mm_proj.pos_embed_module, LearnedPosEmbed):
        return  # already the right type

    # Build new PE using raw_cfg for type/alpha/freqs, but read hidden_size from the
    # projector itself (raw_cfg may not have hidden_size saved in the checkpoint config).
    ref_param = next(mm_proj.parameters())
    hidden_size = ref_param.shape[-1] if ref_param.ndim >= 1 else None
    # proj.weight in the projector MLP is (out, in); layers.2 maps → hidden_size
    for name, p in mm_proj.named_parameters():
        if "layers.2.weight" in name:
            hidden_size = p.shape[0]
            break
    if hidden_size is not None and not hasattr(raw_cfg, "hidden_size"):
        raw_cfg.hidden_size = hidden_size

    new_pe = build_pos_embed(raw_cfg)
    new_pe = new_pe.to(device=ref_param.device, dtype=ref_param.dtype)
    mm_proj.pos_embed_module = new_pe
    print(f"[builder] PE swapped: LearnedPosEmbed → {type(new_pe).__name__} (hidden={hidden_size})")


def _load_non_lora_trainables(model, load_path, raw_cfg=None):
    """
    從 SFT/LoRA 目錄裡載入 non_lora_trainables.bin，打回 base model。
    會清掉常見的前綴：base_model., model.
    raw_cfg: SFT checkpoint config (used to swap PE module if needed).
    """
    nl_path = os.path.join(load_path, "non_lora_trainables.bin")
    if not os.path.exists(nl_path):
        return model

    non_lora_trainables = torch.load(nl_path, map_location="cpu")
    cleaned_sd = {}
    for k, v in non_lora_trainables.items():
        if k.startswith("base_model."):
            k = k[len("base_model.") :]
        if k.startswith("model."):
            k = k[len("model.") :]
        cleaned_sd[k] = v

    # Swap PE module BEFORE loading weights so the right keys exist in the model
    if raw_cfg is not None:
        _maybe_swap_pe_module(model, raw_cfg)

    result = model.load_state_dict(cleaned_sd, strict=False)
    print(f"[non_lora] missing_keys ({len(result.missing_keys)}): {result.missing_keys[:5]}")
    print(f"[non_lora] unexpected_keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}")
    # Verify PE weight was actually loaded
    try:
        inner = getattr(model, "model", model)
        mm_proj = getattr(inner, "mm_projector", model.mm_projector)
        pe_mod = mm_proj.pos_embed_module
        if hasattr(pe_mod, "proj"):
            w = pe_mod.proj.weight
            print(f"[non_lora] PE proj.weight after load: shape={w.shape} norm={w.float().norm().item():.4f} dtype={w.dtype} device={w.device}")
        elif hasattr(pe_mod, "high_res_pos_embed"):
            w = pe_mod.high_res_pos_embed
            print(f"[non_lora] PE high_res_pos_embed after load: shape={w.shape} norm={w.float().norm().item():.4f}")
    except Exception as e:
        print(f"[non_lora] PE verify error: {e}")
    return model


def prepare_config_for_eval(config: PretrainedConfig, kwargs: dict):
    try:
        # 舊版 config 兼容：把 mm_vision_tower 映射到 vision_tower_cfg
        if getattr(config, "vision_tower_cfg", None) is None:
            config.vision_tower_cfg = config.mm_vision_tower
    except AttributeError:
        raise ValueError(
            f"Invalid configuration! Cannot find vision_tower in config:\n{config}"
        )

    # 從 kwargs 裡拿出 torch_dtype，寫到 config.model_dtype 方便下游使用
    torch_dtype = kwargs.pop("torch_dtype", None)
    if torch_dtype is not None:
        config.model_dtype = str(torch_dtype)


def load_pretrained_model(
    model_path,
    model_name,
    model_base=None,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    **kwargs,
):
    """
    統一的 loading 邏輯：

    1. 如果 model_path 是多模態：
       1.1 先嘗試從 config 推 base_root（你的 SmolVLM2 / PS3 SFT 走這條）
       1.2 找不到 base_root 才 fallback 到舊的 model_base + LoRA 流程

    2. 如果 model_path 不是多模態：
       2.1 model_base != None => 把 model_path 當 LoRA，掛在 model_base 上並 merge
       2.2 否則直接 from_pretrained(model_path) 視為已 merge 完整模型
    """
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    # ---------- 多模態模型路徑 ----------
    if is_mm_model(model_path):
        raw_cfg = AutoConfig.from_pretrained(model_path)
        # model_base 優先（接受本地路徑或 HF Hub ID）；沒有才從 config 猜
        if model_base is not None:
            base_root = model_base
        else:
            base_root = _guess_base_root_from_cfg(raw_cfg)

        # ---------- Standalone full model (e.g. VILA-HD from HF Hub) ----------
        if base_root is None:
            print(f"[Config] base_root not found locally — loading as standalone full model: {model_path}")
            config = raw_cfg
            prepare_config_for_eval(config, kwargs)
            if "topdown" in config.model_type.lower():
                model = LlavaTopDownLlamaModel.from_pretrained(
                    model_path, config=config, low_cpu_mem_usage=True, **kwargs
                )
            else:
                model = LlavaLlamaModel.from_pretrained(
                    model_path, config=config, low_cpu_mem_usage=True, **kwargs
                )
            tokenizer = model.tokenizer

        # ---------- SFT / LoRA 結構，透過 base_root 重建 ----------
        else:
            print(f"[Config] Detected SFT/LoRA config, base_root: {base_root}")
            config = AutoConfig.from_pretrained(base_root)
            # Copy all runtime-relevant attributes from the SFT checkpoint config
            # so eval uses exactly the same model behaviour as training.
            #
            # Strategy: blacklist pure training-only attributes; copy everything else.
            # This is safer than a whitelist — new PS3/PE/model flags are picked up
            # automatically without needing to edit this file.
            #
            # Why not just use raw_cfg directly?  The mm_projector sub-module loads
            # from its own config.json inside base_root, which always gives LearnedPosEmbed.
            # _maybe_swap_pe_module (called inside _load_non_lora_trainables) handles that
            # swap after the fact, so we still need the base config as the starting point.
            _TRAINING_ONLY_ATTRS = {
                # LoRA / optimiser knobs
                "lora_enable", "lora_llm", "lora_vt", "lora_r", "lora_alpha",
                "lora_dropout", "lora_bias",
                # Gradient / memory tricks
                "ps3_grad_checkpointing", "gradient_checkpointing",
                # LR / optimiser
                "mm_projector_lr", "learning_rate", "weight_decay",
                "warmup_ratio", "lr_scheduler_type", "optim", "bits",
                # Tuning flags (only meaningful during training)
                "tune_vision_tower", "tune_top_down_selection",
                "tune_language_model",
                # Loss terms
                "token_selection_loss_weight", "train_w_gt_selection_map",
                "smooth_selection_prob_in_training",
                "enable_grounding_loss", "grounding_loss_weight",
                "enable_language_anchor", "language_anchor",
                "language_anchor_weight",
                # HF metadata (must keep base model's values)
                "_name_or_path", "resume_path", "transformers_version",
                "_commit_hash",
            }
            for _attr, _val in vars(raw_cfg).items():
                if _attr in _TRAINING_ONLY_ATTRS or _attr.startswith("_"):
                    continue
                # Only patch if the value actually differs (avoid noisy logs)
                if getattr(config, _attr, object()) != _val:
                    setattr(config, _attr, _val)
                    print(f"[builder] config patch: {_attr} = {_val!r}")
            prepare_config_for_eval(config, kwargs)

            if "topdown" in config.model_type.lower():
                model = LlavaTopDownLlamaModel.from_pretrained(
                        base_root, config=config, low_cpu_mem_usage=True, **kwargs
                    )
            else:
                model = LlavaLlamaModel.from_pretrained(
                        base_root, config=config, low_cpu_mem_usage=True, **kwargs
                    )
            tokenizer = model.tokenizer

            # non_lora_trainables (pass raw_cfg so PE module gets swapped if needed)
            model = _load_non_lora_trainables(model, model_path, raw_cfg=raw_cfg)

            # 掛 LoRA 並 merge
            from peft import PeftModel
            if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                print(f"Loading LoRA adapter from {model_path}")
                model = PeftModel.from_pretrained(model, model_path)

                # After a PE swap (_maybe_swap_pe_module), PEFT sometimes creates
                # LoRA A/B parameters as meta tensors, making the copy a no-op.
                # Detect this and reload with assign=True to actually fill them.
                meta_lora = [n for n, p in model.named_parameters()
                             if p.is_meta and "lora_" in n]
                if meta_lora:
                    print(f"[LoRA] {len(meta_lora)} meta LoRA params detected — reloading with assign=True")
                    adapter_sf = os.path.join(model_path, "adapter_model.safetensors")
                    if os.path.exists(adapter_sf):
                        from safetensors.torch import load_file as _safe_load
                        _dev = next(
                            p for p in model.parameters() if not p.is_meta
                        ).device
                        _adapter_sd = {k: v.to(_dev) for k, v in _safe_load(adapter_sf).items()}
                        model.load_state_dict(_adapter_sd, strict=False, assign=True)
                        print(f"[LoRA] Re-load done — "
                              f"meta params now: {sum(p.is_meta for p in model.parameters())}")
                    else:
                        print("[LoRA] WARNING: adapter_model.safetensors not found, LoRA may be missing!")

                print("Merging LoRA weights...")
                model = model.merge_and_unload()
                print("Model is loaded (base + non_lora + LoRA merged)")


    # ---------- 純語言模型路徑 ----------
    else:
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, **kwargs
            )
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            # 已 merge 完整 LM
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=False, legacy=False
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )

    # ---------- 通用收尾 ----------
    model.eval()
    image_processor = None
    if is_mm_model(model_path):
        model.resize_token_embeddings(len(tokenizer))
        vision_tower = model.get_vision_tower()
        if vision_tower is None:
            raise ValueError("Vision tower failed to load!")
        vision_tower.to(device=device, dtype=torch.float16)
        mm_projector = model.get_mm_projector()
        mm_projector.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model, "llm") and hasattr(model.llm.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
