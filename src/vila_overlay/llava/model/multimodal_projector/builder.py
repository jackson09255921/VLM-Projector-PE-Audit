# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/haotian-liu/LLaVA/

import glob
import os

import torch
from transformers import PretrainedConfig, PreTrainedModel

from .base_projector import MultimodalProjector, MultimodalProjectorConfig


def _remap_pos_embed_from_checkpoint(projector, model_type_or_path):
    """
    HuggingFace from_pretrained bypasses nn.Module._load_from_state_dict,
    so the key remap (high_res_pos_embed → pos_embed_module.high_res_pos_embed)
    never fires.  This function does it explicitly after from_pretrained.
    """
    if not hasattr(projector, "pos_embed_module"):
        return

    ckpt_files = glob.glob(os.path.join(model_type_or_path, "*.safetensors"))
    if not ckpt_files:
        ckpt_files = glob.glob(os.path.join(model_type_or_path, "*.bin"))
    if not ckpt_files:
        return

    ckpt_path = ckpt_files[0]
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(ckpt_path)
    else:
        sd = torch.load(ckpt_path, map_location="cpu")

    if "high_res_pos_embed" in sd and "pos_embed_module.high_res_pos_embed" not in sd:
        if not hasattr(projector.pos_embed_module, "high_res_pos_embed"):
            # LogRetinaPosEmbed — no remap needed, weights train from scratch
            return
        param = projector.pos_embed_module.high_res_pos_embed
        loaded = sd["high_res_pos_embed"].to(dtype=param.dtype, device=param.device)
        with torch.no_grad():
            param.copy_(loaded)
        print("[build_mm_projector] Remapped high_res_pos_embed → pos_embed_module.high_res_pos_embed")


def build_mm_projector(model_type_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    if model_type_or_path is None:
        return None

    ## load from pretrained model
    if config.resume_path:
        assert os.path.exists(model_type_or_path), f"Resume mm projector path {model_type_or_path} does not exist!"
        projector = MultimodalProjector.from_pretrained(model_type_or_path, config, torch_dtype=eval(config.model_dtype))
        _remap_pos_embed_from_checkpoint(projector, model_type_or_path)
        return projector
    ## build from scratch
    else:
        print("WARNING: Building multimodal projector from scratch!")
        mm_projector_cfg = MultimodalProjectorConfig(model_type_or_path)
        mm_projector = MultimodalProjector(mm_projector_cfg, config).to(eval(config.model_dtype))
        return mm_projector
