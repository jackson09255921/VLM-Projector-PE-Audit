# Adopted from FastChat / Stanford Alpaca (license omitted)

import os
import math
import json
import torch
import transformers

from dataclasses import dataclass
from typing import Any, Dict, List

from torch.utils.data import Dataset
from transformers import HfArgumentParser, set_seed, AutoConfig
from transformers.utils import logging as hf_logging

import llava.data.dataset as dataset
import llava.data.datasets_mixture as datasets_mixture
from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX
from llava.data import make_supervised_data_module
from llava.mm_utils import process_image
from llava.model import (
    LlavaLlamaConfig,
    LlavaLlamaModel,
    LlavaTopDownLlamaConfig,
    LlavaTopDownLlamaModel,
)
from llava.train.args import DataArguments, ModelArguments, TrainingArguments
from llava.train.callbacks.autoresume_callback import AutoResumeCallback
from llava.train.llava_trainer import LLaVATopDownTrainer, LLaVATrainer
from llava.train.sequence_parallel import set_pg_manager
from llava.train.slurm_utils import TimeoutTerminateCallback
from llava.train.utils import (
    get_checkpoint_path,
    mprint,
    prepare_config_for_training,
    unit_test_rope_scaling,
    vision_resolution_elevation,
)
from llava.trl.trainer.utils import DPODataCollatorWithPadding

logger = hf_logging.get_logger(__name__)

if "WANDB_PROJECT" not in os.environ:
    os.environ["WANDB_PROJECT"] = "VILA"


# =========================================================================
# 工具 A：權重與 LoRA 相關工具
# =========================================================================

def get_nb_trainable_parameters(model) -> tuple[int, int]:
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        if param.__class__.__name__ == "Params4bit":
            num_bytes = (
                param.element_size()
                if hasattr(param, "element_size")
                else (param.quant_storage.itemsize if hasattr(param, "quant_storage") else 1)
            )
            num_params = num_params * 2 * num_bytes
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return trainable_params, all_param


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE and not ignore_status:
            logger.warning(f"{name}: 權重狀態不可用")
        with zero.GatheredParameters([param]):
            return param.data.detach().cpu().clone()
    return param.detach().cpu().clone()


def get_peft_state_maybe_zero_3(named_params, bias):
    to_return = {
        k: t
        for k, t in named_params
        if "lora_" in k or (bias == "all" and "bias" in k)
    }
    return {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    return {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}


def find_all_linear_names(model, lora_llm, lora_vt):
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_resampler"]
    assert lora_llm or lora_vt, "不能把所有 LoRA 目標都關掉。"
    if not lora_llm:
        multimodal_keywords += ["llm"]
    if not lora_vt:
        multimodal_keywords += ["vision_tower"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
            lora_module_names.add(name)
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """把模型存下來，並過濾掉不支援的參數"""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir, _internal_call=True)
        return

    # 取得權重並移至 CPU
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        
        # 修正：不要直接呼叫 trainer._save，因為它會亂傳參數
        # 改為直接呼叫 model.save_pretrained
        print(f"正在將模型儲存至 {output_dir}...")
        
        # 這是最保險的存法，手動排除 safe_serialization
        trainer.model.save_pretrained(
            output_dir, 
            state_dict=cpu_state_dict,
            # 這裡不傳 safe_serialization 就不會報錯
        )
        
        # 同步儲存 Tokenizer
        if trainer.tokenizer is not None:
            trainer.tokenizer.save_pretrained(output_dir)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """
    執行 Resize 並將新 Token 的權重初始化為現有權重的平均值。
    這能有效防止新加入的 Token 導致 Loss 爆炸。
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        # 計算舊 Token 的平均權重
        input_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        # 將新 Token 的權重設為平均值
        input_embeddings[-num_new_tokens:] = input_avg
        output_embeddings[-num_new_tokens:] = output_avg
        
    # 同步更新 config 的詞表大小，防止 DeepSpeed 或分散式訓練中的不一致
    model.config.vocab_size = len(tokenizer)


from peft import LoraConfig, PeftModel, get_peft_model

def load_non_lora_trainables_if_exist(model, load_path: str):
    """載入 non_lora_trainables.bin，讓非 LoRA 權重能被繼承。"""
    nl_path = os.path.join(load_path, "non_lora_trainables.bin")
    if not os.path.exists(nl_path):
        return model
    mprint(f"[LoRA] 從 {nl_path} 載入非 LoRA 權重...")
    nl_sd = torch.load(nl_path, map_location="cpu")
    cleaned_sd = {}
    for k, v in nl_sd.items():
        if k.startswith("base_model."):
            k = k[len("base_model."):]
        if k.startswith("model."):
            k = k[len("model."):]
        cleaned_sd[k] = v
    model.load_state_dict(cleaned_sd, strict=False)
    return model


def setup_lora_plugin(
    model,
    model_args,
    training_args,
    resume_from_checkpoint: bool,
    resume_path: str | None,
):
    """
    讓 LoRA 像插件：
    - resume_from_checkpoint=True：從 output_dir checkpoint 繼續訓練現有 LoRA
    - model_name_or_path 是 LoRA 目錄：繼承舊 LoRA adapter 再訓練
    - 其他：在 base model 上建立全新 LoRA adapter

    特殊情況：lora_enable=False 但 source checkpoint 有 LoRA（跨 stage 時）：
    → 自動 merge LoRA 進 base model，讓下一個 stage 繼承完整 weights。
    """
    if not training_args.lora_enable:
        # 即使本 stage 不訓練 LoRA，若 source checkpoint 帶有 LoRA adapter，
        # 仍需 merge 進來，否則 instruction following 能力會消失。
        source_adapter = os.path.join(
            model_args.model_name_or_path, "adapter_config.json"
        )
        if os.path.exists(source_adapter):
            mprint(
                "[LoRA] lora_enable=False 但 source checkpoint 包含 LoRA adapter，"
                f"自動 merge 後繼續訓練：{model_args.model_name_or_path}"
            )
            model = load_non_lora_trainables_if_exist(
                model, model_args.model_name_or_path
            )
            model = PeftModel.from_pretrained(
                model, model_args.model_name_or_path, is_trainable=False
            )
            model = model.merge_and_unload()
            mprint("[LoRA] Merge 完成，以 merged model 繼續訓練。")
        return model

    is_source_lora = os.path.exists(
        os.path.join(model_args.model_name_or_path, "adapter_config.json")
    )

    if resume_from_checkpoint:
        load_path = resume_path
        msg = "[LoRA] 中斷恢復：從 checkpoint 繼續訓練"
    elif is_source_lora:
        load_path = model_args.model_name_or_path
        msg = "[LoRA] 繼承：以既有 LoRA adapter 為起點"
    else:
        load_path = None
        msg = "[LoRA] 全新：從 base model 建立新的 LoRA adapter"

    mprint(f"{msg}{f' ({load_path})' if load_path else ''}")

    if load_path:
        model = load_non_lora_trainables_if_exist(model, load_path)
        model = PeftModel.from_pretrained(
            model,
            load_path,
            is_trainable=True,
        )
    else:
        target_modules = find_all_linear_names(
            model, training_args.lora_llm, training_args.lora_vt
        )
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            use_dora=training_args.use_dora,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
        )
        model = get_peft_model(model, lora_config)

    # mprint(model)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def save_lora_plugin_and_non_lora(model, trainer, training_args):
    """把 LoRA adapter 和非 LoRA 權重拆開存檔，方便之後作為 plugin 掛上 / 拔掉。"""
    if training_args.local_rank not in [0, -1]:
        return
    lora_state = get_peft_state_maybe_zero_3(
        model.named_parameters(), training_args.lora_bias
    )
    non_lora_state = get_peft_state_non_lora_maybe_zero_3(
        model.named_parameters(), require_grad_only=True
    )

    model.config.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir, state_dict=lora_state)
    torch.save(
        non_lora_state,
        os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
    )


# =========================================================================
# 工具 B：DPO Dataset & Collator
# =========================================================================

def make_conv(prompt, answer):
    return [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]


def load_data(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f] if path.endswith(".jsonl") else json.load(f)


class DPODataset(Dataset):
    def __init__(self, data_mixture: str, tokenizer, data_args: DataArguments):
        super().__init__()
        data_path = datasets_mixture.DATASETS_LEGACY[data_mixture].data_path
        self.list_data_dict = load_data(data_path)
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = datasets_mixture.DATASETS_LEGACY[data_mixture].image_path

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data_dict = self.list_data_dict[i].copy()
        video_path = os.path.join(self.image_folder, data_dict["video"] + ".mp4")
        images, frames_loaded = dataset.LazySupervisedDataset._load_video(
            video_path,
            getattr(self.data_args, "num_video_frames", 8),
            getattr(self.data_args, "fps", 0.0),
            self.data_args,
        )
        data_dict["images"] = torch.stack(
            [process_image(image, self.data_args, None) for image in images]
        )
        data_dict["prompt"] = "<image>\n" * frames_loaded + data_dict["prompt"].replace("<video>", "").strip()
        return data_dict


@dataclass
class DPODataCollator(DPODataCollatorWithPadding):
    tokenizer: Any = None
    pad_token_id: int = 0
    label_pad_token_id: int = IGNORE_INDEX

    def __call__(self, features: List[Dict]) -> Dict:
        tokenized_batch = []
        for feature in features:
            batch_element = self.tokenize_batch_element(
                feature["prompt"], feature["chosen"], feature["rejected"]
            )
            batch_element["images"] = feature["images"]
            tokenized_batch.append(batch_element)
        return self.collate(tokenized_batch)

    def tokenize_batch_element(self, prompt: str, chosen: str, rejected: str) -> Dict:
        batch = {}
        for k, ans in {"chosen": chosen, "rejected": rejected}.items():
            sources = make_conv(prompt, ans)
            data_dict = dataset.preprocess(sources, self.tokenizer, has_image=True)
            for type_key, tokens in data_dict.items():
                if type_key != "token_type_ids":
                    batch[f"{k}_{type_key}"] = tokens[0]
        return batch

    def collate(self, batch: List[Dict]) -> Dict:
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if "_input_ids" in k:
                    val = self.pad_token_id
                elif "_labels" in k:
                    val = self.label_pad_token_id
                else:
                    val = 0
                padded_batch[k] = torch.nn.utils.rnn.pad_sequence(
                    to_pad, batch_first=True, padding_value=val
                )
            else:
                padded_batch[k] = [ex[k] for ex in batch]
        for k in ["chosen_input_ids", "rejected_input_ids"]:
            padded_batch[k.replace("input_ids", "attention_mask")] = padded_batch[k].ne(
                self.pad_token_id
            )
        return padded_batch


# =========================================================================
# 工具 C：可訓參數設定
# =========================================================================

def set_tunable_params(model, training_args, model_args, resume_from_checkpoint=False):
    # 1. 語言模型 (LLM) 解凍邏輯
    if training_args.lora_enable and training_args.lora_llm:
        mprint("[LoRA] LLM 已由 LoRA 控制梯度")
    else:
        model.get_llm().requires_grad_(training_args.tune_language_model)
        mprint(f"[FFT] LLM 訓練狀態: {training_args.tune_language_model}")

    # 2. 視覺編碼器 (Vision Tower) 與 投影層 (Projector) 解凍邏輯
    vt = model.get_vision_tower()
    if vt:
        if training_args.lora_enable and training_args.lora_vt:
            # ✅ Vision LoRA：透過在 projector 輸入掛 hook 來啟用梯度，不破壞 ViT 結構
            try:
                proj = model.get_mm_projector()
                proj.register_forward_hook(lambda m, i, o: o.requires_grad_(True))
                mprint("[LoRA] Vision / Projector 已啟用 Input Grad Hook (via projector)")
            except Exception as e:
                mprint(f"[LoRA] Vision Input Hook 設定失敗，略過: {e}")
        else:
            # ✅ Vision FFT：全參數微調或凍結
            vt.requires_grad_(training_args.tune_vision_tower)
            mprint(f"[FFT] Vision Tower 訓練狀態: {training_args.tune_vision_tower}")

        # ✅ Multimodal Projector：特徵投影層解凍
        model.get_mm_projector().requires_grad_(training_args.tune_mm_projector)
        mprint(f"[FFT] MM Projector 訓練狀態: {training_args.tune_mm_projector}")

        if training_args.reinit_pos_embed and training_args.zero_init_pos_embed:
            raise ValueError(
                "reinit_pos_embed and zero_init_pos_embed are mutually exclusive"
            )

        if (
            not resume_from_checkpoint
            and (training_args.reinit_pos_embed or training_args.zero_init_pos_embed)
            and training_args.tune_mm_projector
        ):
            proj = model.get_mm_projector()
            if not (
                hasattr(proj, "pos_embed_module")
                and hasattr(proj.pos_embed_module, "high_res_pos_embed")
            ):
                raise ValueError(
                    "Learned-grid initialization was requested, but the projector "
                    "does not expose high_res_pos_embed"
                )
            if training_args.reinit_pos_embed:
                torch.nn.init.normal_(proj.pos_embed_module.high_res_pos_embed, std=0.02)
                mprint("[PE] Reset LearnedPosEmbed.high_res_pos_embed → normal(std=0.02)")
            else:
                torch.nn.init.zeros_(proj.pos_embed_module.high_res_pos_embed)
                mprint("[PE] Reset LearnedPosEmbed.high_res_pos_embed → zeros")
        elif resume_from_checkpoint and (
            training_args.reinit_pos_embed or training_args.zero_init_pos_embed
        ):
            mprint("[PE] Resume detected: preserving checkpoint positional weights")

    # 3. PS3 (Top-down Selection) 關鍵解凍邏輯
    if model_args.ps3 and model_args.look_close_mode == "after_prompt":
        # A. 解凍 Top-down Prompt Head (藏在 Projector 裡面，負責接收 LLM 指令)
        if hasattr(model, "get_top_down_prompt_head"):
            prompt_head = model.get_top_down_prompt_head()
            if prompt_head is not None:
                prompt_head.requires_grad_(training_args.tune_top_down_selection)
                mprint(f"[PS3] Top-down Prompt Head 已解凍: {training_args.tune_top_down_selection}")

        # B. 解凍 Vision Tower 內部的 Token Selection 參數 (核心：控制模型看哪裡)
        if vt and hasattr(vt.vision_tower.vision_model, "token_selection_param_names"):
            target_params = vt.vision_tower.vision_model.token_selection_param_names()
            for n, p in vt.vision_tower.named_parameters():
                if any(tn in n for tn in target_params):
                    p.requires_grad_(training_args.tune_top_down_selection)
            mprint(f"[PS3] Vision Token Selection 參數已解凍: {training_args.tune_top_down_selection}")
        
        mprint(f"[PS3] Token Selection 訓練模式已啟用")
        
# =========================================================================
# 核心流程：重構後的 train()
# =========================================================================

def parse_args_and_seed():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.run_name = os.getenv(
        "RUN_NAME", training_args.output_dir.split("/")[-1]
    )
    set_seed(training_args.seed)
    return model_args, data_args, training_args


def setup_seq_parallel(training_args):
    if training_args.seq_parallel_size > 1:
        set_pg_manager(
            training_args.seq_parallel_size,
            training_args.seq_parallel_ring_size,
            training_args.seq_parallel_ring_type,
        )
        mprint(f"Sequence parallelism is enabled, SP = {training_args.seq_parallel_size}")


def build_compute_and_bnb_args(training_args):
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else torch.bfloat16
        if training_args.bf16
        else torch.float32
    )
    bnb_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        mprint(f"[配置]啟用 {training_args.bits}-bit 量化訓練 (QLORA 模式)...")
        bnb_args["device_map"] = {"": training_args.device}
        bnb_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=["lm_head"],
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type,
        )
    return compute_dtype, bnb_args


def load_base_model_and_config(model_args, data_args, training_args, bnb_args):
    # 取得 checkpoint 資訊
    resume_path, continue_training = get_checkpoint_path(training_args.output_dir)
    
    # 新增 : quick_Store logic
    if training_args.quick_store:
        if not resume_path:
            print(f"quick store 模式下，找不到 checkpoint，請先進行一次完整訓練。")
            exit(1)
        print(f"quick store 模式下，從 {resume_path} 載入模型，跳過訓練。")
        resume_from_checkpoint = True
        continue_training = True

    elif not continue_training:
        print(f"模型已在 {training_args.output_dir}，跳過訓練。")
        exit(0)

    resume_from_checkpoint = bool(resume_path)

    if resume_from_checkpoint and not training_args.lora_enable:
        # 非 LoRA：和原版一樣，直接從 checkpoint config 還原
        config = AutoConfig.from_pretrained(resume_path, trust_remote_code=True)
        config.resume_path = resume_path
        model_cls = eval(config.architectures[0])
    else:
        # 首次訓練或 LoRA 情境：用 base model config
        model_cls, config_cls = (
            (LlavaTopDownLlamaModel, LlavaTopDownLlamaConfig)
            if model_args.ps3
            else (LlavaLlamaModel, LlavaLlamaConfig)
        )
        
        # 讀取目前這個 model_name_or_path 的 config (可能是 base，也可能是SFT/LoRA)
        raw_cfg = config_cls.from_pretrained(
            model_args.model_name_or_path, 
            resume=resume_from_checkpoint,
        )
        
        # mprint(f"{raw_cfg}")
        
        # 嘗試從 llm_cfg._name_or_path 反推出 base_root (到 /model 為止)
        base_root = None 
        try:
            llm_cfg = getattr(raw_cfg, "llm_cfg", None)
            if isinstance(llm_cfg, dict):
                llm_path = llm_cfg.get("_name_or_path", None)
            else:
                llm_path = getattr(llm_cfg, "_name_or_path", None)
        except Exception:
            llm_path = None
            
        
        if isinstance(llm_path, str) and "/llm" in llm_path:
            base_root = llm_path.split("/llm")[0]
        
        if base_root is not None and os.path.isdir(base_root):
            # 給 Lora 的 config，透過記錄 llm_cfg 裡面的 base model 來建立模型
            mprint(f"[Config] 偵測到 SFT/LoRA config，llm_cfg._name_or_path 反推 base_root: {base_root}")
            config = config_cls.from_pretrained(
                base_root, 
                resume=resume_from_checkpoint,
            )
            print("目前的 config:", config)

            # resume_path 設定成 SFT 目錄
            # config.resume_path = model_args.model_name_or_path 
            
            '''修正子模組'''
            try:
                # llm_cfg:base_root/llm
                if hasattr(config, "llm_cfg"):
                    if isinstance(config.llm_cfg, dict):
                        config.llm_cfg["_name_or_path"] = os.path.join(base_root, "llm")
                    else:
                        setattr(config.llm_cfg, "_name_or_path", os.path.join(base_root, "llm"))
                
                # mm_projector_cfg: base_root/mm_projector
                if hasattr(config, "mm_projector_cfg"):
                    if isinstance(config.mm_projector_cfg, dict):
                        config.mm_projector_cfg["_name_or_path"] = os.path.join(base_root, "mm_projector")
                    else:
                        setattr(
                            config.mm_projector_cfg,
                            "_name_or_path",
                            os.path.join(base_root, "mm_projector"),
                        )
                
                # vision tower_cfg: base_root/vision_tower
                if hasattr(config, "vision_tower_cfg"):
                    if isinstance(config.vision_tower_cfg, dict):
                        config.vision_tower_cfg["_name_or_path"] = os.path.join(base_root, "vision_tower")
                    else:
                        setattr(
                            config.vision_tower_cfg,
                            "_name_or_path",
                            os.path.join(base_root, "vision_tower"),
                        )
            except Exception as e:
                mprint(f"[Config] 修正子模組 _name_or_path 時發生錯誤: {e}")
            
            # mprint("[Config] 最終使用的 config：")
            # mprint(config.llm_cfg)
            # mprint(config.mm_projector_cfg)
            # mprint(config.vision_tower_cfg)

            
        else:
            config = raw_cfg
            if getattr(config, "resume_path", None) is not None:
                config.resume_path = model_args.model_name_or_path
    

    # 共用的 config 後處理
    prepare_config_for_training(config, model_args, training_args, data_args)

    mprint("[初始化] 正在建立模型... (Flash Attention 2)")
    model = model_cls(
        config=config,
        attn_implementation="flash_attention_2",
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
        **bnb_args,
    )
    model.llm.config.use_cache = False

    # PS3 相關額外屬性（維持你原本的）
    if getattr(config, "num_look_close", None) is not None:
        model.num_look_close = config.num_look_close
    if getattr(config, "num_token_look_close", None) is not None:
        model.num_token_look_close = config.num_token_look_close
    if getattr(config, "look_close_mode", None) is not None:
        model.look_close_mode = config.look_close_mode

    return model, config, resume_from_checkpoint, resume_path

def quick_store_only(model, trainer, training_args, model_args, resume_from_checkpoint):
    ''' 快速存檔模式，只存模型不訓練 '''
    trainer.save_state()
    
    model.llm.config.use_cache = True  
    model.config.resume_path = model.config._name_or_path = training_args.output_dir
    
    if training_args.lora_enable:
        save_lora_plugin_and_non_lora(model, trainer, training_args)
        mprint(f"[Quick Store] LoRA 模型已儲存至 {training_args.output_dir}")
    else:
        safe_save_model_for_hf_trainer(trainer, training_args.output_dir)
        mprint(f"[Quick Store] 模型已儲存至 {training_args.output_dir}")
    print("Quick store 模式完成，結束程式。")
    exit(0)

def maybe_load_projector_weights(model, model_args, resume_path, training_args):
    if (not resume_path or training_args.lora_enable) and model_args.mlp_path:
        mprint(f"[對齊] 正在從 {model_args.mlp_path} 載入 Projector 權重...")
        state_dict = torch.load(model_args.mlp_path, map_location="cpu")
        mapping = {
            "0.weight": "layers.1.weight",
            "0.bias": "layers.1.bias",
            "1.weight": "layers.2.weight",
            "1.bias": "layers.2.bias",
            "3.weight": "layers.4.weight",
            "3.bias": "layers.4.bias",
        }
        state_dict_new = {}
        for old_k, new_k in mapping.items():
            if old_k in state_dict:
                state_dict_new[new_k] = state_dict[old_k]
        model.get_mm_projector().load_state_dict(state_dict_new, strict=False)


def setup_vision_and_rope(model, config, training_args):
    # mprint(model)
    vision_resolution_elevation(model, config)
    if unit_test_rope_scaling(model, model.llm.config, training_args):
        return False
    return True


def setup_generation_and_checkpointing(model, training_args):
    gen_cfg = getattr(model.llm, "generation_config", None)
    if gen_cfg is not None:
        if (not gen_cfg.do_sample) and any(
            p is not None and p != 1.0 for p in [gen_cfg.temperature, gen_cfg.top_p]
        ):
            gen_cfg.do_sample = True
            mprint("[Config] 已自動開啟 do_sample=True 以適配採樣參數")

    if training_args.gradient_checkpointing:
        mprint("🚀 [優化] 啟用 Gradient Checkpointing 與 Input Grads...")
        if hasattr(model.llm, "enable_input_require_grads"):
            model.llm.enable_input_require_grads()
        else:
            model.get_input_embeddings().register_forward_hook(
                lambda m, i, o: o.requires_grad_(True)
            )
        mprint("🚀 Gradient Checkpointing 已啟用")


def setup_tokenizer_and_special_tokens(model, training_args):
    tokenizer = model.tokenizer
    
    # 1. 建立特殊 Token 字典
    # 這裡除了 BOS/PAD，我們強制加入 VILA-HD 需要的所有標籤
    # 即使 Stage 1 沒用到文字內容，註冊它們也是為了解決 49153 越界報錯
    special_tokens = {
        "additional_special_tokens": ["<st>", "<ed>", "<bbox>", "<image>"]
    }
    
    # 2. 處理基礎 Token (保持原本邏輯)
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = "[BOS]"
    
    # SmolLM2 通常 pad = unk，這裡確保它存在
    tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = "[PAD]"

    # 3. 🚀 執行核心 Resize 邏輯
    # 這一步會把詞表從 49154 擴張到 49158 以上
    smart_tokenizer_and_embedding_resize(special_tokens, tokenizer, model.llm)
    
    # 4. 打印結果驗證 (這對 debug 很有幫助)
    print(f"✅ Tokenizer 處理完成：")
    print(f"   - 最終詞表大小: {len(tokenizer)}")
    print(f"   - <image> ID: {tokenizer.convert_tokens_to_ids('<image>')}")
    print(f"   - <st> ID: {tokenizer.convert_tokens_to_ids('<st>')}")
    
    return tokenizer


def maybe_prepare_llm_for_kbit(model, training_args):
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.llm = prepare_model_for_kbit_training(
            model.llm, use_gradient_checkpointing=training_args.gradient_checkpointing
        )


def setup_multimodal_and_time_tokens(model, model_args, data_args, training_args):
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.num_video_frames = getattr(data_args, "num_video_frames", 8)
        model.config.fps = getattr(data_args, "fps", 0.0)
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.vision_tower_lr = training_args.vision_tower_lr
        model.config.soft_ce_std = model_args.soft_ce_std

        num_patches = vision_tower.num_patches
        downsample_rate = model.get_mm_projector().downsample_rate
        data_args.num_image_tokens = math.ceil(num_patches ** 0.5 / downsample_rate) ** 2
        mprint(f"影像配置同步：每張圖產生 {data_args.num_image_tokens} tokens")

    tokenizer = model.tokenizer
    model.config.num_time_tokens = data_args.num_time_tokens = model_args.num_time_tokens
    model.config.time_token_format = data_args.time_token_format = model_args.time_token_format
    if model_args.num_time_tokens > 0:
        time_tokens = [
            model.config.time_token_format.format(t=t)
            for t in range(model.config.num_time_tokens)
        ]
        if tokenizer.add_tokens(time_tokens) > 0:
            model.resize_token_embeddings(len(tokenizer))
        model.config.time_token_ids = tokenizer.convert_tokens_to_ids(time_tokens)
    else:
        model.config.time_token_ids = []


def setup_conversation_template(model_args):
    template_name = (
        model_args.version
        if model_args.version in conversation_lib.conv_templates
        else "vicuna_v1"
    )
    conversation_lib.default_conversation = conversation_lib.conv_templates[template_name]
    mprint(f"📝 使用對話模板: {template_name}")


def setup_trainable_params_and_dtype(
    model, model_args, training_args, resume_from_checkpoint=False
):
    set_tunable_params(
        model, training_args, model_args, resume_from_checkpoint
    )
    t_params, a_params = get_nb_trainable_parameters(model)
    mprint(f"📊 Trainable: {t_params:,} || All: {a_params:,} || Ratio: {100 * t_params / a_params:.4f}%")
    if t_params == 0:
        logger.warning("⚠️ 警告：目前沒有任何參數被解凍，請檢查訓練設定！")

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if (
                isinstance(module, LoraLayer)
                or any(k in name for k in ["lm_head", "embed_tokens"])
            ) and training_args.bf16:
                module.to(torch.bfloat16)
            if "norm" in name:
                module.to(torch.float32)


def build_trainer(model, tokenizer, model_args, data_args, training_args, bnb_args):
    data_args.s2_scales = list(map(int, model_args.s2_scales.split(",")))
    callbacks = [AutoResumeCallback()]

    if training_args.dpo:
        from llava.trl.trainer.vila_dpo_trainer import VILADPOTrainer  # 若有這支
        assert not model_args.ps3, "DPO 暫時不支援 PS3"
        ref_model = LlavaLlamaModel(
            config=model.config,
            attn_implementation="flash_attention_2",
            model_max_length=training_args.model_max_length,
            **bnb_args,
        )
        train_dataset = DPODataset(
            tokenizer=tokenizer, data_mixture=data_args.data_mixture, data_args=data_args
        )
        data_collator = DPODataCollator(
            tokenizer=tokenizer,
            label_pad_token_id=IGNORE_INDEX,
            pad_token_id=tokenizer.pad_token_id,
        )
        training_args.sample_lens = [len(train_dataset)]
        trainer = VILADPOTrainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            args=training_args,
            beta=training_args.dpo_beta,
            dpo_alpha=1.0,
            gamma=0,
            callbacks=callbacks,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
    else:
        trainer_cls = LLaVATopDownTrainer if model_args.ps3 else LLaVATrainer
        data_module = make_supervised_data_module(
            tokenizer=tokenizer,
            data_args=data_args,
            training_args=training_args,
        )
        
        # # ===== 插入觀測代碼開始 =====
        # from llava.constants import IGNORE_INDEX
        # import torch

        # mprint("\n" + "🚀" * 10 + " [訓練前核心檢查] " + "🚀" * 10)
        # # 取得第一筆數據
        # sample = data_module['train_dataset'][0]

        # # 1. 檢查 Token 是否越界 (防止 49153 報錯)
        # max_id = sample['input_ids'].max().item()
        # vocab_size = len(tokenizer)
        # mprint(f"極大 Token ID: {max_id} (詞表大小: {vocab_size})")
        # if max_id >= vocab_size:
        #     mprint("❌ 錯誤：發現越界 Token，請檢查 Embedding Resize 是否成功！")

        # # 2. 精確對齊檢查：Input vs Labels
        # # 我們把 input_ids 和 labels 並排印出來，看模型在回答什麼
        # ids = sample['input_ids']
        # labels = sample['labels']

        # mprint("\n--- [Input 與 Label 對齊截段] ---")
        # # 找出第一個不是 -100 的索引 (即答案開始的地方)
        # start_idx = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][0].item()
        # # 印出答案開始的前 5 個 token 和後 10 個 token 進行比對
        # for i in range(max(0, start_idx - 5), min(len(ids), start_idx + 15)):
        #     token_str = tokenizer.decode([ids[i]])
        #     label_val = labels[i].item()
        #     status = "Target" if label_val != IGNORE_INDEX else "Prompt"
        #     mprint(f"Index {i:3d} | ID: {ids[i]:6d} | Token: {token_str:10s} | Role: {status}")

        # # 3. 影像 Token 密度檢查
        # image_token_id = tokenizer.convert_tokens_to_ids('<image>')
        # num_img_tokens = (ids == image_token_id).sum().item()
        # mprint(f"\n🖼️ 影像 Token 總數: {num_img_tokens}")
        # if num_img_tokens == 0:
        #     mprint("⚠️ 警告：Input 中沒有任何影像 Token，模型將無法看圖！")

        # mprint("🚀" * 30 + "\n")
        
        # # 插入在你的 mprint("🖼️ 影像 Token 總數: ...") 之後
        # mprint(f"--- [Token ID 詳情] ---")
        # image_token_id = tokenizer.convert_tokens_to_ids('<image>')
        # pad_token_id = tokenizer.pad_token_id
        # mprint(f"<image> ID: {image_token_id} | Pad ID: {pad_token_id}")

        # # 找出所有非 IGNORE_INDEX 的 input_ids 位置
        # # 看看是不是真的只有一個 <image> ID
        # all_img_indices = (ids == image_token_id).nonzero(as_tuple=True)[0]
        # mprint(f"所有 <image> 出現的 Index: {all_img_indices.tolist()}")

        # # 如果只是要檢查，可以強行在這裡停止防止進入訓練
        # exit(0)
        
        
        trainer = trainer_cls(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            callbacks=callbacks,
            **data_module,
        )
        if model_args.quantize_model in ["fp8Activation_qwen2", "fp8ActivationResidual_qwen2"]:
            from llava.model.coat.fp8_trainer import CoatFP8Trainer
            trainer._inner_training_loop = CoatFP8Trainer._inner_training_loop.__get__(
                trainer, trainer_cls
            )

    return trainer


def run_training_and_save(model, trainer, training_args, model_args, resume_from_checkpoint):
    if training_args.quick_store:
        quick_store_only(model, trainer, training_args, model_args, resume_from_checkpoint)
        return
    
    
    curr_vram = torch.cuda.memory_allocated() / 1024**3
    print(f"✅ DataLoader 長度: {len(trainer.get_train_dataloader())}")
    print(f"✅ 訓練樣本總數: {len(trainer.train_dataset)}")
    print(f"[GPU 顯存佔用] Trainer 啟動前: {curr_vram:.2f} GB")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if training_args.debug_e2e:
        exit()

    trainer.save_state()
    
    model.llm.config.use_cache = True
    model.config.resume_path = model.config._name_or_path = training_args.output_dir

    if training_args.lora_enable:
        save_lora_plugin_and_non_lora(model, trainer, training_args)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


def train():
    model_args, data_args, training_args = parse_args_and_seed()
    
    resume_path, _= get_checkpoint_path(training_args.output_dir)
    if training_args.quick_store and not resume_path:
        print(f"quick store 模式下，找不到 checkpoint，請先進行一次完整訓練。")
        exit(1)
    
    
    setup_seq_parallel(training_args)
    _, bnb_args = build_compute_and_bnb_args(training_args)
    model, config, resume_from_checkpoint, resume_path = load_base_model_and_config(
        model_args, data_args, training_args, bnb_args
    )
    maybe_load_projector_weights(model, model_args, resume_path, training_args)
    if not setup_vision_and_rope(model, config, training_args):
        return
    setup_generation_and_checkpointing(model, training_args)
    tokenizer = setup_tokenizer_and_special_tokens(model, training_args)
    maybe_prepare_llm_for_kbit(model, training_args)
    model = setup_lora_plugin(model, model_args, training_args, resume_from_checkpoint, resume_path)
    print(model)
    setup_multimodal_and_time_tokens(model, model_args, data_args, training_args)
    setup_conversation_template(model_args)
    setup_trainable_params_and_dtype(
        model, model_args, training_args, resume_from_checkpoint
    )
    trainer = build_trainer(model, tokenizer, model_args, data_args, training_args, bnb_args)
    run_training_and_save(model, trainer, training_args, model_args, resume_from_checkpoint)


if __name__ == "__main__":
    train()
