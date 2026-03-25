from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reclaif.io import read_jsonl_rows


@dataclass(slots=True)
class CommonTrainingConfig:
    model_name_or_path: str
    output_dir: Path
    learning_rate: float = 5e-5
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "no"
    bf16: bool = False
    fp16: bool = False
    max_length: int = 1024
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    report_to: str = "none"
    use_peft: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    resume_from_checkpoint: str | None = None


@dataclass(slots=True)
class SFTTrainingConfig(CommonTrainingConfig):
    train_jsonl: Path = Path("artifacts/sft_records.jsonl")


@dataclass(slots=True)
class DPOTrainingConfig(CommonTrainingConfig):
    train_jsonl: Path = Path("artifacts/preference_pairs.jsonl")
    beta: float = 0.1
    max_prompt_length: int = 768


def train_sft_from_jsonl(config: SFTTrainingConfig) -> str:
    transformers, trl, datasets, peft = _import_training_stack()
    tokenizer, model = _load_tokenizer_and_model(
        model_name_or_path=config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        attn_implementation=config.attn_implementation,
    )
    dataset = _build_sft_dataset(config.train_jsonl, datasets)
    peft_config = _build_peft_config(config, peft) if config.use_peft else None

    args = trl.SFTConfig(
        output_dir=str(config.output_dir),
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        eval_strategy=config.eval_strategy,
        bf16=config.bf16,
        fp16=config.fp16,
        max_length=config.max_length,
        report_to=[] if config.report_to == "none" else [config.report_to],
        completion_only_loss=True,
    )

    trainer = trl.SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    return str(config.output_dir)


def train_dpo_from_jsonl(config: DPOTrainingConfig) -> str:
    transformers, trl, datasets, peft = _import_training_stack()
    tokenizer, model = _load_tokenizer_and_model(
        model_name_or_path=config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        attn_implementation=config.attn_implementation,
    )
    dataset = _build_dpo_dataset(config.train_jsonl, datasets)
    peft_config = _build_peft_config(config, peft) if config.use_peft else None

    args = trl.DPOConfig(
        output_dir=str(config.output_dir),
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        eval_strategy=config.eval_strategy,
        bf16=config.bf16,
        fp16=config.fp16,
        max_length=config.max_length,
        beta=config.beta,
        report_to=[] if config.report_to == "none" else [config.report_to],
    )

    trainer = trl.DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    return str(config.output_dir)


def _import_training_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import datasets
        import peft
        import transformers
        import trl
    except ImportError as exc:
        raise ImportError(
            "Training dependencies are missing. Install the 'train' extra to enable SFT and DPO."
        ) from exc
    return transformers, trl, datasets, peft


def _load_tokenizer_and_model(
    *,
    model_name_or_path: str,
    trust_remote_code: bool,
    attn_implementation: str | None,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model


def _build_sft_dataset(path: Path, datasets_module: Any):
    rows = read_jsonl_rows(path)
    normalized = [
        {
            "prompt": row["prompt"],
            "completion": row["response"],
            "example_id": row.get("example_id", ""),
            "iteration": row.get("iteration", 0),
        }
        for row in rows
    ]
    return datasets_module.Dataset.from_list(normalized)


def _build_dpo_dataset(path: Path, datasets_module: Any):
    rows = read_jsonl_rows(path)
    normalized = [
        {
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "example_id": row.get("example_id", ""),
            "iteration": row.get("iteration", 0),
            "judge_reasoning": row.get("judge_reasoning", ""),
        }
        for row in rows
    ]
    return datasets_module.Dataset.from_list(normalized)


def _build_peft_config(config: CommonTrainingConfig, peft_module: Any):
    return peft_module.LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.target_modules),
    )
