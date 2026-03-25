import argparse
from pathlib import Path

from reclaif.training import DPOTrainingConfig, train_dpo_from_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DPO recommender from RecLAIF preference pairs.")
    parser.add_argument("--model", required=True, help="Base model name or local path.")
    parser.add_argument(
        "--train-jsonl",
        default="artifacts/preference_pairs.jsonl",
        help="Path to the DPO JSONL artifact.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/dpo",
        help="Directory where the DPO checkpoint will be saved.",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--use-peft", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DPOTrainingConfig(
        model_name_or_path=args.model,
        train_jsonl=Path(args.train_jsonl),
        output_dir=Path(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
        use_peft=args.use_peft,
        bf16=args.bf16,
        fp16=args.fp16,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    output_dir = train_dpo_from_jsonl(config)
    print(f"DPO training finished. Checkpoint saved to: {output_dir}")


if __name__ == "__main__":
    main()
