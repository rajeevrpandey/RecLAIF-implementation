import argparse
from pathlib import Path

from reclaif.training import SFTTrainingConfig, train_sft_from_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SFT recommender from RecLAIF JSONL artifacts.")
    parser.add_argument("--model", required=True, help="Base model name or local path.")
    parser.add_argument(
        "--train-jsonl",
        default="artifacts/sft_records.jsonl",
        help="Path to the SFT JSONL artifact.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/sft",
        help="Directory where the SFT checkpoint will be saved.",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--use-peft", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SFTTrainingConfig(
        model_name_or_path=args.model,
        train_jsonl=Path(args.train_jsonl),
        output_dir=Path(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        use_peft=args.use_peft,
        bf16=args.bf16,
        fp16=args.fp16,
        resume_from_checkpoint=args.resume_from_checkpoint,
        seed=args.seed,
    )
    output_dir = train_sft_from_jsonl(config)
    print(f"SFT training finished. Checkpoint saved to: {output_dir}")


if __name__ == "__main__":
    main()
