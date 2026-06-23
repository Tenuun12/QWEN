import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TRITON_CACHE_DIR", str(Path(__file__).resolve().parent / ".cache" / "triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(__file__).resolve().parent / ".cache" / "torchinductor"))
import torch
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
IGNORE_INDEX = -100


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-VL on image/prompt/markdown JSONL data.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL, help="Local model directory or Hugging Face model id.")
    parser.add_argument("--dataset-file", default="qwen_dataset/qwen2_train_data.jsonl")
    parser.add_argument("--output-dir", default="local_finetuned_model")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-pixels", type=int, default=3136, help="Minimum image pixel budget for Qwen2-VL preprocessing.")
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=768 * 28 * 28,
        help="Maximum image pixel budget for Qwen2-VL preprocessing. Lower this if CUDA runs out of memory.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for a smoke run.")
    parser.add_argument("--max-steps", type=int, default=-1, help="Override training length; useful on CPU.")
    parser.add_argument("--prompt", default=(
        "Convert the following document to markdown.\n"
        "Return only the markdown with no explanation text. Do not include delimiters like ```markdown or ```html.\n\n"
        "RULES:\n"
        "  - You must include all information on the page. Do not exclude headers, footers, or subtext.\n"
        "  - Return tables in an HTML format.\n"
        "  - Charts & infographics must be interpreted to a markdown format. Prefer table format when applicable.\n"
        "  - Prefer using ☐ and ☑ for check boxes."
    ), help="Prompt to use for all training examples.")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker count.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load the base model in 4-bit with Unsloth.")
    return parser.parse_args()


def resolve_model_source(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if path.exists():
        return str(path.resolve())
    # Accept Hugging Face repo ids like "Qwen/Qwen2-VL-2B-Instruct".
    # Only treat Windows-style drive paths or explicit local-style paths as missing local paths.
    if "\\" in model_path or ":" in model_path or model_path.startswith((".", "..")):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if "/" in model_path:
        org, name = model_path.split("/", 1)
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        snapshot_root = hf_home / "hub" / f"models--{org}--{name}" / "snapshots"
        if snapshot_root.exists():
            snapshots = sorted([p for p in snapshot_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
            if snapshots:
                return str(snapshots[0].resolve())
    return model_path


def build_messages(prompt: str, output: str | None = None):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if output is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": output}]})
    return messages


def load_rgb(path: str):
    with Image.open(path) as image:
        return image.convert("RGB")


def resolve_dataset_image_path(image_value: str, dataset_dir: Path) -> Path:
    normalized = image_value.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    root_prefix = f"{dataset_dir.name}/"
    if normalized.startswith(root_prefix):
        normalized = normalized[len(root_prefix):]
    return (dataset_dir / Path(normalized)).resolve()


class JsonlVisionDataset(Dataset):
    def __init__(self, dataset_file: Path, max_samples: int | None = None):
        self.rows = []
        dataset_dir = dataset_file.parent
        with dataset_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                image_value = row.get("image")
                if isinstance(image_value, str):
                    image_path = Path(image_value)
                    if not image_path.is_absolute():
                        row["image"] = str(resolve_dataset_image_path(image_value, dataset_dir))
                self.rows.append(row)
                if max_samples is not None and len(self.rows) >= max_samples:
                    break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def pick_precision():
    if not torch.cuda.is_available():
        return torch.float32, False, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False, True
    return torch.float16, True, False


def import_unsloth():
    try:
        from unsloth import FastVisionModel
    except ImportError as exc:
        raise ImportError(
            "Unsloth is not installed in this environment. Install it first, for example:\n"
            "pip install unsloth\n"
            "or use Unsloth's CUDA-specific install command for your torch/CUDA build."
        ) from exc
    return FastVisionModel


class Qwen2VLCollator:
    def __init__(self, processor, max_length: int, prompt: str):
        self.processor = processor
        self.max_length = max_length
        self.prompt = prompt

    def __call__(self, examples):
        texts = []
        prompt_texts = []
        images = []
        for example in examples:
            texts.append(
                self.processor.apply_chat_template(
                    build_messages(self.prompt, example["output"]),
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    build_messages(self.prompt),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            images.append(load_rgb(example["image"]))

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
        for row_idx, prompt_len in enumerate(prompt_batch["attention_mask"].sum(dim=1).tolist()):
            labels[row_idx, :prompt_len] = IGNORE_INDEX
        batch["labels"] = labels
        return batch


def save_artifacts(model, processor, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))


def main():
    args = parse_args()
    dataset_file = Path(args.dataset_file)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    model_source = resolve_model_source(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_file}")
    dataset = JsonlVisionDataset(dataset_file, max_samples=args.max_samples)
    print(f"Dataset loaded with {len(dataset)} examples")
    print(f"Optimizer learning_rate={args.learning_rate}")

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in this Python environment. "
            "Current torch build: "
            f"{torch.__version__}. "
            "Unsloth requires a GPU-enabled PyTorch install."
        )

    torch_dtype, use_fp16, use_bf16 = pick_precision()
    use_gradient_checkpointing = args.gradient_checkpointing or torch.cuda.is_available()
    FastVisionModel = import_unsloth()
    print(
        "Runtime: "
        f"cuda={torch.cuda.is_available()} "
        f"dtype={torch_dtype} "
        f"fp16={use_fp16} "
        f"bf16={use_bf16} "
        f"gradient_checkpointing={use_gradient_checkpointing}"
    )

    print(f"Loading processor and model from {model_source} with Unsloth")
    model, processor = FastVisionModel.from_pretrained(
        model_name=model_source,
        max_seq_length=args.max_length,
        dtype=None if torch.cuda.is_available() else torch.float32,
        load_in_4bit=args.load_in_4bit,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.image_processor.size = {
        "shortest_edge": args.min_pixels,
        "longest_edge": args.max_pixels,
    }
    print(f"Image preprocessing pixels: min={args.min_pixels} max={args.max_pixels}")

    model.enable_input_require_grads()
    model.config.use_cache = False
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        target_modules="all-linear",
        use_gradient_checkpointing="unsloth" if use_gradient_checkpointing else True,
    )
    model.print_trainable_parameters()
    collator = Qwen2VLCollator(processor, args.max_length, args.prompt)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)

    total_batches = len(dataloader)
    total_steps = args.max_steps if args.max_steps > 0 else max(1, int(total_batches * args.epochs))
    step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    use_autocast = torch.cuda.is_available() and (use_fp16 or use_bf16)
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16

    print("Starting training...")
    for epoch_idx in range(max(1, int(args.epochs if args.epochs >= 1 else 1))):
        print(f"Epoch {epoch_idx + 1}")
        for batch_idx, batch in enumerate(dataloader, start=1):
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            if batch_idx % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                step += 1
                print(f"step={step} loss={loss.item() * args.gradient_accumulation_steps:.4f}")
                if args.max_steps > 0 and step >= args.max_steps:
                    break
        save_artifacts(model, processor, output_dir)
        if args.max_steps > 0 and step >= args.max_steps:
            break

    print(f"Saving fine-tuned adapter and processor to {output_dir}")
    save_artifacts(model, processor, output_dir)


if __name__ == "__main__":
    main()
