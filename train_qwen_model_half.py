#!/usr/bin/env python3
"""
Fine-tune Qwen2-VL on 50% of the training data for local_finetuned_model_half
This script is based on train_local_model.py with the --max-samples parameter
to limit training to approximately 50% of available data.
"""

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

# Import from train_local_model to avoid code duplication
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_local_model import (
    DEFAULT_MODEL, IGNORE_INDEX, resolve_model_source, build_messages, 
    load_rgb, JsonlVisionDataset, pick_precision, import_unsloth, 
    Qwen2VLCollator, save_artifacts
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-VL on 50% of training data.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL, help="Local model directory or Hugging Face model id.")
    parser.add_argument("--dataset-file", default="qwen_dataset/qwen2_train_data.jsonl")
    parser.add_argument("--output-dir", default="local_finetuned_model_half")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-pixels", type=int, default=3136, 
                        help="Minimum image pixel budget for Qwen2-VL preprocessing.")
    parser.add_argument("--max-pixels", type=int, default=768 * 28 * 28,
                        help="Maximum image pixel budget for Qwen2-VL preprocessing.")
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
    parser.add_argument("--max-steps", type=int, default=-1, help="Override training length.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_file = Path(args.dataset_file)
    
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    
    # Count total samples to calculate 50%
    total_samples = 0
    with dataset_file.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                total_samples += 1
    
    # Set max_samples to approximately 50% of the dataset
    half_samples = (total_samples + 1) // 2
    print(f"Total dataset samples: {total_samples}")
    print(f"Using 50% = {half_samples} samples for training")
    
    # Create modified args with max_samples set to 50%
    model_source = resolve_model_source(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset from {dataset_file}")
    # Load only 50% of samples
    dataset = JsonlVisionDataset(dataset_file, max_samples=half_samples)
    print(f"Dataset loaded with {len(dataset)} examples (50% split)")
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
    
    print("Starting training with 50% of data...")
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
