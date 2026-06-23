#!/usr/bin/env python3
"""
Script to analyze training data and create 50% dataset split for local_finetuned_model_half
"""

import json
import argparse
from pathlib import Path
from typing import Tuple
import random

def count_jsonl_samples(jsonl_path: Path) -> int:
    """Count the number of samples in a JSONL file."""
    if not jsonl_path.exists():
        return 0
    count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                count += 1
    return count

def normalize_image_path(sample: dict, base_dir: Path) -> dict:
    """Normalize a sample image path if it is relative.

    This handles both dataset-relative paths like "images/..." and root-prefixed values like
    "qwen_dataset/images/..." when the dataset file lives in the qwen_dataset folder.
    """
    image_value = sample.get("image")
    if isinstance(image_value, str):
        normalized = image_value.replace("\\", "/")
        prefix = f"{base_dir.name}/"
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
        sample["image"] = normalized
    return sample


def resolve_image_path(sample: dict, base_dir: Path) -> dict:
    """Resolve a sample image path to an absolute path if it is relative."""
    image_value = sample.get("image")
    if isinstance(image_value, str):
        image_value = image_value.replace("\\", "/")
        prefix = f"{base_dir.name}/"
        if image_value.startswith(prefix):
            image_value = image_value[len(prefix):]
        image_path = Path(image_value)
        if not image_path.is_absolute():
            sample["image"] = str((base_dir / image_path).resolve())
    return sample


def create_split_dataset(input_jsonl: Path, output_jsonl: Path, split_ratio: float = 0.5, seed: int = 42,
                         resolve_image_paths: bool = False, image_base_dir: Path | None = None):
    """
    Create a split dataset from the input JSONL file.
    
    Args:
        input_jsonl: Path to input JSONL file
        output_jsonl: Path to output JSONL file
        split_ratio: Ratio of data to include (0.5 = 50%)
        seed: Random seed for reproducibility
        resolve_image_paths: Whether to resolve relative image paths to absolute paths
        image_base_dir: Base directory used to resolve relative image paths
    """
    random.seed(seed)
    
    # Load all samples
    samples = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    
    total_samples = len(samples)
    split_size = int(total_samples * split_ratio)
    
    # Randomly select samples
    selected_indices = sorted(random.sample(range(total_samples), split_size))
    selected_samples = [samples[i] for i in selected_indices]

    image_base_dir = image_base_dir or input_jsonl.parent
    if resolve_image_paths:
        for sample in selected_samples:
            resolve_image_path(sample, image_base_dir)
    else:
        for sample in selected_samples:
            normalize_image_path(sample, image_base_dir)
    
    # Write output
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for sample in selected_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    return total_samples, split_size, selected_samples

def main():
    parser = argparse.ArgumentParser(description='Analyze and split training datasets')
    parser.add_argument('--input-dataset', default='qwen_dataset/qwen2_train_data.jsonl',
                        help='Path to input JSONL dataset')
    parser.add_argument('--output-dataset', default='qwen_dataset/qwen2_train_data_half.jsonl',
                        help='Path to output split JSONL dataset')
    parser.add_argument('--split-ratio', type=float, default=0.5,
                        help='Ratio of data to include (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze dataset size without creating split')
    parser.add_argument('--absolute-image-paths', action='store_true',
                        help='Write absolute image paths into the output dataset')
    
    args = parser.parse_args()
    
    # Resolve to absolute paths anchored to this script's directory
    repo_root = Path(__file__).resolve().parent
    input_path = Path(args.input_dataset)
    if not input_path.is_absolute():
        input_path = (repo_root / input_path).resolve()
    else:
        input_path = input_path.resolve()
    output_path = Path(args.output_dataset)
    if not output_path.is_absolute():
        output_path = (repo_root / output_path).resolve()
    else:
        output_path = output_path.resolve()
    
    # Check if dataset exists
    if not input_path.exists():
        print(f"Error: Dataset not found at {input_path}")
        print("\nTo create a dataset, run:")
        print("  python prepare_qwen_dataset.py")
        return 1
    
    # Count samples
    total = count_jsonl_samples(input_path)
    print(f"Dataset: {input_path}")
    print(f"Total samples: {total}")
    
    if args.analyze_only:
        print(f"Split ratio: {args.split_ratio}")
        print(f"Would create: {int(total * args.split_ratio)} samples")
        return 0
    
    # Create split
    print(f"\nCreating {args.split_ratio*100:.0f}% split dataset...")
    total_samples, split_size, selected_samples = create_split_dataset(
        input_path,
        output_path,
        args.split_ratio,
        args.seed,
        resolve_image_paths=args.absolute_image_paths,
        image_base_dir=input_path.parent,
    )
    
    print(f"Original: {total_samples} samples")
    print(f"Split: {split_size} samples ({split_size/total_samples*100:.1f}%)")
    print(f"Output saved to: {output_path}")
    
    return 0

if __name__ == '__main__':
    exit(main())
