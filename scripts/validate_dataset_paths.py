import argparse
import json
from pathlib import Path


def resolve_image_path(image_value: str, dataset_dir: Path) -> Path:
    normalized = image_value.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    prefix = f"{dataset_dir.name}/"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix):]
    path = Path(normalized)
    if path.is_absolute():
        return path
    return (dataset_dir / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate JSONL image paths against the dataset directory.")
    parser.add_argument(
        "--dataset-file",
        default="qwen_dataset/qwen2_train_data_half_absolute.jsonl",
        help="Path to the JSONL dataset file to validate.",
    )
    args = parser.parse_args()

    dataset_file = Path(args.dataset_file)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    dataset_dir = dataset_file.parent.resolve()
    total = 0
    abs_count = 0
    rel_count = 0
    missing = 0
    bad_format = 0

    for i, line in enumerate(dataset_file.open("r", encoding="utf-8")):
        row = json.loads(line)
        img = row.get("image", "")
        path = Path(img)
        if path.is_absolute():
            abs_count += 1
        else:
            rel_count += 1
        resolved = resolve_image_path(img, dataset_dir)
        if not resolved.exists():
            missing += 1
            if missing <= 20:
                print(f"missing row {i}: {img} -> {resolved}")
        if not (
            img.startswith("C:")
            or img.startswith("/")
            or img.startswith("\\")
            or img.startswith("images/")
            or img.startswith("./images/")
            or img.startswith(f"{dataset_dir.name}/images/")
        ):
            bad_format += 1
        total += 1

    print(f"Dataset file: {dataset_file}")
    print(f"Total rows: {total}")
    print(f"Absolute paths: {abs_count}")
    print(f"Relative paths: {rel_count}")
    print(f"Missing files: {missing}")
    print(f"Non-standard path format: {bad_format}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
