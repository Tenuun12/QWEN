import argparse
import json
from pathlib import Path


def normalize_image_value(image_value: str, dataset_dir: Path) -> str:
    if not image_value:
        return ""

    normalized = image_value.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    prefix = f"{dataset_dir.name}/"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]

    image_path = Path(normalized)
    if image_path.is_absolute():
        return str(image_path)

    absolute_path = (dataset_dir / image_path).resolve()
    return str(absolute_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert JSONL dataset image paths to absolute filesystem paths."
    )
    parser.add_argument(
        "--input",
        default="qwen_dataset/qwen2_train_data_half.jsonl",
        help="Input JSONL dataset file with relative or root-relative image paths.",
    )
    parser.add_argument(
        "--output",
        default="qwen_dataset/qwen2_train_data_half_absolute.jsonl",
        help="Output JSONL dataset file with absolute image paths.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset file not found: {input_path}")

    dataset_dir = input_path.parent.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    missing = 0
    missing_examples = []

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            image_value = item.get("image", "")
            absolute_image = normalize_image_value(image_value, dataset_dir)
            item["image"] = absolute_image
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            total += 1
            if absolute_image and not Path(absolute_image).exists():
                missing += 1
                if len(missing_examples) < 20:
                    missing_examples.append((total - 1, absolute_image))

    print(f"Wrote {total} rows to {output_path}")
    print(f"Absolute image paths: {total - missing}")
    print(f"Missing files: {missing}")
    if missing_examples:
        print("First missing examples:")
        for idx, path in missing_examples:
            print(f"  row {idx}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
