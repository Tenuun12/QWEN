import json
from pathlib import Path

INPUT_FILE = Path("qwen_dataset/qwen2_train_data_half_absolute.jsonl")
CANONICAL_OUTPUT = Path("qwen_dataset/qwen2_train_data_half_fixed.jsonl")
ROOT_RELATIVE_OUTPUT = Path("qwen_dataset/qwen2_train_data_half_root_relative.jsonl")
DATASET_DIR = INPUT_FILE.parent.resolve()
IMAGES_DIR = DATASET_DIR / "images"


def normalize_image_path(image_value: str) -> str:
    if not image_value:
        return ""

    normalized = image_value.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith(f"{DATASET_DIR.name}/"):
        normalized = normalized[len(f"{DATASET_DIR.name}/") :]
    if normalized.startswith("images/"):
        return normalized

    path = Path(normalized)
    if path.is_absolute():
        return f"images/{path.name}"

    fallback = IMAGES_DIR / path.name
    if fallback.exists():
        return f"images/{path.name}"

    return normalized


def resolve_dataset_path(image_value: str) -> Path:
    normalized = image_value.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith(f"{DATASET_DIR.name}/"):
        normalized = normalized[len(f"{DATASET_DIR.name}/") :]
    image_path = Path(normalized)
    if image_path.is_absolute():
        return image_path
    return (DATASET_DIR / image_path).resolve()


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    CANONICAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with INPUT_FILE.open("r", encoding="utf-8") as fin, CANONICAL_OUTPUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["image"] = normalize_image_path(item.get("image", ""))
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            total += 1

    print(f"Wrote {total} rows to {CANONICAL_OUTPUT}")

    with CANONICAL_OUTPUT.open("r", encoding="utf-8") as fin, ROOT_RELATIVE_OUTPUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            img = item["image"].replace("\\", "/")
            if img.startswith("./"):
                img = img[2:]
            item["image"] = f"{DATASET_DIR.name}/{img}"
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Also wrote root-relative dataset to {ROOT_RELATIVE_OUTPUT}")

    missing = 0
    with CANONICAL_OUTPUT.open("r", encoding="utf-8") as fin:
        for line in fin:
            item = json.loads(line)
            if not resolve_dataset_path(item["image"]).exists():
                print("Missing in canonical version:", item["image"])
                missing += 1

    with ROOT_RELATIVE_OUTPUT.open("r", encoding="utf-8") as fin:
        for line in fin:
            item = json.loads(line)
            if not Path(item["image"]).exists():
                print("Missing in root-relative version:", item["image"])
                missing += 1

    print(f"Total missing across checks: {missing}")


if __name__ == "__main__":
    main()
