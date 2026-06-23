import json
from pathlib import Path
import shutil

INPUT = Path("qwen_dataset") / "qwen2_train_data_half_relative.jsonl"
OUTPUT = Path("qwen_dataset") / "qwen2_train_data_half_vast.jsonl"
BACKUP = INPUT.with_suffix(INPUT.suffix + ".bak")

def fix_path(img_value: str) -> str:
    # Normalize known prefixes to the vast.ai mount path
    prefixes = ["./images/", "images/", "qwen_dataset/images/"]
    for p in prefixes:
        if img_value.startswith(p):
            return "/data/qwen_dataset/images/" + img_value[len(p):]
    # If already absolute or other, leave unchanged
    return img_value


def main():
    if not INPUT.exists():
        print(f"Input file not found: {INPUT}")
        return

    # Backup
    if not BACKUP.exists():
        shutil.copy2(INPUT, BACKUP)
        print(f"Backup created: {BACKUP}")

    total = 0
    changed = 0
    with INPUT.open("r", encoding="utf-8") as inf, OUTPUT.open("w", encoding="utf-8") as outf:
        for line in inf:
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                # If line isn't valid JSON, copy through
                outf.write(line)
                continue

            img = obj.get("image")
            if isinstance(img, str):
                new = fix_path(img)
                if new != img:
                    obj["image"] = new
                    changed += 1

            outf.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Processed {total} lines, updated {changed} image paths.")
    print(f"Wrote fixed file: {OUTPUT}")


if __name__ == "__main__":
    main()
