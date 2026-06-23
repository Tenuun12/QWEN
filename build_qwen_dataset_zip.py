import zipfile
from pathlib import Path

root = Path("qwen_dataset")
zip_path = root / "qwen_dataset.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in root.rglob("*"):
        if path.is_file():
            zf.write(path, path.relative_to(root))
print(f"Created {zip_path} ({zip_path.stat().st_size} bytes)")
