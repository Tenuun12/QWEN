from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

files = [
    Path('qwen_dataset/qwen2_train_data_half.jsonl'),
    Path('qwen_dataset/qwen2_train_data_half_absolute.jsonl'),
]

for p in files:
    if not p.exists():
        print(f'MISSING {p}')
        continue
    abs_count = 0
    rel_count = 0
    missing_count = 0
    bad_count = 0
    total = 0
    with p.open('r', encoding='utf-8') as f:
        for total, line in enumerate(f, start=1):
            row = json.loads(line)
            img = row.get('image', '')
            path = Path(img)
            if path.is_absolute():
                abs_count += 1
            else:
                rel_count += 1
            if not path.exists():
                missing_count += 1
            if not (img.startswith('C:') or img.startswith('/') or img.startswith('\\')):
                bad_count += 1
    print(p)
    print('  total=', total)
    print('  abs=', abs_count)
    print('  rel=', rel_count)
    print('  missing=', missing_count)
    print('  bad_format=', bad_count)
    with p.open('r', encoding='utf-8') as f:
        first_row = json.loads(f.readline())
        print('  first image=', first_row.get('image'))

    # Verify loader resolution for relative paths.
    if p.exists():
        from train_local_model import JsonlVisionDataset
        ds = JsonlVisionDataset(p, max_samples=1)
        print('  loader image=', ds[0]['image'], 'exists=', Path(ds[0]['image']).exists())
