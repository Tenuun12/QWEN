import json
from pathlib import Path

INPUT_FILE = Path('qwen_dataset/qwen2_train_data_half_absolute.jsonl')
OUTPUT_FILE = Path('qwen_dataset/qwen2_train_data_half_fixed.jsonl')

if not INPUT_FILE.exists():
    raise FileNotFoundError(f'Input file not found: {INPUT_FILE}')

with INPUT_FILE.open('r', encoding='utf-8') as fin, OUTPUT_FILE.open('w', encoding='utf-8') as fout:
    total = 0
    for line in fin:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        image_value = item.get('image', '')
        if image_value:
            image_name = Path(image_value).name
            item['image'] = f'images/{image_name}'
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        total += 1

print(f'Wrote {total} rows to {OUTPUT_FILE}')

missing = 0
with OUTPUT_FILE.open('r', encoding='utf-8') as fin:
    for line in fin:
        item = json.loads(line)
        if not (Path('qwen_dataset') / item['image']).exists():
            print('Missing:', item['image'])
            missing += 1
print(f'Total missing: {missing}')
