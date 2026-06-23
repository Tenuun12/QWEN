from pathlib import Path
import json

p = Path('qwen_dataset/qwen2_train_data_half_absolute.jsonl')
missing = 0
broken = []
for i, line in enumerate(p.open('r', encoding='utf-8')):
    row = json.loads(line)
    image = row.get('image')
    if not image:
        missing += 1
        broken.append((i, image, 'empty'))
        continue
    path = Path(image)
    if not path.is_absolute():
        path = (p.parent / path).resolve()
    if not path.exists():
        missing += 1
        broken.append((i, image, str(path)))
        if missing > 20:
            break
print('total', i + 1, 'missing', missing)
for item in broken:
    print(item)
