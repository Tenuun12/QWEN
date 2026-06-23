from pathlib import Path
import json

files = [
    Path('train_fixed_root.jsonl'),
    Path('qwen_dataset/qwen2_train_data_half_absolute.jsonl'),
    Path('qwen_dataset/qwen2_train_data_half_fixed.jsonl'),
]
for p in files:
    print('---', p)
    if not p.exists():
        print('MISSING')
        continue
    with p.open('r', encoding='utf-8') as f:
        for i in range(2):
            line = f.readline()
            if not line:
                break
            print(line.strip())
    with p.open('r', encoding='utf-8') as f:
        print('count', sum(1 for _ in f))
