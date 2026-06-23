import json
from pathlib import Path

path = Path('qwen_dataset/qwen2_train_data_half_absolute.jsonl')
backup = path.with_suffix(path.suffix + '.bak')
if backup.exists():
    raise SystemExit(f'Backup already exists: {backup}')
path.rename(backup)

prefixes = ['C:\\QWEN-2B_train\\qwen_dataset\\images\\', 'c:\\QWEN-2B_train\\qwen_dataset\\images\\']
new_lines = []
count = 0
fixed = 0
with backup.open('r', encoding='utf-8') as rf:
    for line in rf:
        data = json.loads(line)
        img = data.get('image', '')
        new_img = img
        for prefix in prefixes:
            if img.startswith(prefix):
                new_img = 'images/' + img[len(prefix):].replace('\\', '/')
                fixed += 1
                break
        if new_img != img:
            data['image'] = new_img
        new_lines.append(json.dumps(data, ensure_ascii=False))
        count += 1

with path.open('w', encoding='utf-8') as wf:
    wf.write('\n'.join(new_lines) + '\n')

print('processed', count)
print('fixed', fixed)
