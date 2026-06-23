from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
jsonl = root / 'qwen_dataset' / 'qwen2_train_data_half_relative.jsonl'
backup = jsonl.with_suffix('.jsonl.bak')
fixed = root / 'qwen_dataset' / 'qwen2_train_data_half_relative_fixed.jsonl'

if not jsonl.exists():
    print('ERROR: input file not found:', jsonl)
    raise SystemExit(1)

shutil.copy2(jsonl, backup)

replacements = 0
lines = []
with jsonl.open('r', encoding='utf-8') as f:
    for line in f:
        if './images/' in line:
            line = line.replace('./images/', 'qwen_dataset/images/')
            replacements += 1
        lines.append(line)

with fixed.open('w', encoding='utf-8') as f:
    f.writelines(lines)

print(f'Wrote {fixed}\nReplacements made (lines changed): {replacements}\nBackup at: {backup}')
