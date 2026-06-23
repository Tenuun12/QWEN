import json, random
lines = open('qwen_dataset/qwen2_train_data.jsonl', encoding='utf-8').read().splitlines()
print('total examples=', len(lines))
for i in random.sample(range(len(lines)), min(5, len(lines))):
    row = json.loads(lines[i])
    out = row.get('output') or ''
    print('index=', i, 'image=', row.get('image')[:120])
    print('output_len=', len(out))
    print(repr(out[:300]))
    print('---')
