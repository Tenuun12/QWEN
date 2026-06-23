from safetensors.torch import load_file
from pathlib import Path
import torch
p=Path('local_finetuned_model2/adapter_model.safetensors')
print('Loading',p)
store = load_file(str(p))
keys = list(store.keys())
print('num_keys=', len(keys))
print('\nfirst 20 keys:')
for k in keys[:20]:
    print(k)

print('\nChecking norms for some keys:')
sample_keys = [
    'base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight',
    'base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight',
    'base_model.model.model.language_model.layers.16.self_attn.q_proj.lora_A.weight',
]
for k in sample_keys:
    if k in store:
        t = store[k]
        print(k, 'shape', t.shape, 'norm', float(torch.norm(t).item()), 'abs_sum', float(t.abs().sum().item()))
    else:
        print(k, 'MISSING')

# compute global nonzero count
import numpy as np
nonzero = 0
total = 0
for k in keys:
    arr = store[k]
    nonzero += (arr != 0).sum().item()
    total += arr.size
print('total nonzero elements:', nonzero, 'of', total)
