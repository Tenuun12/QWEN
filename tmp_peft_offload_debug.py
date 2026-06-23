import inspect
from transformers import Qwen3VLForConditionalGeneration
from peft import PeftModel
import torch

base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen3-VL-4B-Instruct',
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='auto',
    offload_folder='tmp',
    offload_state_dict=False,
    low_cpu_mem_usage=True,
)
print('base model class', type(base_model))
print('has base_model attr', hasattr(base_model, 'base_model'))
print('named modules sample:')
for name, module in list(base_model.named_modules())[:40]:
    print(name)

orig = PeftModel._update_offload

def patched(self, offload_index, adapters_weights):
    print('PATCHED _update_offload called')
    print('offload_index keys sample:', list(offload_index.keys())[:10])
    print('named modules contains base_model.model?', 'base_model.model' in dict(self.named_modules()))
    print('named modules contains base_model.model.model?', 'base_model.model.model' in dict(self.named_modules()))
    print('named modules contains base_model.model.model.model?', 'base_model.model.model.model' in dict(self.named_modules()))
    return orig(self, offload_index, adapters_weights)

PeftModel._update_offload = patched

try:
    model = PeftModel.from_pretrained(
        base_model,
        'checkpoint-120',
        offload_folder='tmp',
        low_cpu_mem_usage=True,
        device_map='auto',
    )
    print('PEFT loaded', type(model))
    print('PEFT named_modules sample:')
    for name, module in list(model.named_modules())[:40]:
        print(name)
except Exception as e:
    import traceback
    traceback.print_exc()
