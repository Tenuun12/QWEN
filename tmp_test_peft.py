from transformers import Qwen3VLForConditionalGeneration
from peft import PeftModel
import torch

base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen3-VL-4B-Instruct',
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='auto',
    low_cpu_mem_usage=True,
)
print('loaded base model')
model = PeftModel.from_pretrained(
    base_model,
    'checkpoint-120',
    low_cpu_mem_usage=True,
)
print('loaded PEFT adapter')
print(type(model))
print('model device map', getattr(model, 'hf_device_map', None))
