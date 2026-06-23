from transformers import Qwen3VLForConditionalGeneration
import torch
import os

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="tmp",
    offload_state_dict=True,
)
keys = [k for k in model.state_dict().keys() if "embed_tokens" in k or "language_model" in k or "model.model" in k]
print("TOTAL", len(keys))
for k in keys[:200]:
    print(k)
