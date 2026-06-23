from pathlib import Path
import traceback
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

base_path = Path(r"C:\Users\CoreTech\.cache\huggingface\hub\models--Qwen--Qwen2-VL-2B-Instruct\snapshots\895c3a49bc3fa70a340399125c650a463535e71c")
adapter_path = Path('local_finetuned_model2')
image_path = Path('qwen_dataset/pdfs_for_compare/image.png')

print('Loading processor...')
proc = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
print('Processor loaded')

print('Loading base model...')
base = Qwen2VLForConditionalGeneration.from_pretrained(base_path, trust_remote_code=True).eval()
print('Base model loaded')

print('Attaching adapter...')
try:
    model = PeftModel.from_pretrained(base, adapter_path, device_map='auto')
    print('Adapter attached')
except Exception:
    print('PEFT attach FAILED')
    traceback.print_exc()
    raise SystemExit(1)

print('Preparing inputs...')
image = Image.open(image_path).convert('RGB')
content = [{'type': 'image', 'image': image}, {'type':'text', 'text': 'Convert the following document to markdown.'}]
conversation = [{'role':'user','content':content}]
inputs = proc.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt')
inputs = {k: v.to(next(model.parameters()).device) for k,v in inputs.items()}
print('Input ids shape', inputs['input_ids'].shape)
print('Generating...')
try:
    # Turn off caching and limit tokens to make generation faster for diagnostics.
    model.config.use_cache = False
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    print('Generated shape', out.shape)
    gen_ids = [out[i, inputs['input_ids'].shape[-1]:] for i in range(out.shape[0])]
    decoded = proc.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print('Decoded output:\n', decoded[0])
except Exception:
    print('Generation FAILED')
    traceback.print_exc()
    raise

print('Done')
