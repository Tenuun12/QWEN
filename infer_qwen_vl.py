import argparse
import importlib.util
from io import BytesIO
from urllib.request import urlopen

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen2-VL inference on an image or a video.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Local model directory or Hugging Face repo ID.",
    )
    parser.add_argument(
        "--image-url",
        default=None,
        help="URL of the image to describe.",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Local path to the image file.",
    )
    parser.add_argument(
        "--video-path",
        default=None,
        help="Optional path to a local video file for video inference.",
    )
    parser.add_argument(
        "--prompt",
        default="Convert the following document to markdown. Return only the markdown with no explanation text. Do not include delimiters like ```markdown or ```html. RULES: - You must include all information on the page. Do not exclude headers, footers, or subtext. - Return tables in an HTML format. - Charts & infographics must be interpreted to a markdown format. Prefer table format when applicable. - Prefer using ☐ and ☑ for check boxes.",
        help="Text prompt to accompany the image or video.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype to load the model in. bfloat16 is recommended, "
        "especially when some layers get offloaded to CPU (fp16 on CPU "
        "can produce NaN/Inf and crash generate()).",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the model with 4-bit (bitsandbytes) quantization. "
        "Strongly recommended for large models (e.g. 27B) to avoid CPU offload entirely.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum number of new tokens to generate.",
    )
    return parser.parse_args()


def load_image_from_url(url: str) -> Image.Image:
    with urlopen(url) as response:
        return Image.open(BytesIO(response.read())).convert("RGB")


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model_and_processor(model_name: str, dtype: torch.dtype, load_in_4bit: bool):
    print(f"Loading model: {model_name} (dtype={dtype}, 4bit={load_in_4bit})")

    accelerate_installed = importlib.util.find_spec("accelerate") is not None
    if not accelerate_installed:
        print("WARNING: accelerate is not installed. Loading model on a single device instead.")

    device_map = "auto" if accelerate_installed else None

    quantization_config = None
    if load_in_4bit:
        bnb_spec = importlib.util.find_spec("bitsandbytes")
        if bnb_spec is None:
            raise RuntimeError(
                "--load-in-4bit was requested but bitsandbytes is not installed. "
                "Install it with: pip install bitsandbytes"
            )
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    try:
        kwargs = dict(
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config

        model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print("Processor loaded successfully.")
    except Exception as e:
        print(f"Error loading processor: {e}")
        raise

    return model, processor


def run_inference(model, processor, conversation, max_new_tokens: int):
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
    )
    generated_ids = [
        output_ids[i, inputs.input_ids[i].shape[-1] :]
        for i in range(output_ids.shape[0])
    ]

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return output_text


def main():
    args = parse_args()
    dtype = resolve_dtype(args.dtype)

    print("Loading model and processor...")
    model, processor = load_model_and_processor(args.model, dtype, args.load_in_4bit)
    print("Model loaded.")

    if args.video_path:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": args.video_path},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
    else:
        if args.image_path:
            image = Image.open(args.image_path).convert("RGB")
        elif args.image_url:
            print("Loading image from URL...")
            image = load_image_from_url(args.image_url)
        else:
            raise ValueError("Must provide either --image-path or --image-url")
        print("Image loaded.")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]

    print("Running inference...")
    result = run_inference(model, processor, conversation, args.max_new_tokens)
    print("Output:")
    print(result[0] if isinstance(result, list) and len(result) == 1 else result)


if __name__ == "__main__":
    main()