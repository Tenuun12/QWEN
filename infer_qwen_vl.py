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
    return parser.parse_args()


def load_image_from_url(url: str) -> Image.Image:
    with urlopen(url) as response:
        return Image.open(BytesIO(response.read())).convert("RGB")


def load_model_and_processor(model_name: str):
    print(f"Loading model: {model_name}")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    accelerate_installed = importlib.util.find_spec("accelerate") is not None
    device_map = "auto" if accelerate_installed else None

    if not accelerate_installed:
        print("WARNING: accelerate is not installed. Loading model on a single device instead.")

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
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


def run_inference(model, processor, conversation):
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=4096,
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
    print("Loading model and processor...")
    model, processor = load_model_and_processor(args.model)
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
    result = run_inference(model, processor, conversation)
    print("Output:")
    print(result)

