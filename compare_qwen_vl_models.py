import argparse
import gc
import glob
import json
import os
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TRITON_CACHE_DIR", str(Path(__file__).resolve().parent / ".cache" / "triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(__file__).resolve().parent / ".cache" / "torchinductor"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from peft import PeftModel
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)


DEFAULT_BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_PROMPT = (
    "Convert the following document to markdown. Return only the markdown with no explanation text. "
    "Do not include delimiters like ```markdown or ```html. RULES: - You must include all information on the page. "
    "Do not exclude headers, footers, or subtext. - Return tables in an HTML format. "
    "Charts & infographics must be interpreted to a markdown format. Prefer table format when applicable. "
    "Prefer using ☐ and ☑ for check boxes."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare base Qwen2-VL with a local fine-tuned LoRA adapter.")
    parser.add_argument("--image-path", default="C:\\QWEN-2B_train\\qwen_dataset\\pdfs_for_compare\\image.png")
    parser.add_argument(
        "--pdf-path",
        "--input-path",
        dest="input_path",
        default=None,
        help="Optional PDF, image, or directory of images to render and compare instead of --image-path.",
    )
    parser.add_argument(
        "--image-paths",
        default=None,
        help="Optional comma-separated list of image files or a directory containing images.",
    )
    parser.add_argument("--pages", default="all", help="PDF pages to use for PDF input, e.g. all, 1,2, or 1-3.")
    parser.add_argument("--render-scale", type=float, default=2.0, help="PDF render scale for vision input.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to run the model on.")
    parser.add_argument("--reference-path", default=None, help="Optional text or PDF file containing reference output for scoring.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-model", default="local_finetuned_model2")
    parser.add_argument("--dataset-file", default="qwen_dataset/qwen2_train_data.jsonl")
    parser.add_argument("--output-dir", default="comparison_outputs")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    return parser.parse_args()


def resolve_model_source(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if path.exists():
        return str(path.resolve())
    if "\\" in model_path or ":" in model_path or model_path.startswith((".", "..")):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if "/" in model_path:
        org, name = model_path.split("/", 1)
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        snapshot_root = hf_home / "hub" / f"models--{org}--{name}" / "snapshots"
        if snapshot_root.exists():
            snapshots = sorted(
                [p for p in snapshot_root.iterdir() if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if snapshots:
                return str(snapshots[0].resolve())
    return model_path


def normalize_path_tokens(path: Path) -> list[str]:
    text = re.sub(r"[^\w]+", " ", path.name.lower())
    return [token for token in text.split() if token]


def has_text_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if re.search(r"[^0-9]", token)]


def find_reference(dataset_file: Path, image_paths: list[Path]) -> str | None:
    if not dataset_file.exists():
        return None

    targets = {path.resolve() for path in image_paths}
    target_basenames = [normalize_path_tokens(path) for path in image_paths]
    target_text_tokens = [has_text_tokens(tokens) for tokens in target_basenames]

    fallback_candidates = []
    with dataset_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            row_image = Path(row["image"])
            if not row_image.is_absolute():
                row_image = (dataset_file.parent / row_image).resolve()
            if row_image in targets:
                return row.get("output")

            row_tokens = has_text_tokens(normalize_path_tokens(row_image))
            for target_tokens in target_text_tokens:
                if target_tokens and all(tok in row_tokens for tok in target_tokens):
                    fallback_candidates.append((row_image, row.get("output")))
                    break

    if len(fallback_candidates) == 1:
        print(f"Warning: no exact match found for image path; using reference for {fallback_candidates[0][0]}")
        return fallback_candidates[0][1]
    if len(fallback_candidates) > 1:
        print("Warning: multiple possible reference matches found for the input image; no reference selected.")
    return None


def parse_pages(page_spec: str, page_count: int) -> list[int]:
    if page_spec.lower() == "all":
        return list(range(page_count))

    pages = []
    for part in page_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            pages.extend(range(int(start) - 1, int(end)))
        else:
            pages.append(int(part) - 1)

    bad_pages = [page + 1 for page in pages if page < 0 or page >= page_count]
    if bad_pages:
        raise ValueError(f"Page(s) out of range for {page_count}-page PDF: {bad_pages}")
    return pages


def render_pdf_pages(pdf_path: Path, page_spec: str, render_scale: float) -> list[Image.Image]:
    import fitz

    doc = fitz.open(pdf_path)
    matrix = fitz.Matrix(render_scale, render_scale)
    images = []
    for page_index in parse_pages(page_spec, len(doc)):
        pixmap = doc[page_index].get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        images.append(image)
    return images


def load_image_list(path_spec: str) -> tuple[list[Path], str]:
    path_spec = path_spec.strip()
    if not path_spec:
        return [], ""

    if "," in path_spec:
        image_paths = [Path(p.strip()).expanduser().resolve() for p in path_spec.split(",") if p.strip()]
        return image_paths, ", ".join(str(p) for p in image_paths)

    path = Path(path_spec).expanduser().resolve()
    if path.is_dir():
        image_paths = sorted(
            [p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}]
        )
        return image_paths, str(path)
    return [path], str(path)


def extract_text_from_pdf(path: Path) -> str:
    import fitz

    doc = fitz.open(path)
    text = []
    for page in doc:
        page_text = page.get_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text).strip()


REFERENCE_CANDIDATE_NAMES = (
    "reference",
    "actual_text",
    "ground_truth",
    "expected",
    "target",
    "answer",
    "label",
)
REFERENCE_CANDIDATE_SUFFIXES = (".txt", ".pdf")


def find_reference_path_near_input(input_path: Path) -> Path | None:
    if input_path.is_dir():
        search_dirs = [input_path, input_path.parent]
    else:
        search_dirs = [input_path.parent]

    for search_dir in search_dirs:
        if not search_dir.exists() or not search_dir.is_dir():
            continue
        for name in REFERENCE_CANDIDATE_NAMES:
            for suffix in REFERENCE_CANDIDATE_SUFFIXES:
                candidate = search_dir / f"{name}{suffix}"
                if candidate.exists():
                    return candidate
        for suffix in REFERENCE_CANDIDATE_SUFFIXES:
            candidates = sorted(search_dir.glob(f"*{suffix}"))
            if len(candidates) == 1:
                return candidates[0]
    return None


def read_reference_file(path: str | None) -> str | None:
    if not path:
        return None
    reference_path = Path(path).expanduser().resolve()
    if reference_path.is_dir():
        reference_path = find_reference_path_near_input(reference_path)
        if reference_path is None:
            return None
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")
    suffix = reference_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(reference_path)
    if suffix == ".docx":
        from docx import Document

        doc = Document(reference_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs).strip()
    return reference_path.read_text(encoding="utf-8").strip()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    cleaned = re.sub(r"[^\w\s]", " ", normalized)
    return [token for token in cleaned.split() if token]


def compute_precision_recall(reference: str | None, prediction: str) -> tuple[float | None, float | None]:
    if reference is None:
        return None, None
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)
    if not ref_tokens and not pred_tokens:
        return 1.0, 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0, 0.0
    ref_counts = Counter(ref_tokens)
    pred_counts = Counter(pred_tokens)
    common = sum((ref_counts & pred_counts).values())
    precision = 0.0 if not pred_tokens else common / len(pred_tokens)
    recall = 0.0 if not ref_tokens else common / len(ref_tokens)
    return precision, recall


def f1_score(reference: str | None, prediction: str) -> float | None:
    if reference is None:
        return None
    precision, recall = compute_precision_recall(reference, prediction)
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def word_error_rate(reference: str | None, prediction: str) -> float | None:
    if reference is None:
        return None
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)
    if not ref_tokens:
        return 0.0 if not pred_tokens else 1.0

    # Levenshtein distance on tokenized words
    n = len(ref_tokens)
    m = len(pred_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
    return dp[n][m] / n


def load_input_images(args, image_path: Path) -> tuple[list[Image.Image], str, list[Path]]:
    if args.image_paths:
        image_paths, input_name = load_image_list(args.image_paths)
        images = [Image.open(p).convert("RGB") for p in image_paths]
        return images, input_name, image_paths

    if args.input_path:
        input_path = Path(args.input_path).expanduser().resolve()
        if input_path.suffix.lower() == ".pdf":
            images = render_pdf_pages(input_path, args.pages, args.render_scale)
            return images, f"{input_path} pages={args.pages}", [input_path]
        if input_path.is_dir():
            image_paths = sorted(
                [p for p in input_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}]
            )
            images = [Image.open(p).convert("RGB") for p in image_paths]
            return images, str(input_path), image_paths
        return [Image.open(input_path).convert("RGB")], str(input_path), [input_path]

    return [Image.open(image_path).convert("RGB")], str(image_path), [image_path]


def similarity(reference: str | None, prediction: str) -> float | None:
    if reference is None:
        return None
    return SequenceMatcher(None, reference, prediction).ratio()


def load_processor(model_name: str, min_pixels: int, max_pixels: int, fallback_model_name: str | None = None):
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        if fallback_model_name is not None and fallback_model_name != model_name:
            return load_processor(fallback_model_name, min_pixels, max_pixels, fallback_model_name=None)
        raise
    if getattr(processor, "image_processor", None) is not None:
        processor.image_processor.size = {
            "shortest_edge": min_pixels,
            "longest_edge": max_pixels,
        }
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def maybe_resolve_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    return str(path.resolve()) if path.exists() else model_path


def resolve_path_or_pattern(path_spec: str) -> Path:
    path = Path(path_spec)
    if path.exists():
        return path.resolve()
    if any(ch in path_spec for ch in "*?["):
        matches = sorted(glob.glob(path_spec))
        if len(matches) == 1:
            return Path(matches[0]).resolve()
        if len(matches) > 1:
            print(f"Warning: path pattern {path_spec!r} matched multiple files; using {matches[0]}")
            return Path(matches[0]).resolve()
    return path


def get_adapter_base_model_path(adapter_model: str) -> str | None:
    info = inspect_adapter_base_model(adapter_model)
    if not info:
        return None
    base_model = info.get("base_model_name_or_path")
    return str(Path(base_model).expanduser().resolve()) if base_model and Path(base_model).exists() else base_model


def inspect_adapter_base_model(adapter_model: str) -> dict[str, str] | None:
    path = Path(adapter_model)
    if not path.exists() or not path.is_dir():
        return None
    config_path = path / "adapter_config.json"
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    auto_mapping = data.get("auto_mapping") or {}
    return {
        "base_model_name_or_path": data.get("base_model_name_or_path"),
        "base_model_class": auto_mapping.get("base_model_class") or data.get("base_model_class"),
    }


def get_qwen_model_class(model_name_or_path: str | None = None, base_model_class: str | None = None):
    if base_model_class:
        lowered = base_model_class.lower()
        if "qwen3" in lowered:
            return Qwen3VLForConditionalGeneration
        if "qwen2" in lowered:
            return Qwen2VLForConditionalGeneration
    if model_name_or_path is None:
        return AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if config.model_type == "qwen2_vl":
        return Qwen2VLForConditionalGeneration
    if config.model_type == "qwen3_vl":
        return Qwen3VLForConditionalGeneration
    return AutoModelForCausalLM


def load_qwen_model(model_name: str, offload_dir: Path, device: str, model_cls=None):
    use_cuda = (device == "cuda") or (device == "auto" and torch.cuda.is_available())
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16 if use_cuda else torch.float32
    offload_dir.mkdir(parents=True, exist_ok=True)
    device_map = "auto" if use_cuda else {"": "cpu"}
    model_cls = model_cls or get_qwen_model_class(model_name)

    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "offload_folder": str(offload_dir),
        "offload_buffers": True,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    needs_bnb_kwargs = False
    if isinstance(model_name, str):
        lower_name = model_name.lower()
        if "bnb" in lower_name or "4bit" in lower_name or "bitsandbytes" in lower_name:
            needs_bnb_kwargs = True
    if use_cuda and needs_bnb_kwargs:
        load_kwargs["llm_int8_enable_fp32_cpu_offload"] = True
        load_kwargs["bnb_4bit_compute_dtype"] = dtype
        load_kwargs["bnb_4bit_use_double_quant"] = True

    try:
        return model_cls.from_pretrained(model_name, **load_kwargs).eval()
    except TypeError as exc:
        message = str(exc)
        if "unexpected keyword argument" in message:
            unsupported = []
            for kw in ["llm_int8_enable_fp32_cpu_offload", "bnb_4bit_compute_dtype", "bnb_4bit_use_double_quant"]:
                if kw in message and kw in load_kwargs:
                    unsupported.append(kw)
            if unsupported:
                for kw in unsupported:
                    load_kwargs.pop(kw, None)
                print(f"Retrying model load without unsupported kwarg(s): {unsupported}")
                return model_cls.from_pretrained(model_name, **load_kwargs).eval()
        raise
    except ValueError as exc:
        message = str(exc)
        if use_cuda and "llm_int8_enable_fp32_cpu_offload" in message:
            print("Retrying model load with low_cpu_mem_usage disabled.")
            load_kwargs["low_cpu_mem_usage"] = False
            return model_cls.from_pretrained(model_name, **load_kwargs).eval()
        raise


def load_finetuned_model(base_model_name: str, adapter_model: str, offload_dir: Path, device: str):
    adapter_info = inspect_adapter_base_model(adapter_model)
    if adapter_info and adapter_info.get("base_model_class"):
        lowered = adapter_info["base_model_class"].lower()
        if "qwen2" in lowered and "qwen3" in base_model_name.lower():
            raise ValueError(
                f"Adapter {adapter_model} is trained for Qwen2-VL while base model {base_model_name} is Qwen3-VL."
                " Use a matching base model or adapter."
            )
        if "qwen3" in lowered and "qwen2" in base_model_name.lower():
            raise ValueError(
                f"Adapter {adapter_model} is trained for Qwen3-VL while base model {base_model_name} is Qwen2-VL."
                " Use a matching base model or adapter."
            )
    model = load_qwen_model(base_model_name, offload_dir, device)
    use_cuda = (device == "cuda") or (device == "auto" and torch.cuda.is_available())
    device_map = "auto" if use_cuda else {"": "cpu"}
    model = PeftModel.from_pretrained(
        model,
        adapter_model,
        offload_folder=str(offload_dir),
        device_map=device_map,
        offload_buffers=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    return model


def first_device(model):
    return next(model.parameters()).device


def run_inference(model, processor, images: list[Image.Image], prompt: str, max_new_tokens: int, temperature: float):
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    conversation = [
        {
            "role": "user",
            "content": content,
        }
    ]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(first_device(model))

    generation_kwargs = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        generation_kwargs.update({"do_sample": False})

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    generated_ids = [
        output_ids[i, inputs.input_ids[i].shape[-1] :]
        for i in range(output_ids.shape[0])
    ]
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()


def unload(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def main():
    args = parse_args()
    image_path = resolve_path_or_pattern(args.image_path)
    dataset_file = Path(args.dataset_file)
    output_dir = Path(args.output_dir)
    base_model = resolve_model_source(args.base_model)
    adapter_model = maybe_resolve_path(args.adapter_model)
    input_images, input_name, input_paths = load_input_images(args, image_path)
    reference = read_reference_file(args.reference_path)
    if reference is None and not (args.input_path and Path(args.input_path).suffix.lower() == ".pdf"):
        reference = find_reference(dataset_file, input_paths if input_paths else [image_path])
    if reference is None and args.input_path:
        auto_ref_path = find_reference_path_near_input(Path(args.input_path).expanduser().resolve())
        if auto_ref_path is not None:
            print(f"Auto-found reference file: {auto_ref_path}")
            reference = read_reference_file(str(auto_ref_path))

    print(f"Input: {input_name}")
    print(f"Base model: {base_model}")
    print(f"Fine-tuned adapter: {adapter_model}")
    print(f"Device: {args.device}")
    print(f"Reference found: {reference is not None}")

    adapter_base_model = None
    if (info := inspect_adapter_base_model(adapter_model)) is not None:
        print("Adapter metadata:")
        print(f"  base_model_name_or_path: {info.get('base_model_name_or_path')}")
        print(f"  base_model_class: {info.get('base_model_class')}")
        adapter_base_model = get_adapter_base_model_path(adapter_model)
        if adapter_base_model and adapter_base_model != base_model:
            print(f"Warning: adapter was trained from a different base model: {adapter_base_model}")
            print("Using the adapter's original base model to load the fine-tuned model for compatibility.")

    finetuned_base_model = adapter_base_model or base_model

    print("\nLoading base model...")
    base_processor = load_processor(base_model, args.min_pixels, args.max_pixels)
    base = load_qwen_model(base_model, output_dir / "offload_base", args.device)
    base_output = run_inference(base, base_processor, input_images, args.prompt, args.max_new_tokens, args.temperature)
    unload(base)

    print("Loading fine-tuned model...")
    finetuned_processor = load_processor(adapter_model, args.min_pixels, args.max_pixels, fallback_model_name=finetuned_base_model)
    finetuned = load_finetuned_model(finetuned_base_model, adapter_model, output_dir / "offload_finetuned", args.device)
    finetuned_output = run_inference(
        finetuned,
        finetuned_processor,
        input_images,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
    )
    unload(finetuned)

    base_score = similarity(reference, base_output)
    finetuned_score = similarity(reference, finetuned_output)
    base_precision, base_recall = compute_precision_recall(reference, base_output)
    finetuned_precision, finetuned_recall = compute_precision_recall(reference, finetuned_output)
    base_f1 = f1_score(reference, base_output)
    finetuned_f1 = f1_score(reference, finetuned_output)
    base_wer = word_error_rate(reference, base_output)
    finetuned_wer = word_error_rate(reference, finetuned_output)

    write_text(output_dir / "base_output.md", base_output)
    write_text(output_dir / "finetuned_output.md", finetuned_output)
    if reference is not None:
        write_text(output_dir / "reference.md", reference)

    winner = None
    if base_f1 is not None and finetuned_f1 is not None:
        if finetuned_f1 > base_f1:
            winner = "finetuned"
        elif base_f1 > finetuned_f1:
            winner = "base"
    elif base_score is not None and finetuned_score is not None:
        if finetuned_score > base_score:
            winner = "finetuned"
        elif base_score > finetuned_score:
            winner = "base"

    summary = {
        "input": input_name,
        "base_model": args.base_model,
        "finetuned_base_model": finetuned_base_model,
        "adapter_model": args.adapter_model,
        "max_new_tokens": args.max_new_tokens,
        "base_chars": len(base_output),
        "finetuned_chars": len(finetuned_output),
        "reference_chars": len(reference) if reference is not None else None,
        "base_similarity": base_score,
        "finetuned_similarity": finetuned_score,
        "base_precision": base_precision,
        "base_recall": base_recall,
        "finetuned_precision": finetuned_precision,
        "finetuned_recall": finetuned_recall,
        "base_f1": base_f1,
        "finetuned_f1": finetuned_f1,
        "base_wer": base_wer,
        "finetuned_wer": finetuned_wer,
        "winner": winner,

    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== Base output ===")
    print(base_output)
    print("\n=== Fine-tuned output ===")
    print(finetuned_output)
    print("\n=== Metrics ===")
    print(f"base      precision={base_precision}, recall={base_recall}, f1={base_f1}, wer={base_wer}")
    print(f"finetuned precision={finetuned_precision}, recall={finetuned_recall}, f1={finetuned_f1}, wer={finetuned_wer}")
    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
