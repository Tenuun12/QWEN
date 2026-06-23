import os
from pathlib import Path

from openai import OpenAI

BASE_MODEL = "QWEN2-VL-2B-Instruct"
DATASET_ARCHIVE = Path("qwen_dataset/qwen_dataset.zip")
OUTPUT_MODEL_SUFFIX = "qwen2-vl-2b-instruct-finetuned"


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set in the environment.")

    if not DATASET_ARCHIVE.exists():
        raise SystemExit(f"Dataset archive not found: {DATASET_ARCHIVE}")

    client = OpenAI()

    print(f"Uploading dataset archive {DATASET_ARCHIVE}...")
    with DATASET_ARCHIVE.open("rb") as f:
        upload = client.files.create(file=f, purpose="fine-tune")

    training_file_id = upload.id
    print(f"Uploaded file id: {training_file_id}")

    print(f"Creating fine-tune job with base model {BASE_MODEL}...")
    job = client.fine_tunes.create(
        training_file=training_file_id,
        model=BASE_MODEL,
        suffix=OUTPUT_MODEL_SUFFIX,
        n_epochs=1,
    )

    print("Fine-tune job created:")
    print(f"  id: {job.id}")
    print(f"  status: {job.status}")
    print(f"  model: {getattr(job, 'fine_tuned_model', None)}")
    print("Use the OpenAI dashboard or API to monitor the fine-tuning job.")


if __name__ == "__main__":
    main()
