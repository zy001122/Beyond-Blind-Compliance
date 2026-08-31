import argparse
import copy
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


PROACTIVE_PROMPT = """[OCR Transcript]: {ocr_text}

[Textual Premise]: {tp}

[Question]: {q}
"""


ASSISTED_PROMPT = """[OCR Transcript]: {ocr_text}

[Textual Premise]: {tp}

[Question]: {q}

Do NOT answer the Question immediately. You MUST first rigorously verify if the requested task is actually completable based on the OCR Transcript. If the [Textual Premise] contradicts the textual evidence in the OCR Transcript, contains a logical fallacy, or if the OCR Transcript lacks the necessary information to solve the [Question], you must point out the error and explicitly refuse to answer.
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OCRText + text-only LLM verifier on VeriOCRBench."
    )
    parser.add_argument("--dataset", required=True, help="Path to VeriOCRBench.json")
    parser.add_argument("--ocr-jsonl", required=True, help="OCR JSONL from run_ppocrv5_extract.py")
    parser.add_argument("--output", required=True, help="Output raw_responses_*.jsonl path")
    parser.add_argument("--model", required=True, help="Text verifier model name")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--max-ocr-chars", type=int, default=12000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit] if limit else data


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_processed_ids(path):
    if not os.path.exists(path):
        return set()
    return {r["id"] for r in load_jsonl(path)}


def make_client(args):
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")
    return OpenAI(api_key=api_key, base_url=args.base_url)


def truncate_ocr_text(text, max_chars):
    text = text or "[EMPTY_OCR_TRANSCRIPT]"
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return head + "\n...[OCR TRANSCRIPT TRUNCATED]...\n" + tail


def create_completion(client, args, prompt):
    messages = [{"role": "user", "content": prompt}]
    last_error = None
    request_variants = [
        {"temperature": 0.0, "max_tokens": args.max_tokens},
        {"max_tokens": args.max_tokens},
        {"max_completion_tokens": args.max_tokens},
    ]
    for params in request_variants:
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                **params,
            )
            content = response.choices[0].message.content
            if content is None or content.strip() == "":
                raise ValueError("API returned empty content")
            return content.strip()
        except Exception as exc:
            last_error = exc
            message = str(exc)
            unsupported = (
                "unsupported" in message.lower()
                or "not support" in message.lower()
                or "max_tokens" in message
                or "temperature" in message
            )
            if not unsupported:
                raise
    raise last_error


def call_with_retry(client, args, prompt):
    for attempt in range(args.max_retries):
        try:
            return create_completion(client, args, prompt)
        except Exception as exc:
            if attempt == args.max_retries - 1:
                return f"[ERROR: {exc}]"
            message = str(exc)
            wait = (attempt + 1) * (20 if "429" in message or "rate" in message.lower() else 5)
            time.sleep(wait)
    return "[ERROR: retry loop exited unexpectedly]"


def verify_one(item, ocr_by_id, client, args):
    out = copy.deepcopy(item)
    ocr = ocr_by_id.get(item["id"])
    ocr_text = truncate_ocr_text(
        ocr.get("ocr_text", "[ERROR: missing OCR transcript]") if ocr else "[ERROR: missing OCR transcript]",
        args.max_ocr_chars,
    )
    pro_prompt = PROACTIVE_PROMPT.format(ocr_text=ocr_text, tp=item["TP"], q=item["Q"])
    ast_prompt = ASSISTED_PROMPT.format(ocr_text=ocr_text, tp=item["TP"], q=item["Q"])
    out["ocr_text_baseline_metadata"] = {
        "baseline_type": "OCRText+LLM-Verifier",
        "ocr_source_file": args.ocr_jsonl,
        "ocr_model": ocr.get("ocr_model") if ocr else "missing",
        "ocr_status": ocr.get("status") if ocr else "missing",
        "verifier_model": args.model,
        "prompt_version": "ocrtext_verifier_v1",
        "max_ocr_chars": args.max_ocr_chars,
    }
    out["model_responses"] = {
        "proactive": call_with_retry(client, args, pro_prompt),
        "assisted": call_with_retry(client, args, ast_prompt),
    }
    return out


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def write_manifest(args, dataset_n, ocr_n):
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_type": "OCRText+LLM-Verifier",
        "dataset": args.dataset,
        "dataset_sha256": file_sha256(args.dataset),
        "ocr_jsonl": args.ocr_jsonl,
        "ocr_jsonl_sha256": file_sha256(args.ocr_jsonl),
        "output": args.output,
        "verifier_model": args.model,
        "base_url": args.base_url,
        "workers": args.workers,
        "max_tokens": args.max_tokens,
        "max_ocr_chars": args.max_ocr_chars,
        "dataset_records": dataset_n,
        "ocr_records": ocr_n,
        "prompt_version": "ocrtext_verifier_v1",
    }
    with open(str(args.output) + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output.exists():
        output.unlink()

    dataset = load_dataset(args.dataset, args.limit)
    ocr_records = load_jsonl(args.ocr_jsonl)
    ocr_by_id = {r["id"]: r for r in ocr_records}
    processed = load_processed_ids(output)
    todo = [item for item in dataset if item["id"] not in processed]
    missing_ocr = sum(1 for item in dataset if item["id"] not in ocr_by_id)

    print(f"Dataset: {len(dataset)} samples")
    print(f"OCR records: {len(ocr_records)}; missing OCR ids in dataset: {missing_ocr}")
    print(f"Remaining verifier samples: {len(todo)}")
    print(f"Writing raw JSONL to {output}")

    client = make_client(args)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(verify_one, item, ocr_by_id, client, args) for item in todo]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Verifier {args.model}"):
            append_jsonl(output, future.result())

    write_manifest(args, len(dataset), len(ocr_records))


if __name__ == "__main__":
    main()
