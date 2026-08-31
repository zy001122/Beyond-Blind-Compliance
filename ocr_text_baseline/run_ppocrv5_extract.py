import argparse
import json
import os
import time
import traceback
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract OCR transcripts from VeriOCRBench images with PP-OCRv5."
    )
    parser.add_argument("--dataset", required=True, help="Path to VeriOCRBench.json")
    parser.add_argument("--output", required=True, help="Output OCR JSONL path")
    parser.add_argument("--lang", default="en", help="PaddleOCR language, e.g. en/ch")
    parser.add_argument("--ocr-version", default="PP-OCRv5")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--path-prefix-from", default=None)
    parser.add_argument("--path-prefix-to", default=None)
    parser.add_argument("--image-root", default=None, help="Optional root directory prepended to relative image paths")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug-errors", action="store_true")
    return parser.parse_args()


def load_dataset(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit] if limit else data


def load_processed_ids(path):
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def get_image_path(item):
    return item.get("image_path_used") or item.get("corrupted_image") or item.get("original_image")


def remap_path(path, prefix_from, prefix_to):
    if prefix_from and prefix_to and path and path.startswith(prefix_from):
        return prefix_to + path[len(prefix_from):]
    return path


def init_ocr(args):
    from paddleocr import PaddleOCR

    # PaddleOCR 3.x API. Keep fallback for older installations.
    try:
        return PaddleOCR(
            ocr_version=args.ocr_version,
            lang=args.lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except TypeError:
        return PaddleOCR(
            lang=args.lang,
            use_angle_cls=False,
            show_log=False,
        )


def normalize_box(box):
    if box is None:
        return None
    if hasattr(box, "tolist"):
        box = box.tolist()
    try:
        return [[float(p[0]), float(p[1])] for p in box]
    except Exception:
        return box


def to_builtin(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def pick_field(data, names, default=None):
    for name in names:
        if name in data and data[name] is not None:
            return to_builtin(data[name])
    return default


def box_sort_key(item):
    box = item.get("box")
    if not box or not isinstance(box, list):
        return (10**9, 10**9)
    try:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return (min(ys), min(xs))
    except Exception:
        return (10**9, 10**9)


def parse_new_api_result(result):
    rows = []
    pages = result if isinstance(result, list) else [result]
    for page in pages:
        data = None
        if isinstance(page, dict):
            data = page
        elif hasattr(page, "json"):
            data = page.json
            if callable(data):
                data = data()
        elif hasattr(page, "res"):
            data = page.res
            if callable(data):
                data = data()
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                data = None
        if isinstance(data, dict) and "res" in data and isinstance(data["res"], dict):
            data = data["res"]
        if not isinstance(data, dict):
            continue

        texts = pick_field(data, ["rec_texts", "texts", "text"], [])
        scores = pick_field(data, ["rec_scores", "scores"], [])
        boxes = pick_field(data, ["rec_polys", "dt_polys", "rec_boxes", "boxes"], [])
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list):
            continue
        for i, text in enumerate(texts):
            text = str(text).strip()
            if not text:
                continue
            score = to_builtin(scores[i]) if i < len(scores) else None
            try:
                score = float(score) if score is not None else None
            except Exception:
                pass
            rows.append({
                "text": text,
                "score": score,
                "box": normalize_box(boxes[i]) if i < len(boxes) else None,
            })
    return rows


def parse_old_api_result(result):
    rows = []
    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if page is None:
            continue
        # Old PaddleOCR: [[box, (text, score)], ...]
        if isinstance(page, list):
            lines = page
        else:
            continue
        for line in lines:
            try:
                box = line[0]
                text = line[1][0]
                score = line[1][1]
            except Exception:
                continue
            text = str(text).strip()
            if text:
                score = to_builtin(score)
                try:
                    score = float(score) if score is not None else None
                except Exception:
                    pass
                rows.append({"text": text, "score": score, "box": normalize_box(box)})
    return rows


def run_ocr(ocr, image_path):
    if hasattr(ocr, "predict"):
        result = ocr.predict(image_path)
        return parse_new_api_result(result)
    if hasattr(ocr, "ocr"):
        result = ocr.ocr(image_path, cls=False)
        return parse_old_api_result(result)
    raise AttributeError("PaddleOCR object has neither predict() nor ocr() method")


def make_transcript(rows):
    rows = sorted(rows, key=box_sort_key)
    lines = []
    for i, row in enumerate(rows, 1):
        score = row.get("score")
        score_text = f", conf={float(score):.3f}" if isinstance(score, (int, float)) else ""
        box = row.get("box")
        box_text = f", box={box}" if box is not None else ""
        lines.append(f"[{i:03d}] {row['text']} ({score_text.lstrip(', ')}{box_text})")
    return "\n".join(lines)


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output.exists():
        output.unlink()

    dataset = load_dataset(args.dataset, args.limit)
    processed_ids = load_processed_ids(output)
    todo = [item for item in dataset if item["id"] not in processed_ids]
    print(f"Loaded {len(dataset)} samples; {len(todo)} remaining.")
    print(f"Writing OCR JSONL to {output}")

    ocr = init_ocr(args)
    for item in tqdm(todo, desc=f"PP-OCRv5 extract ({args.lang})"):
        image_path = remap_path(get_image_path(item), args.path_prefix_from, args.path_prefix_to)
        if args.image_root and image_path and not os.path.isabs(image_path):
            image_path = os.path.join(args.image_root, image_path)
        record = {
            "id": item.get("id"),
            "error_type": item.get("error_type"),
            "source_domain": item.get("original_info", {}).get("source_domain"),
            "image_path_used": image_path,
            "ocr_model": args.ocr_version,
            "ocr_lang": args.lang,
            "ocr_items": [],
            "ocr_text": "",
            "status": "ok",
        }
        if not image_path or not os.path.exists(image_path):
            record["status"] = "missing_image"
            record["ocr_text"] = f"[ERROR: missing image path: {image_path}]"
            append_jsonl(output, record)
            continue
        try:
            rows = run_ocr(ocr, image_path)
            record["ocr_items"] = rows
            record["ocr_text"] = make_transcript(rows) if rows else "[NO_TEXT_DETECTED]"
        except Exception as exc:
            record["status"] = "ocr_error"
            record["ocr_text"] = f"[ERROR: {exc}]"
            if args.debug_errors:
                record["traceback"] = traceback.format_exc()
        append_jsonl(output, record)
        if args.sleep > 0:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
