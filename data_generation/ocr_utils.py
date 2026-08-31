"""Shared PP-OCRv5 helpers for VeriOCRBench data construction.

The paper reports PP-OCRv5 as the OCR extractor used to obtain Visual Atomic
Facts (VAFs). This module centralizes the PaddleOCR initialization so the
released construction scripts use the same default OCR version.
"""

import json


def init_paddleocr(lang="ch", ocr_version="PP-OCRv5"):
    """Initialize PaddleOCR with PP-OCRv5 by default.

    PaddleOCR 3.x exposes the ``ocr_version`` argument. The fallback keeps the
    scripts usable with older PaddleOCR installations, while still recording
    PP-OCRv5 as the intended version in the released code and README.
    """
    from paddleocr import PaddleOCR

    try:
        return PaddleOCR(
            ocr_version=ocr_version,
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except TypeError:
        return PaddleOCR(
            lang=lang,
            use_angle_cls=False,
            show_log=False,
        )


def _to_builtin(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _pick_field(data, names, default=None):
    for name in names:
        if name in data and data[name] is not None:
            return _to_builtin(data[name])
    return default


def _normalize_quad(box):
    if box is None:
        return None
    if hasattr(box, "tolist"):
        box = box.tolist()
    try:
        return [[float(p[0]), float(p[1])] for p in box]
    except Exception:
        return box


def _quad_to_rect(box):
    box = _normalize_quad(box)
    if not box or not isinstance(box, list):
        return []
    try:
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
    except Exception:
        return []


def _parse_new_api_result(result):
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

        texts = _pick_field(data, ["rec_texts", "texts", "text"], [])
        scores = _pick_field(data, ["rec_scores", "scores"], [])
        boxes = _pick_field(data, ["rec_polys", "dt_polys", "rec_boxes", "boxes"], [])
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list):
            continue
        for i, text in enumerate(texts):
            text = str(text).strip()
            if not text:
                continue
            score = _to_builtin(scores[i]) if i < len(scores) else None
            try:
                score = float(score) if score is not None else None
            except Exception:
                pass
            rows.append({
                "text": text,
                "score": score,
                "quad": _normalize_quad(boxes[i]) if i < len(boxes) else None,
                "bbox": _quad_to_rect(boxes[i]) if i < len(boxes) else [],
            })
    return rows


def _parse_old_api_result(result):
    rows = []
    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if page is None or not isinstance(page, list):
            continue
        for line in page:
            try:
                box = line[0]
                text = str(line[1][0]).strip()
                score = float(_to_builtin(line[1][1]))
            except Exception:
                continue
            if text:
                rows.append({
                    "text": text,
                    "score": score,
                    "quad": _normalize_quad(box),
                    "bbox": _quad_to_rect(box),
                })
    return rows


def run_paddleocr(ocr, image_path):
    """Run PaddleOCR and return normalized text rows."""
    if hasattr(ocr, "predict"):
        return _parse_new_api_result(ocr.predict(image_path))
    if hasattr(ocr, "ocr"):
        return _parse_old_api_result(ocr.ocr(image_path, cls=False))
    raise AttributeError("PaddleOCR object has neither predict() nor ocr() method")


def extract_ocr_facts_from_image(ocr, image_path, min_conf=0.85):
    """Extract high-confidence Visual Atomic Facts from an image."""
    facts = []
    for idx, row in enumerate(run_paddleocr(ocr, image_path), 1):
        text = row.get("text", "").strip()
        score = row.get("score")
        if not text:
            continue
        if isinstance(score, (int, float)) and score < min_conf:
            continue
        facts.append({
            "ocr_id": idx,
            "content": text,
            "bbox": row.get("bbox", []),
            "confidence": round(float(score), 3) if isinstance(score, (int, float)) else None,
        })
    return facts
