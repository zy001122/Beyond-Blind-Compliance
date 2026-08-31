import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Audit VeriOCRBench JSONL outputs.")
    parser.add_argument("--path", required=True)
    return parser.parse_args()


def is_error(text):
    return isinstance(text, str) and text.strip().startswith("[ERROR:")


def main():
    args = parse_args()
    records = []
    invalid = []
    with open(args.path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except Exception as exc:
                invalid.append({"line": i, "error": str(exc)})

    ids = [r.get("id") for r in records]
    type_counts = {}
    status_counts = {}
    errors = {"proactive": 0, "assisted": 0, "either": 0}
    missing = {"model_responses": 0, "proactive": 0, "assisted": 0, "ocr_text": 0}
    for r in records:
        et = str(r.get("error_type", "missing")).lower()
        type_counts[et] = type_counts.get(et, 0) + 1
        if "status" in r:
            status = r.get("status", "missing")
            status_counts[status] = status_counts.get(status, 0) + 1
        if "ocr_text" in r and not r.get("ocr_text"):
            missing["ocr_text"] += 1
        mr = r.get("model_responses")
        if not mr:
            missing["model_responses"] += 1
            continue
        pro = mr.get("proactive")
        ast = mr.get("assisted")
        if not pro:
            missing["proactive"] += 1
        if not ast:
            missing["assisted"] += 1
        pro_err = is_error(pro)
        ast_err = is_error(ast)
        errors["proactive"] += int(pro_err)
        errors["assisted"] += int(ast_err)
        errors["either"] += int(pro_err or ast_err)

    print(json.dumps({
        "path": os.path.abspath(args.path),
        "records": len(records),
        "unique_ids": len(set(ids)),
        "duplicate_ids": len(ids) - len(set(ids)),
        "invalid_json_lines": invalid[:10],
        "invalid_json_count": len(invalid),
        "type_counts": type_counts,
        "status_counts": status_counts,
        "missing": missing,
        "api_errors": errors,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
