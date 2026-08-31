"""
Anonymous Inference Script for VeriOCRBench
Executes the Dual-Paradigm (Proactive & Assisted) evaluation pipeline across various MLLMs.
Supports robust error handling, exponential backoff, and resume-from-checkpoint via JSONL.
"""

import os
import json
import time
import base64
import argparse
from tqdm import tqdm
from openai import OpenAI

# ==========================================
# 1. Prompt Templates (Strictly Aligned with the Paper)
# ==========================================

# Paradigm A: Proactive Evaluation (no explicit verification warning)
PROACTIVE_PROMPT_TEMPLATE = """[Image Premise]: {{IP}}

[Textual Premise]: {TP}

[Question]: {Q}
"""

# Paradigm B: Assisted Evaluation (Explicit meta-instruction for task verification)
ASSISTED_PROMPT_TEMPLATE = """[Image Premise]: {{IP}}

[Textual Premise]: {TP}

[Question]: {Q}

Do NOT answer the Question immediately. You MUST first rigorously verify if the requested task is actually completable based on the Image Premise. If the [Textual Premise] contradicts the visual facts in the Image Premise, contains a logical fallacy, or if the Image Premise lacks the necessary information to solve the [Question], you must point out the error and explicitly refuse to answer."""

# ==========================================
# 2. API Utility Functions
# ==========================================

def encode_image_to_base64(image_path):
    """Encodes a local image to Base64 format for OpenAI-compatible vision APIs."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_with_retry(client, model_name, base64_image, prompt_text, max_retries=5):
    """
    Executes the API call with robust error handling, handling rate limits (429),
    and safety filtering bypasses through exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.0, # Deterministic setting for benchmark evaluation
                max_tokens=512,
            )
            
            content = response.choices[0].message.content
            
            # Safeguard against strict safety filters returning None or empty strings
            if content is None or content.strip() == "":
                raise ValueError("API unexpectedly returned an empty string (Possible safety filter trigger).")
                
            return content.strip()
            
        except Exception as e:
            err_msg = str(e)
            tqdm.write(f"  [Warning] API call failed (Attempt {attempt+1}/{max_retries}): {err_msg}")
            
            if attempt == max_retries - 1:
                tqdm.write("  Error: Max retries exhausted. Recording error.")
                return f"[ERROR: {err_msg}]" 
            
            # Exponential backoff for rate limits
            if "429" in err_msg or "Rate limit" in err_msg or "Too Many Requests" in err_msg:
                wait_time = (attempt + 1) * 20
                tqdm.write(f"   Rate limit hit. Sleeping for {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                wait_time = (attempt + 1) * 5
                tqdm.write(f"  Network or internal error. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            
    return "[ERROR: Max retries exceeded strictly]"


def resolve_image_path(image_path, image_root=None):
    """Resolve relative dataset image paths against an optional image root."""
    if not image_path:
        return None
    if os.path.isabs(image_path):
        return image_path
    if image_root:
        return os.path.join(image_root, image_path)
    return image_path

# ==========================================
# 3. Main Inference Pipeline
# ==========================================

def run_inference_pipeline(args):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    
    if not client.api_key:
        print("WARNING: OPENAI_API_KEY environment variable is not set.")

    input_json_path = args.input
    output_jsonl_path = os.path.join(args.output_dir, f"raw_responses_{args.model.replace('/', '_')}.jsonl")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # Resume-from-checkpoint logic (JSONL mode)
    # ---------------------------------------------------------
    processed_ids = set()
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["id"])
                except json.JSONDecodeError:
                    continue
        print(f" Resuming session: Found {len(processed_ids)} previously evaluated samples.")

    if not os.path.exists(input_json_path):
        print(f"Error: Dataset file not found at {input_json_path}")
        return

    with open(input_json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # ---------------------------------------------------------
    # Evaluation Loop
    # ---------------------------------------------------------
    with open(output_jsonl_path, "a", encoding="utf-8") as out_file:
        for item in tqdm(dataset, desc=f"Evaluating {args.model}"):
            item_id = item["id"]
            if item_id in processed_ids:
                continue

            # Fallback chain for image path compatibility
            image_path = item.get("image_path_used", item.get("corrupted_image", item.get("original_image")))
            image_path = resolve_image_path(image_path, args.image_root)
            
            if not image_path or not os.path.exists(image_path):
                tqdm.write(f"\n[Skip] Image file not found: {image_path} (ID: {item_id})")
                continue

            base64_image = encode_image_to_base64(image_path)

            # Format Prompts
            proactive_prompt = PROACTIVE_PROMPT_TEMPLATE.format(TP=item["TP"], Q=item["Q"])
            assisted_prompt = ASSISTED_PROMPT_TEMPLATE.format(TP=item["TP"], Q=item["Q"])

            # Execute Dual-Paradigm Inference
            proactive_resp = generate_with_retry(client, args.model, base64_image, proactive_prompt)
            assisted_resp = generate_with_retry(client, args.model, base64_image, assisted_prompt)

            # Preserve all original metadata and append the model's raw responses
            item["model_responses"] = {
                "proactive": proactive_resp,
                "assisted": assisted_resp
            }

            # Write incrementally to disk
            out_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            out_file.flush()
            
            # Brief pause to respect API rate limits during large-scale evaluation
            time.sleep(1.0) 

    print(f"\nStage 1 (Inference) completed for {args.model}! Results saved to {output_jsonl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VeriOCRBench Multimodal Model Evaluation Pipeline")
    parser.add_argument("--model", type=str, default="gpt-4o", help="The identifier of the MLLM to evaluate")
    parser.add_argument("--input", type=str, default="../data/VeriOCRBench.json", help="Path to the VeriOCRBench metadata JSON")
    parser.add_argument("--output_dir", type=str, default="Evaluation_Results", help="Directory to save the JSONL outputs")
    parser.add_argument("--image_root", "--image-root", type=str, default=None, help="Optional root directory prepended to relative image paths")
    
    args = parser.parse_args()
    run_inference_pipeline(args)
