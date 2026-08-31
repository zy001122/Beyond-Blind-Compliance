"""
Anonymous Task Synthesis Script for VeriOCRBench
Generates contextual, factual, and logical traps (Types 3-8) using an OCR-grounded LLM generation pipeline.
"""

import os
import json
import base64
import argparse
import time
import re
import logging
from tqdm import tqdm
from openai import OpenAI
from ocr_utils import extract_ocr_facts_from_image, init_paddleocr
from paper_prompt_templates import (
    PROMPT_DOMAIN_MISMATCH as PAPER_PROMPT_DOMAIN_MISMATCH,
    PROMPT_FACT_TAMPERING as PAPER_PROMPT_FACT_TAMPERING,
    PROMPT_HALLUCINATORY_TARGET as PAPER_PROMPT_HALLUCINATORY_TARGET,
    PROMPT_LOGICAL_PARADOX as PAPER_PROMPT_LOGICAL_PARADOX,
    PROMPT_MISSING_VARIABLE as PAPER_PROMPT_MISSING_VARIABLE,
    PROMPT_VAGUE_REFERENCE as PAPER_PROMPT_VAGUE_REFERENCE,
)

# =================  Configuration & Anonymous Init =================
# STRICTLY ANONYMOUS: Read API key from environment variables.
API_KEY = os.environ.get("API_KEY", "") 
BASE_URL = os.environ.get("BASE_URL", "") 
MODEL_NAME = os.environ.get("GENERATOR_MODEL", "gpt-4o")

DATASET_ROOT = "./" 
TASK_ALLOCATION_DIR = os.path.join(DATASET_ROOT, "experiment_samples")
OUTPUT_DIR = "./VeriOCR_Synthetic_Data/Cognitive_Traps"

# ================= Initialize PaddleOCR =================
# Suppress verbose debug logs to keep tqdm progress bar clean
logging.getLogger("ppocr").setLevel(logging.WARNING)

OCR_VERSION = os.environ.get("PADDLEOCR_VERSION", "PP-OCRv5")
OCR_LANG = os.environ.get("PADDLEOCR_LANG", "ch")

print(f"Loading PaddleOCR ({OCR_VERSION}, lang={OCR_LANG})...")
ocr_engine = None
try:
    ocr_engine = init_paddleocr(lang=OCR_LANG, ocr_version=OCR_VERSION)
    print("PaddleOCR successfully loaded.")
except ImportError:
    print("Error: PaddleOCR is not installed. Run `pip install paddleocr` and install PaddlePaddle for your platform.")
    exit(1)

# Prompt templates are imported from paper_prompt_templates.py so the released code
# stays synchronized with the appendix.

# Master dictionary mapping new taxonomy types to their prompts
ALL_PROMPTS = {
    "type3": PAPER_PROMPT_DOMAIN_MISMATCH,
    "type4": PAPER_PROMPT_HALLUCINATORY_TARGET,
    "type5": PAPER_PROMPT_FACT_TAMPERING,
    "type6": PAPER_PROMPT_VAGUE_REFERENCE,
    "type7": PAPER_PROMPT_LOGICAL_PARADOX,
    "type8": PAPER_PROMPT_MISSING_VARIABLE,
}

# =================  Utilities =================
def get_actual_image_path(relative_path_from_json):
    full_path_guess = os.path.join(DATASET_ROOT, relative_path_from_json)
    if os.path.exists(full_path_guess):
        return full_path_guess
    
    base_path_no_ext = os.path.splitext(full_path_guess)[0]
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG']
    for ext in extensions:
        candidate = base_path_no_ext + ext
        if os.path.exists(candidate):
            return candidate
    return None

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_llm_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match: return json.loads(match.group(1))
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match: return json.loads(match.group(1))
        return None

# ================= OCR Extraction Module =================
def extract_ocr_facts(image_path):
    """
    Extract text with PP-OCRv5 to serve as the VAF constraint for the LLM.
    """
    if ocr_engine is None:
        return []
    try:
        return extract_ocr_facts_from_image(ocr_engine, image_path, min_conf=0.85)
    except Exception as e:
        print(f"OCR extraction failed for {image_path}: {e}")
        return []

# ================= Core Generation Engine =================
def generate_task(client, model, image_path, type_key):
    base64_image = encode_image(image_path)
    
    full_prompt = ALL_PROMPTS.get(type_key)
    if not full_prompt:
        print(f"Error: Prompt for '{type_key}' not found.")
        return None 

    # Execute OCR Extraction
    ocr_results = extract_ocr_facts(image_path)
    ocr_json_str = json.dumps(ocr_results, ensure_ascii=False, indent=2)

    user_message_content = f"""Please analyze this image and generate the verification task JSON according to the system instructions.

*** STRICT OCR FACTS LIST (GROUND TRUTH) ***
Below is the precise text physically detected in the image by our OCR engine. 
CRITICAL RULE: You MUST ONLY select your "Atomic Facts" from this list. DO NOT hallucinate, guess, or invent any text (like anatomical names, external labels) that is not explicitly in this list. If the list is empty, adapt the task to rely purely on visual objects instead of text.

{ocr_json_str}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": full_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message_content},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ],
                }
            ],
            temperature=0.7,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        return parse_llm_json(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: API Error: {e}")
        return None

def main():
    if not API_KEY:
        print("WARNING: API_KEY environment variable is missing. The script will likely fail.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='5', help="Error type to generate (e.g., 3 to 8 aligned with the final taxonomy)")
    args = parser.parse_args()
    
    raw_type = args.type
    type_num = re.findall(r'\d+', raw_type)[0]
    target_type = f"type{type_num}"

    # Ensure the input files match your local directory structure
    input_json_name = f"Type_{type_num}.json"
    input_json_path = os.path.join(TASK_ALLOCATION_DIR, input_json_name)

    if not os.path.exists(input_json_path):
        print(f"Error: Input task file not found: {input_json_path}")
        print("Please ensure your input JSON files are named Type_3.json to Type_8.json matching the new taxonomy.")
        return

    with open(input_json_path, 'r', encoding='utf-8') as f:
        source_tasks = json.load(f)
    
    print(f" Starting Generation - Taxonomy: {target_type} (Guarded by PaddleOCR)")
    print(f" Reading file: {input_json_path} (Total {len(source_tasks)} images)")
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_file = os.path.join(OUTPUT_DIR, f"Final_Deceptive_{target_type}.json")
    results = []
    
    for task in tqdm(source_tasks):
        raw_img_path = task.get('image_path', '')
        real_img_path = get_actual_image_path(raw_img_path)
        
        if not real_img_path:
            continue
            
        data = generate_task(client, MODEL_NAME, real_img_path, target_type)
        
        if data:
            trap_logic = data.get("trap_logic", {})
            task_gen = data.get("task_generation", {})
            
            target_id = trap_logic.get("target_fact_id")
            gt_bbox = []
            if data.get("atomic_facts"):
                for fact in data["atomic_facts"]:
                    if fact.get("id") == target_id:
                        gt_bbox = fact.get("bbox", [])
                        break

            entry = {
                "id": f"gen_{target_type}_{int(time.time()*1000)}_{len(results)}",
                "original_info": task,
                "image_path_used": real_img_path,
                "error_type": target_type,
                "document_type": data.get("document_type"),
                "full_atomic_facts": data.get("atomic_facts"), 
                "trap_logic_trace": trap_logic, 
                "gt_evidence_bbox": gt_bbox, 
                "TP": task_gen.get("TP"),
                "Q": task_gen.get("Q")
            }
            results.append(entry)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        time.sleep(0.2)

    print(f"Completed! Saved {len(results)} results to {output_file}")

if __name__ == "__main__":
    main()
