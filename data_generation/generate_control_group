"""
Anonymous Task Synthesis Script for VeriOCRBench
Generates the Control Group (Trap-Free Normal Tasks) to evaluate the Over-Refusal Rate (ORR).
"""

import json
import os
import time
import uuid
import base64
import logging
from openai import OpenAI
from tqdm import tqdm
from ocr_utils import extract_ocr_facts_from_image, init_paddleocr
from paper_prompt_templates import PROMPT_TYPE_NORMAL as PAPER_PROMPT_TYPE_NORMAL

# =================  Configuration & Anonymous Init =================
# STRICTLY ANONYMOUS: Read API key from environment variables.
API_KEY = os.environ.get("API_KEY", "YOUR_API_KEY")
BASE_URL = os.environ.get("BASE_URL", "YOUR_BASE_URL")
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "gpt-4o") 

DATASET_ROOT = "./" 
TASK_ALLOCATION_DIR = os.path.join(DATASET_ROOT, "experiment_samples")
OUTPUT_DIR = "./VeriOCR_Synthetic_Data/Control_Group"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= Initialize PaddleOCR =================
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

# Prompt template is imported from paper_prompt_templates.py so the released code
# stays synchronized with the appendix.

# Use the paper-aligned canonical template so the released code and appendix stay synchronized.
PROMPT_TYPE_NORMAL = PAPER_PROMPT_TYPE_NORMAL

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

def extract_ocr_facts(image_path):
    if ocr_engine is None:
        return []
    try:
        return extract_ocr_facts_from_image(ocr_engine, image_path, min_conf=0.85)
    except Exception as e:
        print(f"OCR extraction failed: {e}")
        return []

# ================= Core Generation Engine =================
def generate_normal_sample(image_path):
    """Invokes the LLM to generate Control Group data grounded in OCR facts."""
    
    ocr_results = extract_ocr_facts(image_path)
    ocr_json_str = json.dumps(ocr_results, ensure_ascii=False, indent=2)
    
    user_message_content = f"""Please analyze this image and generate the NORMAL verification task JSON according to the system instructions.

*** STRICT OCR FACTS LIST (GROUND TRUTH) ***
Below is the precise text physically detected in the image by our OCR engine. 
CRITICAL RULE: You MUST ONLY select your "Atomic Facts" from this list. DO NOT hallucinate, guess, or invent any text (like Product A, Item B).

{ocr_json_str}"""
    
    base64_image = encode_image(image_path)
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=GENERATOR_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": PROMPT_TYPE_NORMAL},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_message_content},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ],
                    }
                ],
                temperature=0.1 #  Core: Low temperature prevents hallucinations in control tasks
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"  [Generation Error - Attempt {attempt+1}/3] {e}")
            time.sleep(2)
    return None

def run_generation():
    if not API_KEY:
        print("WARNING: API_KEY environment variable is missing. The script will likely fail.")

    SOURCE_DATA_PATH = os.path.join(TASK_ALLOCATION_DIR, "Type_normal.json") 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "Final_Control_Tasks.json")
    
    if not os.path.exists(SOURCE_DATA_PATH):
        print(f"Error: Input task file not found {SOURCE_DATA_PATH}")
        return

    with open(SOURCE_DATA_PATH, "r", encoding="utf-8") as f:
        source_data = json.load(f)
        
    results = []
    
    for item in tqdm(source_data, desc="Generating Control Group"):
        raw_img_path = item.get("image_path", "")
        real_img_path = get_actual_image_path(raw_img_path)
        
        if not real_img_path:
            print(f"WARNING: Image not found: {raw_img_path}")
            continue
            
        gen_result = generate_normal_sample(real_img_path)
        
        if gen_result:
            final_record = {
                "id": f"gen_normal_{str(uuid.uuid4())[:8]}",
                "original_info": {
                    "image_path": raw_img_path,
                    "source_domain": item.get("source_domain", "general")
                },
                "image_path_used": real_img_path,
                "error_type": "normal",
                "document_type": gen_result.get("document_type", ""),
                "full_atomic_facts": gen_result.get("atomic_facts", []),
                "trap_logic_trace": gen_result.get("trap_logic", {}),
                "TP": gen_result.get("task_generation", {}).get("TP", ""),
                "Q": gen_result.get("task_generation", {}).get("Q", "")
            }
            results.append(final_record)
            
            # Write to disk incrementally to prevent data loss on interruption
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
        time.sleep(0.5)

    print(f"Successfully generated {len(results)} control group tasks! Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_generation()
