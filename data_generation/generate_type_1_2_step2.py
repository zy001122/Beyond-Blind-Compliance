"""
Anonymous Task Synthesis Script for VeriOCRBench
Step 2 of the 'Generate-then-Corrupt' Pipeline for Perceptual Traps (Type 1 & Type 2).
"""

import os
import json
import base64
import time
from openai import OpenAI
from paper_prompt_templates import PERCEPTUAL_TRAP_PROMPT_TEMPLATE

# ==========================================
# 1. Configuration & Anonymous Init
# ==========================================
API_KEY = os.environ.get("API_KEY", "YOUR_API_KEY")
BASE_URL = os.environ.get("BASE_URL", "YOUR_BASE_URL")

MODEL_NAME = os.environ.get("GENERATOR_MODEL", "gpt-4o")

# Input paths aligned with the updated taxonomy and Step 1 output
INPUT_FILES = [
    "VeriOCR_Synthetic_Data/Step1_Output_type1_degradation.json", 
    "VeriOCR_Synthetic_Data/Step1_Output_type2_occlusion.json"
]
OUTPUT_DIR = "VeriOCR_Final_Benchmark"

# ==========================================
# 2. Core Generation Engine (Generate-then-Corrupt)
# ==========================================
class DeceptiveTaskGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def construct_prompt(self, item):
        """
        Constructs the prompt exactly as presented in Appendix Figure 11.
        """
        candidates_str = ""
        for cand in item['candidates']:
            candidates_str += f"- Candidate ID: {cand['candidate_id']} | Bounding Box: {cand['target_bbox']} | Exact Text: \"{cand['ground_truth_hidden']}\"\n"

        return PERCEPTUAL_TRAP_PROMPT_TEMPLATE.format(candidates_str=candidates_str)

    def generate_task(self, item, max_retries=3):
        img_path = item['original_image']
        if not os.path.exists(img_path): return None
        
        b64_img = self.encode_image(img_path)
        prompt = self.construct_prompt(item)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                            ]
                        }
                    ],
                    max_tokens=800,
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                res_json = json.loads(content)
                
                chosen_id = res_json.get("selected_candidate_id")
                chosen_candidate = next((c for c in item['candidates'] if c['candidate_id'] == chosen_id), None)
                
                if not chosen_candidate:
                    print(f"  [Attempt {attempt+1}] Model returned invalid candidate_id: {chosen_id}")
                    continue

                # Final assembly linking the generated natural task to the corrupted image
                return {
                    "original_info": item.get("original_info", {}),
                    "original_image": item['original_image'],
                    "corrupted_image": chosen_candidate['corrupted_image'],
                    "task_type": item['task_type'],
                    "visual_status": chosen_candidate['visual_status'],
                    "ground_truth_hidden": chosen_candidate['ground_truth_hidden'],
                    "target_bbox": chosen_candidate['target_bbox'],
                    "reasoning_type": res_json.get("reasoning_type"),
                    "TP": res_json.get("TP"),
                    "Q": res_json.get("Q"),
                    "GT_Answer": res_json.get("GT_Answer"),
                    "trap_logic_trace": {
                        "corrupted_status": chosen_candidate['visual_status'],
                        "reasoning_type": res_json.get("reasoning_type"),
                        "rationale": res_json.get("trap_explanation")
                    }
                }
            except json.JSONDecodeError as e:
                print(f"  [Attempt {attempt+1}] JSON decode failed, retrying: {e}")
            except Exception as e:
                print(f"  [Attempt {attempt+1}] API/Network error: {e}")
            
            time.sleep(1.5)
        
        return None

    def run(self):
        if not API_KEY:
            print("WARNING: API_KEY environment variable is missing. The script will likely fail.")

        for input_file in INPUT_FILES:
            if not os.path.exists(input_file): 
                print(f"WARNING: Input file not found, skipping: {input_file}")
                continue
            
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            final_results = []
            # Align naming with the paper's "Deceptive Traps" terminology
            out_name = os.path.basename(input_file).replace("Step1_Output_", "Final_Deceptive_")
            
            print(f"\n Generating purely natural reasoning tasks for {input_file}...")
            for i, item in enumerate(data):
                res = self.generate_task(item)
                if res:
                    final_results.append(res)
                    print(f"[{i+1}/{len(data)}] Selected Fact: {res['ground_truth_hidden']}")
                    print(f"  -> Q: {res['Q']}")
            
            out_path = os.path.join(OUTPUT_DIR, out_name)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            print(f"Finished generating: {out_name}")

if __name__ == "__main__":
    gen = DeceptiveTaskGenerator()
    gen.run()
