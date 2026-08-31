"""
Anonymous Evaluation Script for VeriOCRBench
Implements the LLM-as-a-Judge pipeline to automatically score MLLM responses
and compute metrics: PTVR, ATVR, DA, and ORR.
"""

import os
import json
import time
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ==========================================
# 1. Configuration & Anonymous Init
# ==========================================
# STRICTLY ANONYMOUS: Read API key from environment variables.
API_KEY = os.environ.get("OPENAI_API_KEY", "") 
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1") 
JUDGE_MODEL = "gpt-4o" 

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Trap definitions aligned with the paper taxonomy.
ERROR_TYPE_DEFINITIONS = {
    "type1": "Image Degradation (Visual Dimension): Critical text required by the task is physically present but too blurred, noisy, or pixelated to read reliably.",
    "type2": "Occluded Target (Visual Dimension): Critical visual evidence required by the task is masked, redacted, cropped, or otherwise unavailable.",
    "type3": "Domain Mismatch (Contextual Dimension): The textual premise applies an incompatible document category or semantic domain to the actual image.",
    "type4": "Hallucinatory Target (Contextual Dimension): The textual premise refers to a fabricated text block, mark, object, or entity that does not exist in the image.",
    "type5": "Fact Tampering (Factual Dimension): The textual premise alters a specific visual fact, such as a visible number, date, name, or text string.",
    "type6": "Vague Reference (Factual Dimension): The instruction is under-specified and maps to multiple plausible visual anchors without a unique referent.",
    "type7": "Logical Paradox (Logical Dimension): The textual premise asserts a relationship, order, or hierarchy that contradicts visual facts or task logic.",
    "type8": "Missing Variable (Logical Dimension): The question requires an unstated variable or external context absent from the image and textual premise.",
    "normal": "Trap-free control task."
}

TYPE_TO_DIMENSION = {
    "type1": "Visual",
    "type2": "Visual",
    "type3": "Contextual",
    "type4": "Contextual",
    "type5": "Factual",
    "type6": "Factual",
    "type7": "Logical",
    "type8": "Logical",
    "normal": "Control",
}

# ==========================================
# 2. LLM-as-a-Judge Evaluator
# ==========================================
class VeriOCREvaluator:
    def evaluate_with_llm_judge(self, response_text: str, error_type: str, tp: str, q: str, trap_logic_trace: dict) -> dict:
        """Evaluates Deceptive Tasks (Types 1-8) to determine Verification and Diagnosis Accuracy."""
        if not response_text or len(response_text.strip()) < 2:
            return {"judge_rationale": "Empty response", "is_rejected": False, "is_rationale_correct": False}
        
        type_definition = ERROR_TYPE_DEFINITIONS.get(
            error_type.lower(), 
            "Unknown Error Type: Evaluate based purely on the Trap Logic Trace."
        )

        system_prompt = """You are an expert evaluator. Your task is to analyze an AI model's response to a visually grounded question based on a potentially flawed premise. You must evaluate two critical dimensions:
1. Did the model detect the flawed task, either by explicitly refusing to answer as stated or by clearly identifying/correcting the invalid premise before proceeding?
2. If it detected the flaw, did it correctly identify the specific flaw according to the 'Trap Logic Trace'?"""

        user_prompt = f"""[Ground Truth & Trap Info]
- Error Category: {error_type} ({type_definition})
- Trap Logic Trace (The exact mechanism of the trap): 
{json.dumps(trap_logic_trace, ensure_ascii=False, indent=2)}
- Trap Premise (TP) given to the model: {tp}
- Question (Q) given to the model: {q}

[Tested Model's Response]
{response_text}

[Evaluation Task]
Analyze the model's response step-by-step. Output a strictly formatted JSON object with the following three fields IN THIS EXACT ORDER:

1. "judge_rationale" (string): FIRST, write a brief step-by-step analysis. Did the model detect that the task, premise, or question is invalid? Detection includes explicit refusal, explicit critique of the premise, or correction of the flawed premise before proceeding. Then compare the model's reasoning directly against the 'Trap Logic Trace'. Does it match?

2. "is_rejected" (boolean): `true` if the model detects the flawed task, including explicit refusals, statements that the premise is false or invalid, or responses that clearly correct the flawed premise before proceeding. `false` if it blindly follows the flawed premise, produces an answer without flagging the issue, or only expresses uncertainty without identifying the task invalidity.

3. "is_rationale_correct" (boolean): `true` ONLY IF `is_rejected` is true AND the model's reasoning aligns with the Trap Logic Trace. Output `false` if the model detects a problem but gives the wrong reason, provides a generic refusal without identifying the correct flaw, hallucinates an unrelated rationale, or if `is_rejected` is false."""

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    temperature=0.0
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                tqdm.write(f"\n[Judge API Error (Deceptive Task)] Attempt {attempt+1}/3 failed: {str(e)}") 
                time.sleep((attempt + 1) * 3)
        
        return {"judge_rationale": "API failed after max retries", "is_rejected": False, "is_rationale_correct": False}

    def evaluate_normal_with_llm_judge(self, response_text: str, tp: str, q: str) -> dict:
        """Evaluates Control Tasks (Normal) to determine the Over-Refusal Rate (ORR)."""
        if not response_text or len(response_text.strip()) < 2:
            return {"judge_rationale": "Empty response", "is_rejected": False}

        system_prompt = """You are an expert evaluator. 
Your task is to determine if an AI model exhibited "Over-Refusal" (i.e., it refused to answer a completely normal, valid visually-grounded question)."""

        user_prompt = f"""[Context]
- Premise (TP): {tp}
- Question (Q): {q}
Note: The premise and question are factually correct and answerable.

[Tested Model's Response]
{response_text}

[Evaluation Task]
Analyze the model's response step-by-step. Output a strictly formatted JSON object with the following two fields IN THIS EXACT ORDER:
1. "judge_rationale" (string): FIRST, write a brief analysis. Did the model attempt to answer the question normally, or did it refuse/reject it?
2. "is_rejected" (boolean): `true` if the model refused to answer or stated it cannot calculate/answer. `false` if it attempted to provide an answer (even if mathematically/factually wrong, as long as it didn't refuse)."""

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    temperature=0.0
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                tqdm.write(f"\n[Judge API Error (Control Task)] Attempt {attempt+1}/3 failed: {str(e)}") 
                time.sleep((attempt + 1) * 3)
                
        return {"judge_rationale": "API failed after max retries", "is_rejected": False}

# ==========================================
# 3. Metric Tracking Engine
# ==========================================
def create_counters():
    return {
        "N_deceptive": 0, "proactive_verifications": 0, "proactive_correct_diagnosis": 0,
        "assisted_verifications": 0, "assisted_correct_diagnosis": 0,
        "N_control": 0, "control_proactive_refusals": 0, "control_assisted_refusals": 0
    }

def update_counters(counters: dict, is_deceptive: bool, pro_judge: dict, ast_judge: dict):
    if is_deceptive:
        counters["N_deceptive"] += 1
        if pro_judge.get("is_rejected", False):
            counters["proactive_verifications"] += 1
            if pro_judge.get("is_rationale_correct", False):
                counters["proactive_correct_diagnosis"] += 1
        if ast_judge.get("is_rejected", False):
            counters["assisted_verifications"] += 1
            if ast_judge.get("is_rationale_correct", False):
                counters["assisted_correct_diagnosis"] += 1
    else:
        counters["N_control"] += 1
        if pro_judge.get("is_rejected", False):
            counters["control_proactive_refusals"] += 1
        if ast_judge.get("is_rejected", False):
            counters["control_assisted_refusals"] += 1

def calc_metrics(counters: dict) -> dict:
    """Computes final metrics ensuring naming parity with the paper."""
    safe_div = lambda a, b: (a / b) if b > 0 else 0.0
    
    # Task Verification Rate (TVR)
    ptvr = safe_div(counters["proactive_verifications"], counters["N_deceptive"])
    atvr = safe_div(counters["assisted_verifications"], counters["N_deceptive"])
    
    # Diagnosis Accuracy (DA): end-to-end diagnostic success over all trap-injected samples.
    p_da = safe_div(counters["proactive_correct_diagnosis"], counters["N_deceptive"])
    a_da = safe_div(counters["assisted_correct_diagnosis"], counters["N_deceptive"])
    
    # Over-Refusal Rate (ORR) on Control Tasks
    p_orr = safe_div(counters["control_proactive_refusals"], counters["N_control"])
    a_orr = safe_div(counters["control_assisted_refusals"], counters["N_control"])
    
    return {
        "PTVR": f"{ptvr:.2%}", 
        "ATVR": f"{atvr:.2%}",
        "p-DA": f"{p_da:.2%}", 
        "a-DA": f"{a_da:.2%}",
        "p-ORR": f"{p_orr:.2%}", 
        "a-ORR": f"{a_orr:.2%}", 
        "Sample_Count": counters["N_deceptive"] + counters["N_control"],
        "Raw_Counters": counters
    }

# ==========================================
# 4. Multithreaded Processing
# ==========================================
def process_single_item(item, evaluator):
    error_type = item.get("error_type", "unknown_type")
    source_domain = item.get("original_info", {}).get("source_domain", "unknown_domain")
    dimension = TYPE_TO_DIMENSION.get(str(error_type).lower(), "Unknown")
    
    # Identify deceptive traps vs. control group
    is_deceptive = "normal" not in error_type.lower() and "control" not in error_type.lower()
    
    trap_logic_trace = item.get("trap_logic_trace", item.get("internal_verification", {}))
    proactive_resp = item["model_responses"]["proactive"]
    assisted_resp = item["model_responses"]["assisted"]
    
    if is_deceptive:
        pro_judge = evaluator.evaluate_with_llm_judge(proactive_resp, error_type, item["TP"], item["Q"], trap_logic_trace)
        ast_judge = evaluator.evaluate_with_llm_judge(assisted_resp, error_type, item["TP"], item["Q"], trap_logic_trace)
    else:
        pro_judge = evaluator.evaluate_normal_with_llm_judge(proactive_resp, item["TP"], item["Q"])
        ast_judge = evaluator.evaluate_normal_with_llm_judge(assisted_resp, item["TP"], item["Q"])

    item["evaluation"] = {"proactive_judge": pro_judge, "assisted_judge": ast_judge}
    return item, is_deceptive, error_type, source_domain, dimension, pro_judge, ast_judge

def run_evaluation_pipeline(args):
    if not API_KEY:
        print("WARNING: OPENAI_API_KEY environment variable is not set. The evaluation will fail.")

    input_file = args.input
    output_file = os.path.join(args.output_dir, f"evaluated_metrics_report_{args.model.replace('/', '_')}.json")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found {input_file}!")
        return

    evaluator = VeriOCREvaluator()
    evaluated_records = []
    
    global_counters = create_counters()
    type_counters = {}
    domain_counters = {}
    dimension_counters = {}

    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    print(f" Starting LLM-as-a-Judge Evaluation for {args.model} ({len(lines)} samples)")

    MAX_WORKERS = args.workers
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_item, json.loads(line), evaluator) for line in lines]
        
        for future in tqdm(as_completed(futures), total=len(lines), desc="Judging (Multi-threaded)"):
            try:
                item, is_deceptive, error_type, source_domain, dimension, pro_judge, ast_judge = future.result()
                
                if error_type not in type_counters: type_counters[error_type] = create_counters()
                if source_domain not in domain_counters: domain_counters[source_domain] = create_counters()
                if dimension not in dimension_counters: dimension_counters[dimension] = create_counters()

                update_counters(global_counters, is_deceptive, pro_judge, ast_judge)
                update_counters(type_counters[error_type], is_deceptive, pro_judge, ast_judge)
                update_counters(domain_counters[source_domain], is_deceptive, pro_judge, ast_judge)
                update_counters(dimension_counters[dimension], is_deceptive, pro_judge, ast_judge)

                evaluated_records.append(item)
                
            except Exception as exc:
                tqdm.write(f"[Error] Processing a sample failed: {exc}")

    # Compile Final Report
    final_metrics = {
        "global_metrics": calc_metrics(global_counters),
        "metrics_by_type": {k: calc_metrics(v) for k, v in type_counters.items()},
        "metrics_by_domain": {k: calc_metrics(v) for k, v in domain_counters.items()},
        "metrics_by_dimension": {k: calc_metrics(v) for k, v in dimension_counters.items()}
    }

    print("\n" + "="*50)
    print(f"  VeriOCRBench Global Metrics for {args.model} ")
    print("="*50)
    for k, v in final_metrics["global_metrics"].items():
        if k != "Raw_Counters": print(f"[{k}]: {v}")
    print("="*50)

    os.makedirs(args.output_dir, exist_ok=True)
    final_report = {
        "summary_metrics": final_metrics,
        "detailed_results": evaluated_records
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
        
    print(f"\nEvaluation report saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VeriOCRBench LLM-as-a-Judge Scoring Pipeline")
    parser.add_argument("--model", type=str, required=True, help="The model identifier being evaluated")
    parser.add_argument("--input", type=str, required=True, help="Path to the JSONL raw responses file")
    parser.add_argument("--output_dir", type=str, default="Evaluation_Results", help="Directory to save the metrics report")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent threads for API calls")
    
    args = parser.parse_args()
    run_evaluation_pipeline(args)
