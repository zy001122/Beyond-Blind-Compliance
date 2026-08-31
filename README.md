# Beyond Blind Compliance: Benchmarking Task Verification in OCR Reasoning

## Updates

- [2026/09] Initial code release.
- [2026/08] Accepted to EMNLP 2026 Main Conference.

## Contents

- [Introduction](#introduction)
- [Key Findings](#key-findings)
- [Data Construction](#data-construction)
- [Install](#install)
- [Run Code](#run-code)

## Introduction

Multimodal large language models (MLLMs) have achieved strong performance on OCR-centric document understanding and text-rich visual reasoning. However, existing evaluations largely assume that every task is valid and answerable.

We introduce **VeriOCRBench**, a benchmark for evaluating whether MLLMs can verify that an OCR-grounded reasoning task is executable before answering.

VeriOCRBench contains **1,800 human-verified samples**, including **1,600 trap-injected invalid tasks** across **8 trap types** and **4 verification dimensions** (Visual, Contextual, Factual, and Logical), together with **200 trap-free control tasks**.

We evaluate model behavior under proactive and assisted prompting using three metrics: **Task Verification Rate (TVR)**, **Diagnosis Accuracy (DA)**, and **Over-Refusal Rate (ORR)**.

## Key Findings

- MLLMs frequently follow invalid OCR-grounded tasks without identifying the underlying problem.
- Explicit verification prompts improve task verification, but can also increase over-refusal on valid tasks.
- Detecting an invalid task does not necessarily imply correctly diagnosing its source of invalidity.

## Data Construction

VeriOCRBench uses a Visual Atomic Fact (VAF)-grounded construction pipeline based on **PP-OCRv5**.

| Dimension | Trap Types |
| --- | --- |
| Visual | Image Degradation, Occluded Target |
| Contextual | Domain Mismatch, Hallucinatory Target |
| Factual | Fact Tampering, Vague Reference |
| Logical | Logical Paradox, Missing Variable |
| Control | Trap-Free Normal Task |

Reference construction scripts are provided in `data_generation/`. The corresponding prompt templates are collected in `paper_prompt_templates.py`.

## Install

```bash
pip install -r requirements.txt
```

Set the API key and base URL required by your API provider:

```bash
export API_KEY="YOUR_API_KEY"
export BASE_URL="YOUR_BASE_URL"
```

## Run Code

### Multimodal Inference

Run proactive and assisted inference:

```bash
python inference_evaluation/inference.py \
  --model <model_name> \
  --input <dataset_json> \
  --output_dir <output_dir>
```

If image paths in the dataset are relative, use `--image_root` to specify the image root directory.

### LLM-as-a-Judge Evaluation

Evaluate model responses and compute TVR, DA, and ORR:

```bash
python inference_evaluation/eval_llm_judge.py \
  --model <model_name> \
  --input <raw_responses_jsonl> \
  --output_dir <output_dir>
```

### OCR+LLM Baseline

First extract OCR transcripts with PP-OCRv5:

```bash
python ocr_text_baseline/run_ppocrv5_extract.py \
  --dataset <dataset_json> \
  --output <ocr_output_jsonl> \
  --lang <ocr_language>
```

Then run the text-only verifier:

```bash
python ocr_text_baseline/run_text_verifier.py \
  --dataset <dataset_json> \
  --ocr-jsonl <ocr_output_jsonl> \
  --output <response_output_jsonl> \
  --model <model_name>
```

Audit a JSONL output file with:

```bash
python ocr_text_baseline/audit_jsonl.py \
  --path <jsonl_file>
```

### Data Construction

Reference scripts for the control group and Types 1--8 are provided in:

```text
data_generation/
```

Input paths and generation settings can be adjusted according to the local data setup.
