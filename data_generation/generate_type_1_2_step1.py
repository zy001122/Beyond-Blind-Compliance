import os
import json
import random
import logging
from PIL import Image, ImageDraw, ImageFilter
from ocr_utils import extract_ocr_facts_from_image, init_paddleocr

# Handle Pillow compatibility for modern versions
try:
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_NEAREST = Image.NEAREST

# Suppress verbose PaddleOCR logs
logging.getLogger("ppocr").setLevel(logging.WARNING)

# ==========================================
# 1. Visual Corruption Engine (Aligned with VeriOCRBench Taxonomy)
# ==========================================
class VisualSpoiler:
    def apply_damage(self, img, bbox, task_type):
        w, h = img.size
        # Force standard Python int
        xmin, ymin, xmax, ymax = [int(v) for v in bbox]
        box = (max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax))
        
        img_damaged = img.copy()
        strategy = ""
        desc = ""

        # --- Type 1: Image Degradation (Global) ---
        if task_type == "type1_degradation":
            strategy = random.choice(["heavy_blur", "extreme_pixelate"])
            
            if strategy == "heavy_blur":
                radius = max(5, min(w, h) // 40) 
                img_damaged = img_damaged.filter(ImageFilter.GaussianBlur(radius=radius))
                desc = "entire image heavily blurred"
            
            elif strategy == "extreme_pixelate":
                scale_factor = 10  
                # Prevent 0-dimension error for extremely small images
                low_w = max(1, w // scale_factor)
                low_h = max(1, h // scale_factor)
                low_res = img_damaged.resize((low_w, low_h), RESAMPLE_NEAREST)
                img_damaged = low_res.resize((w, h), RESAMPLE_NEAREST)
                desc = "entire image highly pixelated"

        # --- Type 2: Occluded Target (Local Spatial Masking) ---
        elif task_type == "type2_occlusion":
            strategy = random.choice(["black_box", "tape_cover"])
            draw = ImageDraw.Draw(img_damaged)
            
            if strategy == "black_box":
                draw.rectangle(box, fill="black")
                desc = "target text strictly redacted with a solid black box"
                
            elif strategy == "tape_cover":
                pad = 4
                draw.rectangle([max(0, box[0]-pad), max(0, box[1]-pad), min(w, box[2]+pad), min(h, box[3]+pad)], fill=(240, 235, 225))
                desc = "target text covered by an opaque tape/patch"

        return img_damaged, strategy, desc

# ==========================================
# 2. Pipeline: Generate-then-Corrupt
# ==========================================
class PipelineStep1:
    def __init__(self):
        ocr_version = os.environ.get("PADDLEOCR_VERSION", "PP-OCRv5")
        ocr_lang = os.environ.get("PADDLEOCR_LANG", "ch")
        print(f"Loading PaddleOCR ({ocr_version}, lang={ocr_lang})...")
        self.ocr = init_paddleocr(lang=ocr_lang, ocr_version=ocr_version)
        self.spoiler = VisualSpoiler()
        print("PaddleOCR successfully loaded.")

    def process(self, json_path, output_root, task_type):
        if not os.path.exists(json_path):
            print(f"Input file not found: {json_path}")
            return

        with open(json_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        save_dir = os.path.join(output_root, f"{task_type.capitalize()}")
        debug_dir = os.path.join(output_root, f"Debug_Check_{task_type.split('_')[0]}")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(debug_dir, exist_ok=True)
        
        processed_records = []
        print(f"Processing {task_type}, total images: {len(data_list)}...")

        for idx, item in enumerate(data_list):
            try:
                src_path = item['image_path']
                if not os.path.exists(src_path): 
                    print(f"  [Skip] Image not found: {src_path}")
                    continue
                
                img = Image.open(src_path).convert("RGB")
                
                # Filter high-confidence text anchors
                candidates = []
                for fact in extract_ocr_facts_from_image(self.ocr, src_path, min_conf=0.80):
                    text_content = fact["content"]
                    bbox = fact["bbox"]
                    if bbox and len(text_content.strip()) >= 2:
                        candidates.append((bbox, text_content))
                
                if not candidates:
                    continue

                # Sample up to 3 viable candidates for the LLM agent to choose from
                sampled_candidates = random.sample(candidates, min(3, len(candidates)))
                
                # Metadata assembly
                orig_info = item.get("original_info", {})
                if not orig_info:
                    orig_info = {
                        "image_path": item.get("image_path", src_path),
                        "source_domain": item.get("source_domain", "unknown_domain"),
                        "target_error_type": item.get("target_error_type", task_type)
                    }
                
                record = {
                    "original_info": orig_info,
                    "original_image": src_path,
                    "task_type": task_type,
                    "candidates": [] 
                }

                filename = os.path.basename(src_path)
                name, ext = os.path.splitext(filename)

                # --- Type 1: Image Degradation (Global) ---
                if task_type == "type1_degradation":
                    # Global corruption is applied once per image
                    bad_img, strategy, desc = self.spoiler.apply_damage(img, sampled_candidates[0][0], task_type)
                    save_path = os.path.join(save_dir, f"corrupted_{strategy}_{filename}")
                    bad_img.save(save_path)
                    
                    for i, (bbox, text) in enumerate(sampled_candidates):
                        record["candidates"].append({
                            "candidate_id": f"cand_{i+1}",
                            "ground_truth_hidden": text,
                            "target_bbox": bbox,
                            "corrupted_image": save_path,
                            "visual_status": desc
                        })

                # --- Type 2: Occluded Target (Local) ---
                elif task_type == "type2_occlusion":
                    debug_img = img.copy()
                    draw_debug = ImageDraw.Draw(debug_img)
                    colors = ["red", "blue", "green"] 
                    
                    for i, (bbox, text) in enumerate(sampled_candidates):
                        # Apply masking to this specific entity only
                        bad_img, strategy, desc = self.spoiler.apply_damage(img, bbox, task_type)
                        save_path = os.path.join(save_dir, f"corrupted_{strategy}_cand{i+1}_{name}{ext}")
                        bad_img.save(save_path)
                        
                        record["candidates"].append({
                            "candidate_id": f"cand_{i+1}",
                            "ground_truth_hidden": text,
                            "target_bbox": bbox,
                            "corrupted_image": save_path,
                            "visual_status": desc
                        })
                        
                        # Draw debug bounding boxes
                        color = colors[i % len(colors)]
                        draw_debug.rectangle(bbox, outline=color, width=3)
                        draw_debug.text((bbox[0], max(0, bbox[1]-15)), f"Cand {i+1}", fill=color)

                    debug_path = os.path.join(debug_dir, f"debug_{filename}")
                    debug_img.save(debug_path)

                processed_records.append(record)
                
                if (idx + 1) % 5 == 0:
                    print(f"  Processed {idx + 1}/{len(data_list)}")

            except Exception as e:
                print(f"  [Error] Failed on {src_path}: {e}")

        out_json = os.path.join(output_root, f"Step1_Output_{task_type}.json")
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(processed_records, f, indent=2, ensure_ascii=False)
        print(f"Completed! Results saved to: {out_json}")

if __name__ == "__main__":
    runner = PipelineStep1()
    
    # WARNING: Ensure your input JSON paths match the paper-facing taxonomy names.
    runner.process("experiment_samples/Type_1_Degradation.json", "VeriOCR_Synthetic_Data", "type1_degradation")
    runner.process("experiment_samples/Type_2_Occlusion.json", "VeriOCR_Synthetic_Data", "type2_occlusion")
