import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm.auto import tqdm

from utils.DetRecDataloaderV2 import BoxTextDataset
from utils.ConvNextOCR_helper_EvalV2 import *


IMG_DIR = Path(r"D:\hvs\Hyvsion_Projects\OCR\dataset\public\IIIT5k_dataset\train")
LABEL_PATH = Path(r"D:\hvs\Hyvsion_Projects\OCR\dataset\public\IIIT5k_dataset\traindata.json")

REC_CHECKPOINT = Path(r"D:\hvs\Hyvsion_Projects\OCR\trainedModel\IIIT5K_PreTrain\run_10\Rec\best_E036_L1.869584_Rec.pth")
DET_CHECKPOINT = Path(r"D:\hvs\Hyvsion_Projects\OCR\trainedModel\IIIT5K_PreTrain\run_10\Det\best_E326_L0.220883_Det.pth")

OUTPUT_CSV = Path(r"D:\hvs\Hyvsion_Projects\OCR\trainedModel\IIIT5K_PreTrain\run_10\eval_rec_results.csv")

VISUALS_DIR = Path(r"D:\hvs\Hyvsion_Projects\OCR\trainedModel\IIIT5K_PreTrain\run_10\eval_rec_visuals")
DET_BOX_VIS_DIR = Path(r"D:\hvs\Hyvsion_Projects\OCR\trainedModel\IIIT5K_PreTrain\run_10\detected_boxes")

ROTATE_LEFT = False
ROTATE_RIGHT = False

TEXT_BOXDETECTION_INFERENCE = False   # True = Det(+GT polygons)/crop + Rec, False = Rec-only


TEXT_IMG_H = 96
TEXT_IMG_W = 256
REC_PAD_PRESERVE_ASPECT = True
Det_IMG_H = 640
Det_IMG_W = 640

USE_GT_POLYGONS_For_Cropping = False # True = use GT polygons to crop text regions for recognition; False = use detected boxes
DET_INPUT_SIZE = (Det_IMG_H, Det_IMG_W)  # (H, W)
DET_BIN_THRESH = 0.3
DET_BOX_THRESH = 0.5
DET_UNCLIP_RATIO = 2.0
DET_MAX_CANDIDATES = 1000
MATCH_IOU_THRESH = 0.3
CROP_MARGIN = 0.00

VISUALS_MAX = None
SHOW_DET_BOXES = True
SAVE_VISUALS = True

# ✅ NEW: Evaluate only specific pair ids (None = evaluate all)
# Example: IDS_TO_EVAL = {0}  # only id 0
IDS_TO_EVAL = None  # or set like {0,1,2}






















def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Always need REC
    rec_model, converter = load_recognition_model(REC_CHECKPOINT, device)

    # Only load DET if we are doing box detection inference
    det_model = None
    det_transform = None
    if TEXT_BOXDETECTION_INFERENCE:
        det_model = load_detection_model(DET_CHECKPOINT, device)
        det_h, det_w = DET_INPUT_SIZE
        det_transform = build_transform(det_h, det_w, rgb=True)

        if SAVE_VISUALS:
            VISUALS_DIR.mkdir(parents=True, exist_ok=True)
        if SHOW_DET_BOXES and not USE_GT_POLYGONS_For_Cropping:
            DET_BOX_VIS_DIR.mkdir(parents=True, exist_ok=True)

        dataset = BoxTextDataset(
            img_dir=str(IMG_DIR),
            json_path=str(LABEL_PATH),
            imgH=det_h,
            imgW=det_w,
            rgb=True,
            load_label=True,
            load_box_polygons=True,
            return_filename=True,
            rotate_left=ROTATE_LEFT,
            rotate_right=ROTATE_RIGHT,
            return_original_polygons=True,
            return_pair_ids=True,
        )
    else:
        # --------- REC ONLY DATASET ---------
        # Important: set imgH/imgW to TEXT_IMG_H/TEXT_IMG_W for your cropped images case.
        if SAVE_VISUALS:
            VISUALS_DIR.mkdir(parents=True, exist_ok=True)

        base_dataset = BoxTextDataset(
            img_dir=str(IMG_DIR),
            json_path=str(LABEL_PATH),
            imgH=TEXT_IMG_H,
            imgW=TEXT_IMG_W,
            rgb=True,
            load_label=True,
            load_box_polygons=False,     # ✅ rec-only
            return_filename=True,
            rotate_left=ROTATE_LEFT,
            rotate_right=ROTATE_RIGHT,
            # no polygons / ids needed:
            return_original_polygons=False,
            return_pair_ids=False,
        )
        dataset = RecognitionEvalDataset(base_dataset)

    ids_set = set(IDS_TO_EVAL) if (IDS_TO_EVAL is not None and TEXT_BOXDETECTION_INFERENCE) else None
    if (IDS_TO_EVAL is not None) and (not TEXT_BOXDETECTION_INFERENCE):
        print("[Info] IDS_TO_EVAL is ignored in Rec-only mode (no pair_ids / polygons).")

    total_words = 0
    correct_words = 0
    char_accuracy_sum = 0.0
    rows: List[List[str]] = []
    visual_count = 0


    with torch.no_grad():
        if not TEXT_BOXDETECTION_INFERENCE:
            # ============================================
            # REC-ONLY INFERENCE PATH
            # ============================================
            rec_tensors: List[torch.Tensor] = []
            metas: List[Dict[str, Any]] = []

            B = 64  # batch size for eval

            def flush_batch():
                nonlocal total_words, correct_words, char_accuracy_sum, visual_count
                if not rec_tensors:
                    return

                batch = torch.stack(rec_tensors).to(device)
                logits, logit_lengths = rec_model(batch)
                log_probs = logits.log_softmax(dim=-1)
                pred_indices = log_probs.detach().argmax(dim=-1).permute(1, 0)
                decoded = converter.decode(pred_indices.cpu(), logit_lengths.cpu())

                for meta, pred_text in zip(metas, decoded):
                    gt_text = str(meta["gt"]).strip()
                    pred_text = str(pred_text).strip()

                    gt_cmp = gt_text.lower()
                    pred_cmp = pred_text.lower()

                    total_words += 1
                    match = (gt_cmp == pred_cmp)
                    if match:
                        correct_words += 1

                    dist = levenshtein_distance(pred_cmp, gt_cmp)
                    norm = max(len(gt_cmp), 1)
                    char_accuracy_sum += 1.0 - (dist / norm)

                    sample_id = meta["sid"]
                    rows.append([sample_id, gt_text, pred_text, match])

                    if SAVE_VISUALS and (VISUALS_MAX is None or visual_count < VISUALS_MAX):
                        # show crop with GT & Pred
                        rgb = tensor_norm_to_rgb_uint8(meta["img_tensor"])
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                        view = bgr
                        cv2.putText(view, f"GT:   {gt_text}", (8, view.shape[0] - 34),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 128, 0), 1, cv2.LINE_AA)
                        cv2.putText(view, f"Pred: {pred_text}", (8, view.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 1, cv2.LINE_AA)

                        out_name = f"{Path(sample_id).stem}_{visual_count:05d}.png"
                        out_path = VISUALS_DIR / out_name
                        if cv2.imwrite(str(out_path), view):
                            visual_count += 1

                rec_tensors.clear()
                metas.clear()

            for idx in tqdm(range(len(dataset)), desc="Evaluating (Rec-only)", leave=False):
                img_tensor, gt, sample_id = dataset[idx]

                rec_tensors.append(img_tensor)
                metas.append({"sid": sample_id, "gt": gt, "img_tensor": img_tensor})

                if len(rec_tensors) >= B:
                    flush_batch()

            flush_batch()

        else:
            # ============================================
            # YOUR EXISTING DET(+GT)/CROP + REC PATH
            # (kept the same, only wrapped under this branch)
            # ============================================
            det_h, det_w = DET_INPUT_SIZE

            for idx in tqdm(range(len(dataset)), desc="Evaluating", leave=False):
                sample = dataset[idx]
                img_tensor, labels, _, orig_polys, pair_ids, filename = unpack_dataset_sample(sample)

                gt_items = build_gt_items(labels, orig_polys, pair_ids, ids_set)
                if not gt_items:
                    continue

                image_path = IMG_DIR / filename
                pil_img = Image.open(image_path).convert("RGB")
                if ROTATE_LEFT:
                    pil_img = pil_img.transpose(Image.ROTATE_90)
                elif ROTATE_RIGHT:
                    pil_img = pil_img.transpose(Image.ROTATE_270)

                gt_polys = [it["poly"] for it in gt_items]

                pred_polys: List[np.ndarray] = []
                matches: List[Tuple[int, int]] = []
                matched_pred_indices: set = set()

                if USE_GT_POLYGONS_For_Cropping:
                    pred_polys = [p for p in gt_polys]
                    matches = [(i, i) for i in range(len(gt_polys))]
                    matched_pred_indices = set(range(len(gt_polys)))
                else:
                    detections = run_detection_inference(pil_img, det_model, DET_BIN_THRESH=DET_BIN_THRESH, DET_BOX_THRESH=DET_BOX_THRESH, DET_MAX_CANDIDATES=DET_MAX_CANDIDATES, DET_UNCLIP_RATIO=DET_UNCLIP_RATIO, DET_INPUT_SIZE=DET_INPUT_SIZE, det_transform=det_transform, device=device)
                    pred_polys = [poly for poly, _ in detections]
                    matches = match_detections_to_gt(pred_polys, gt_polys, MATCH_IOU_THRESH)
                    matched_pred_indices = {pred_idx for _, pred_idx in matches}

                    if SHOW_DET_BOXES:
                        det_out_path = DET_BOX_VIS_DIR / f"{Path(filename).stem}.png"
                        try:
                            save_detection_visual(pil_img, gt_polys, pred_polys, matched_pred_indices, det_out_path)
                        except Exception as exc:
                            print(f"[WARN] Failed to save detection visual for {filename}: {exc}")

                rec_tensors: List[torch.Tensor] = []
                rec_meta: List[Dict[str, object]] = []
                pred_text_for_gtid: Dict[int, str] = {}

                for gt_idx, pred_idx in matches:
                    if gt_idx >= len(gt_items):
                        continue
                    gt_id = int(gt_items[gt_idx]["id"])
                    gt_label = str(gt_items[gt_idx]["label"])

                    poly = pred_polys[pred_idx] if pred_idx < len(pred_polys) else None
                    if poly is None or len(poly) == 0:
                        continue

                    crop = crop_polygon_from_image(pil_img, np.asarray(poly, dtype=np.float32), CROP_MARGIN)
                    if crop is None:
                        continue

                    crop_rgb = crop.convert("RGB")
                    crop_rgb_np = np.array(crop_rgb)
                    tensor = prepare_recognition_tensor(
                        crop_rgb,
                        TEXT_IMG_H,
                        TEXT_IMG_W,
                        preserve_aspect=REC_PAD_PRESERVE_ASPECT,
                    )
                    rec_tensors.append(tensor)

                    gt_crop_rgb = None
                    gt_poly = gt_items[gt_idx]["poly"]
                    gt_crop = crop_polygon_from_image(pil_img, np.asarray(gt_poly, dtype=np.float32), margin=0.0)
                    if gt_crop is not None:
                        gt_crop_rgb = gt_crop.convert("RGB")

                    rec_meta.append({
                        "gt_id": gt_id,
                        "label": gt_label,
                        "crop_rgb": crop_rgb_np,
                        "gt_crop_rgb": np.array(gt_crop_rgb) if gt_crop_rgb is not None else None,
                        "filename": filename,
                    })

                if rec_tensors:
                    batch = torch.stack(rec_tensors).to(device)
                    logits, logit_lengths = rec_model(batch)
                    log_probs = logits.log_softmax(dim=-1)
                    pred_indices = log_probs.detach().argmax(dim=-1).permute(1, 0)
                    decoded = converter.decode(pred_indices.cpu(), logit_lengths.cpu())

                    for meta, pred_text in zip(rec_meta, decoded):
                        gt_id = int(meta["gt_id"])
                        pred_text_for_gtid[gt_id] = pred_text

                        if SAVE_VISUALS and (VISUALS_MAX is None or visual_count < VISUALS_MAX):
                            crop_rgb = meta.get("crop_rgb")
                            if crop_rgb is not None:
                                crop_rgb = np.asarray(crop_rgb)
                                if crop_rgb.ndim == 2:
                                    crop_rgb = np.stack([crop_rgb] * 3, axis=-1)
                                if crop_rgb.ndim == 3 and crop_rgb.shape[2] == 3 and crop_rgb.size > 0:
                                    h, w = crop_rgb.shape[:2]
                                    text_pad = 40
                                    canvas = np.full((h + text_pad, w, 3), 255, dtype=np.uint8)
                                    canvas[:h] = crop_rgb
                                    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
                                    gt_text = str(meta.get("label", ""))
                                    cv2.putText(bgr, f"GT:   {gt_text}", (8, h + 18), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.55, (0, 128, 0), 1, cv2.LINE_AA)
                                    cv2.putText(bgr, f"Pred: {pred_text}", (8, h + 34), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.55, (255, 0, 0), 1, cv2.LINE_AA)

                                    out_name = f"{Path(meta.get('filename', 'sample')).stem}_id{gt_id}_{visual_count:05d}.png"
                                    out_path = VISUALS_DIR / out_name
                                    if cv2.imwrite(str(out_path), bgr):
                                        visual_count += 1

                for it in gt_items:
                    gt_id = int(it["id"])
                    gt_text = str(it["label"] or "")
                    pred_text = pred_text_for_gtid.get(gt_id, "")

                    gt_cmp = gt_text.lower()
                    pred_cmp = pred_text.lower()

                    total_words += 1
                    match = (pred_cmp == gt_cmp)
                    if match:
                        correct_words += 1

                    dist = levenshtein_distance(pred_cmp, gt_cmp)
                    norm = max(len(gt_cmp), 1)
                    char_accuracy_sum += 1.0 - (dist / norm)

                    rows.append([f"{filename}#id{gt_id}", gt_text, pred_text, match])

    word_acc = correct_words / total_words if total_words else 0.0
    char_acc = char_accuracy_sum / total_words if total_words else 0.0

    print(f"Samples evaluated : {total_words}")
    print(f"Word accuracy     : {word_acc * 100:.2f}%")
    print(f"Char accuracy     : {char_acc * 100:.2f}%")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "ground_truth", "prediction", "exact_match"])
        writer.writerows(rows)

    print(f"Saved detailed results to {OUTPUT_CSV}")
    if SAVE_VISUALS:
        print(f"Saved {visual_count} visuals to {VISUALS_DIR}")
        
if __name__ == "__main__":
    evaluate()