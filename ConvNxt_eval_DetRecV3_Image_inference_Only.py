import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset

from backbone.convNextV2Block import convnextv2_nano
from model.dbnetRec_convnextv2 import ConvNeXtV2_BiLSTM_CTC
from model.dbnetDet_convnextv2 import Nano_detection_model
from utils.ConvNextOCR_helper_EvalV2 import build_transform, db_postprocess, rescale_boxes, prepare_recognition_tensor


IMG_DIR = Path(r"D:\hvs\Hyvsion_Projects\OCR\dataset\projects\Dot_matrix_crop_251120\test")

REC_CHECKPOINT = Path(r"D:\hvs\Hyvsion_Projects\OCR\trainedModel\Text_Det_Rec\run_17\Rec\best_E2000_L0.000033_Rec.pth")
DET_CHECKPOINT = Path(r"D:\hvs\Hyvsion_Projects\OCR\trainedModel\Text_Det_Rec\run_17\Det\best_E347_L0.225306_Det.pth")

OUTPUT_CSV = Path(r"D:\hvs\Hyvsion_Projects\OCR\trainedModel\Text_Det_Rec\run_17\ImageOnly_inference_results.csv")

VISUALS_DIR = Path(r"D:\hvs\Hyvsion_Projects\OCR\trainedModel\Text_Det_Rec\run_17\ImageOnly_inference_visuals")

ROTATE_LEFT = False
ROTATE_RIGHT = False

TEXT_BOXDETECTION_INFERENCE = False  # True = detect + recognize, False = recognize-only

TEXT_IMG_H = 96
TEXT_IMG_W = 96
REC_PAD_PRESERVE_ASPECT = False
DET_IMG_H = 640
DET_IMG_W = 640

DET_INPUT_SIZE = (DET_IMG_H, DET_IMG_W)
DET_BIN_THRESH = 0.3
DET_BOX_THRESH = 0.5
DET_UNCLIP_RATIO = 2.0
DET_MAX_CANDIDATES = 1000
CROP_MARGIN = 0.0

VISUALS_MAX: Optional[int] = None
SAVE_VISUALS = True


class ImageFolderDataset(Dataset):
    """Simple dataset over all images in IMG_DIR."""

    def __init__(self, root: Path, rgb: bool = True,
                 exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")):
        if not root.exists():
            raise FileNotFoundError(f"Input folder not found: {root}")
        self.rgb = rgb
        exts_lower = {e.lower() for e in exts}
        self.paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts_lower)
        if not self.paths:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Path]:
        img_path = self.paths[idx]
        pil = Image.open(img_path).convert("RGB" if self.rgb else "L")
        return pil, img_path


def tensor_norm_to_rgb_uint8(img_t: torch.Tensor) -> np.ndarray:
    t = img_t.detach().cpu()
    if t.ndim == 4:
        t = t[0]
    if t.ndim != 3:
        raise ValueError(f"Expected (C,H,W) tensor, got {tuple(t.shape)}")

    c, h, w = t.shape
    arr = t.numpy().transpose(1, 2, 0)  # HWC
    arr = (arr * 0.5 + 0.5) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr


class CTCLabelConverter:
    def __init__(self, characters: str, ignore_case: bool = True):
        if ignore_case:
            characters = characters.lower()
        self.ignore_case = ignore_case
        self.characters = ["<blank>"] + list(dict.fromkeys(characters))
        self.blank_idx = 0
        self.char_to_idx = {char: idx for idx, char in enumerate(self.characters)}

    def decode(self, preds: torch.Tensor, pred_lengths: torch.Tensor) -> List[str]:
        results: List[str] = []
        for seq, length in zip(preds, pred_lengths):
            prev = self.blank_idx
            string = []
            for idx in seq[:length]:
                idx_i = int(idx)
                if idx_i != self.blank_idx and idx_i != prev:
                    string.append(self.characters[idx_i])
                prev = idx_i
            results.append("".join(string))
        return results


def levenshtein_distance(src: str, tgt: str) -> int:
    if src == tgt:
        return 0
    if not src:
        return len(tgt)
    if not tgt:
        return len(src)
    prev_row = list(range(len(tgt) + 1))
    for i, sc in enumerate(src, start=1):
        curr_row = [i]
        for j, tc in enumerate(tgt, start=1):
            insert_cost = curr_row[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (sc != tc)
            curr_row.append(min(insert_cost, delete_cost, replace_cost))
        prev_row = curr_row
    return prev_row[-1]


def polygon_to_bbox(poly: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    if poly is None or len(poly) == 0:
        return None
    poly = np.asarray(poly)
    if poly.ndim != 2 or poly.shape[1] < 2:
        return None
    xs = poly[:, 0]
    ys = poly[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def bbox_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def match_detections_to_gt(
    pred_polys: List[np.ndarray],
    gt_polys: List[np.ndarray],
    iou_threshold: float
) -> List[Tuple[int, int]]:
    matches: List[Tuple[int, int]] = []
    used_pred = set()
    for gt_idx, gt_poly in enumerate(gt_polys):
        gt_box = polygon_to_bbox(gt_poly)
        if gt_box is None:
            continue
        best_iou = 0.0
        best_pred = -1
        for pred_idx, pred_poly in enumerate(pred_polys):
            if pred_idx in used_pred:
                continue
            pred_box = polygon_to_bbox(pred_poly)
            if pred_box is None:
                continue
            iou = bbox_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_pred = pred_idx
        if best_pred >= 0 and best_iou >= iou_threshold:
            matches.append((gt_idx, best_pred))
            used_pred.add(best_pred)
    return matches


def load_detection_model(weight_path: Path, device: torch.device) -> Nano_detection_model:
    model = Nano_detection_model()
    checkpoint = torch.load(weight_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def load_recognition_model(weight_path: Path, device: torch.device) -> Tuple[ConvNeXtV2_BiLSTM_CTC, CTCLabelConverter]:
    checkpoint = torch.load(weight_path, map_location="cpu")
    ckpt_chars = checkpoint.get("converter")
    if ckpt_chars is None:
        raise RuntimeError("Recognition checkpoint missing 'converter'")
    converter = CTCLabelConverter("".join(ckpt_chars[1:]), ignore_case=False)
    converter.characters = ckpt_chars
    converter.char_to_idx = {char: idx for idx, char in enumerate(converter.characters)}

    backbone = convnextv2_nano()
    model = ConvNeXtV2_BiLSTM_CTC(backbone=backbone, in_channels_c5=320, vocab_size=len(converter.characters))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model, converter


def run_detection_inference(pil_img: Image.Image, det_model: torch.nn.Module, det_transform,
                            device: torch.device) -> List[Tuple[np.ndarray, float]]:
    det_tensor = det_transform(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = det_model(det_tensor)

    binary_t = None
    if isinstance(pred, dict):
        shrink_t = pred.get("shrink")
        thresh_t = pred.get("thresh")
        if shrink_t is None or thresh_t is None:
            return []
        shrink_t = shrink_t[0, 0]
        thresh_t = thresh_t[0, 0]
        if "binary" in pred:
            binary_t = pred["binary"][0, 0]
        else:
            head = getattr(det_model, "head", None)
            if head is None or not hasattr(head, "step_function"):
                return []
            binary_t = head.step_function(shrink_t, thresh_t)
    elif isinstance(pred, torch.Tensor):
        if pred.ndim == 4:
            binary_t = pred[0, 0]
        elif pred.ndim == 3:
            binary_t = pred[0]
        else:
            return []
    else:
        return []

    prob_map = binary_t.detach().float().cpu().numpy()
    detections = db_postprocess(prob_map, bin_thresh=float(DET_BIN_THRESH), box_thresh=float(DET_BOX_THRESH),
                                max_candidates=int(DET_MAX_CANDIDATES), unclip_ratio=float(DET_UNCLIP_RATIO))
    if not detections:
        return []

    det_h, det_w = DET_INPUT_SIZE
    scale_x = pil_img.width / float(det_w)
    scale_y = pil_img.height / float(det_h)
    detections = rescale_boxes(detections, scale_x=scale_x, scale_y=scale_y)

    outputs: List[Tuple[np.ndarray, float]] = []
    for poly, score in detections:
        if poly is None:
            continue
        poly = np.asarray(poly, dtype=np.float32)
        if poly.ndim != 2 or poly.shape[0] == 0:
            continue
        outputs.append((poly, float(score)))
    return outputs


def crop_polygon_from_image(pil_img: Image.Image, polygon: np.ndarray, margin: float) -> Optional[Image.Image]:
    xs = polygon[:, 0]
    ys = polygon[:, 1]
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max())
    y2 = float(ys.max())
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    mx = bw * margin
    my = bh * margin
    x1 = max(0.0, x1 - mx)
    y1 = max(0.0, y1 - my)
    x2 = min(float(pil_img.width), x2 + mx)
    y2 = min(float(pil_img.height), y2 + my)
    if x2 <= x1 or y2 <= y1:
        return None
    return pil_img.crop((int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))))


def draw_detections(pil_img: Image.Image, polygons: List[np.ndarray], texts: Optional[List[str]] = None) -> np.ndarray:
    vis = np.array(pil_img.convert("RGB"))
    if vis.ndim != 3 or vis.size == 0:
        return vis
    vis = np.ascontiguousarray(vis)
    label_iter = texts if texts is not None else []
    for idx, poly in enumerate(polygons):
        if poly is None or len(poly) == 0:
            continue
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 200, 0), thickness=2)
        if texts is not None and idx < len(label_iter):
            text = label_iter[idx] or ""
            if text:
                arr = np.asarray(poly, dtype=np.float32)
                cx = int(arr[:, 0].mean())
                cy = int(arr[:, 1].mean())
                cv2.putText(vis, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return vis


# --------- NEW: unpack + build GT list with ids ----------
def _safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def unpack_dataset_sample(sample: Any) -> Tuple[torch.Tensor, List[str], List[Any], List[Any], List[Optional[int]], str]:
    """
    Expected you will create dataset with:
      load_label=True, load_box_polygons=True, return_original_polygons=True, return_pair_ids=True, return_filename=True

    Typical output becomes:
      (img, labels, polys_resized, polys_orig, pair_ids, filename)

    This function is robust even if you add/remove fields later.
    """
    if not isinstance(sample, (tuple, list)) or len(sample) < 4:
        raise RuntimeError(f"Unexpected dataset sample type/len: {type(sample)} / {len(sample) if hasattr(sample,'__len__') else 'NA'}")

    img = sample[0]
    labels = sample[1] if len(sample) > 1 else []
    polys_resized = sample[2] if len(sample) > 2 else []
    # next could be orig polys OR pair_ids depending on flags
    filename = ""
    pair_ids: List[Optional[int]] = []
    polys_orig = []

    # filename usually last and string
    if isinstance(sample[-1], str):
        filename = sample[-1]

    # Try find pair_ids: it’s a list of ints/None typically
    # In your current dataloader order it comes after orig_polys when return_original_polygons=True.
    # So: [img, labels, scaled_polys, orig_polys, pair_ids, filename]
    # We'll detect it by looking for a list whose elements are int/None.
    candidate_lists = [x for x in sample[1:] if isinstance(x, list)]
    pid_list = None
    for x in candidate_lists:
        if len(x) == 0:
            continue
        ok = True
        for v in x:
            if v is None:
                continue
            if isinstance(v, (int, np.integer)):
                continue
            # allow strings that can be cast to int
            if isinstance(v, str) and v.strip().lstrip("-").isdigit():
                continue
            ok = False
            break
        if ok:
            pid_list = x
            break

    if pid_list is not None:
        pair_ids = [_safe_int(v) for v in pid_list]

    # orig polys: prefer the field right before pair_ids if we have both,
    # else just use polys_resized as fallback
    if len(sample) >= 5 and isinstance(sample[3], list):
        polys_orig = sample[3]
    else:
        polys_orig = polys_resized

    labels_list = labels if isinstance(labels, list) else [labels]
    return img, [str(x) for x in labels_list], polys_resized, polys_orig, pair_ids, filename


def build_gt_items(
    labels: List[str],
    orig_polys: List[Any],
    pair_ids: List[Optional[int]],
    ids_to_eval: Optional[set]
) -> List[Dict[str, Any]]:
    """
    Returns list of dicts: [{"id": int, "label": str, "poly": np.ndarray}, ...]
    Sorted by id (stable).
    """
    items: List[Dict[str, Any]] = []
    n = min(len(labels), len(orig_polys), len(pair_ids) if pair_ids else 10**9)

    # if pair_ids is missing/empty, fallback to index ids
    if not pair_ids or len(pair_ids) < n:
        pair_ids = [i for i in range(len(labels))]

    n = min(len(labels), len(orig_polys), len(pair_ids))

    for i in range(n):
        pid = _safe_int(pair_ids[i])
        if pid is None:
            pid = i
        if ids_to_eval is not None and pid not in ids_to_eval:
            continue
        poly = orig_polys[i]
        if poly is None or len(poly) == 0:
            continue
        arr = np.asarray(poly, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        items.append({"id": pid, "label": labels[i] or "", "poly": arr})

    items.sort(key=lambda d: d["id"])
    return items


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    rec_model, converter = load_recognition_model(REC_CHECKPOINT, device)

    dataset = ImageFolderDataset(IMG_DIR, rgb=True)

    if SAVE_VISUALS:
        VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[List[str]] = []
    visual_count = 0

    if TEXT_BOXDETECTION_INFERENCE:
        det_model = load_detection_model(DET_CHECKPOINT, device)
        det_h, det_w = DET_INPUT_SIZE
        det_transform = build_transform(det_h, det_w, rgb=True)

        for idx in tqdm(range(len(dataset)), desc="Detect+Rec", leave=False):
            pil_img, img_path = dataset[idx]
            if ROTATE_LEFT:
                pil_img = pil_img.transpose(Image.ROTATE_90)
            elif ROTATE_RIGHT:
                pil_img = pil_img.transpose(Image.ROTATE_270)

            detections = run_detection_inference(pil_img, det_model, det_transform, device)
            if not detections:
                rows.append([img_path.name, "-1", "0.0", ""])
                continue

            rec_tensors: List[torch.Tensor] = []
            meta: List[Tuple[int, float, np.ndarray]] = []
            detection_texts: Dict[int, str] = {}
            for det_idx, (poly, score) in enumerate(detections):
                crop = crop_polygon_from_image(pil_img, poly, CROP_MARGIN)
                if crop is None:
                    continue
                tensor = prepare_recognition_tensor(
                    crop.convert("RGB"),
                    TEXT_IMG_H,
                    TEXT_IMG_W,
                    preserve_aspect=REC_PAD_PRESERVE_ASPECT,
                )
                rec_tensors.append(tensor)
                meta.append((det_idx, score, poly))

            if not rec_tensors:
                rows.append([img_path.name, "-1", "0.0", ""])
                continue

            batch = torch.stack(rec_tensors).to(device)
            with torch.no_grad():
                logits, logit_lengths = rec_model(batch)
                log_probs = logits.log_softmax(dim=-1)
                pred_indices = log_probs.detach().argmax(dim=-1).permute(1, 0)
                decoded = converter.decode(pred_indices.cpu(), logit_lengths.cpu())

            for (det_idx, score, poly), text in zip(meta, decoded):
                rows.append([img_path.name, str(det_idx), f"{score:.4f}", text])
                detection_texts[det_idx] = text

            if SAVE_VISUALS and (VISUALS_MAX is None or visual_count < VISUALS_MAX):
                polygons = []
                texts = []
                for det_idx, (poly, _) in enumerate(detections):
                    polygons.append(poly)
                    texts.append(detection_texts.get(det_idx, ""))
                vis = draw_detections(pil_img, polygons, texts)
                out_path = VISUALS_DIR / f"{img_path.stem}_det.png"
                if cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)):
                    visual_count += 1
    else:
        BATCH_SIZE = 64
        rec_tensors: List[torch.Tensor] = []
        paths: List[Path] = []

        def flush_batch():
            nonlocal rec_tensors, paths, visual_count
            if not rec_tensors:
                return

            batch = torch.stack(rec_tensors).to(device)
            with torch.no_grad():
                logits, logit_lengths = rec_model(batch)
                log_probs = logits.log_softmax(dim=-1)
                pred_indices = log_probs.detach().argmax(dim=-1).permute(1, 0)
                decoded = converter.decode(pred_indices.cpu(), logit_lengths.cpu())

            for path, text, tensor in zip(paths, decoded, rec_tensors):
                rows.append([path.name, text])
                if SAVE_VISUALS and (VISUALS_MAX is None or visual_count < VISUALS_MAX):
                    rgb = tensor_norm_to_rgb_uint8(tensor)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    if text:
                        cv2.putText(bgr, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                    out_path = VISUALS_DIR / f"{path.stem}_{visual_count:05d}.png"
                    if cv2.imwrite(str(out_path), bgr):
                        visual_count += 1

            rec_tensors = []
            paths = []

        for idx in tqdm(range(len(dataset)), desc="Rec-only", leave=False):
            pil_img, img_path = dataset[idx]
            if ROTATE_LEFT:
                pil_img = pil_img.transpose(Image.ROTATE_90)
            elif ROTATE_RIGHT:
                pil_img = pil_img.transpose(Image.ROTATE_270)

            tensor = prepare_recognition_tensor(
                pil_img.convert("RGB"),
                TEXT_IMG_H,
                TEXT_IMG_W,
                preserve_aspect=REC_PAD_PRESERVE_ASPECT,
            )
            rec_tensors.append(tensor)
            paths.append(img_path)

            if len(rec_tensors) >= BATCH_SIZE:
                flush_batch()

        flush_batch()

    header = ["filename", "prediction"] if not TEXT_BOXDETECTION_INFERENCE else ["filename", "detection_id", "score", "prediction"]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Saved {len(rows)} predictions to {OUTPUT_CSV}")
    if SAVE_VISUALS:
        print(f"Saved {visual_count} visual previews to {VISUALS_DIR}")


if __name__ == "__main__":
    evaluate()