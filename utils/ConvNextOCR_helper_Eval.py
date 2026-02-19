# train_dbnet.py
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Any, Iterable, Set
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from utils.db_helper import build_db_maps
from utils.util import DBLoss
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from model.dbnetDet_convnextv2 import Nano_detection_model, DBNetConvNeXtV2
from model.dbnetRec_convnextv2 import ConvNeXtV2_BiLSTM_CTC
from backbone.convNextV2Block import convnextv2_nano, convnextv2_nano_dims
from torchvision import transforms
from PIL import Image

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    cv2 = None
    _cv2_error = exc
else:
    _cv2_error = None

try:
    import pyclipper
except ImportError as exc:  # pragma: no cover
    pyclipper = None
    _pyclipper_error = exc
else:
    _pyclipper_error = None

################# Eval###########################



def build_transform(img_h: int, img_w: int, rgb: bool) -> transforms.Compose:
    mean = (0.5, 0.5, 0.5) if rgb else (0.5,)
    std = (0.5, 0.5, 0.5) if rgb else (0.5,)
    return transforms.Compose([
        transforms.Resize((img_h, img_w), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _require_pyclipper():
    if pyclipper is None:
        raise RuntimeError("pyclipper is required for polygon expansion") from _pyclipper_error

def _polygon_area(poly: np.ndarray) -> float:
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def _polygon_perimeter(poly: np.ndarray) -> float:
    if poly.shape[0] < 2:
        return 0.0
    d = poly - np.roll(poly, -1, axis=0)
    return float(np.sqrt((d * d).sum(axis=1)).sum())

def _unclip(poly: np.ndarray, ratio: float) -> Optional[np.ndarray]:
    _require_pyclipper()
    area = _polygon_area(poly)
    peri = _polygon_perimeter(poly)
    if area < 1e-6 or peri < 1e-6:
        return None
    distance = area * ratio / (peri + 1e-6)
    if distance < 1.0:
        return None
    scale = 2.0
    pco = pyclipper.PyclipperOffset()
    path = [(int(x * scale), int(y * scale)) for x, y in poly.tolist()]
    pco.AddPath(path, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = pco.Execute(distance * scale)
    if not expanded:
        return None
    expanded = np.array(expanded[0], dtype=np.float32) / scale
    return expanded


def rescale_boxes(boxes: List[Tuple[np.ndarray, float]], scale_x: float, scale_y: float) -> List[Tuple[np.ndarray, float]]:
    scaled: List[Tuple[np.ndarray, float]] = []
    for box, score in boxes:
        box_scaled = box.copy()
        box_scaled[:, 0] *= scale_x
        box_scaled[:, 1] *= scale_y
        scaled.append((box_scaled, score))
    return scaled



def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def _require_opencv():
    if cv2 is None:
        raise RuntimeError("opencv-python is required for inference") from _cv2_error


def db_postprocess(prob_map: np.ndarray,
                   bin_thresh: float,
                   box_thresh: float,
                   max_candidates: int,
                   unclip_ratio: float) -> List[Tuple[np.ndarray, float]]:
    _require_opencv()
    h, w = prob_map.shape
    binary = (prob_map > bin_thresh).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    detections: List[Tuple[np.ndarray, float]] = []
    for contour in contours[:max_candidates]:
        if contour.shape[0] < 3:
            continue
        poly = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if poly.shape[0] < 4:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 1)
        score = float(cv2.mean(prob_map, mask=mask)[0])
        if score < box_thresh:
            continue
        poly_pts = poly.reshape(-1, 2).astype(np.float32)
        expanded = _unclip(poly_pts, unclip_ratio)
        if expanded is None or expanded.shape[0] < 4:
            continue
        rect = cv2.minAreaRect(expanded.astype(np.float32))
        box = cv2.boxPoints(rect)
        box = _order_points(box)
        detections.append((box, score))
    return detections


def tensor_norm_to_rgb_uint8(img_t: torch.Tensor) -> np.ndarray:
    """
    Convert normalized tensor (C,H,W) in roughly [-1,1] to RGB uint8 (H,W,3).
    Assumes Normalize(mean=0.5,std=0.5).
    """
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





class CTCLabelConverter_decode:
    def __init__(self, characters: str, ignore_case: bool = True):
        if ignore_case:
            characters = characters.lower()
        self.ignore_case = ignore_case
        self.characters = ["<blank>"] + list(dict.fromkeys(characters))
        self.blank_idx = 0
        self.char_to_idx = {char: idx for idx, char in enumerate(self.characters)}

    def decode(self, preds, pred_lengths):
        results = []
        for seq, length in zip(preds, pred_lengths):
            prev = self.blank_idx
            string = []
            for idx in seq[:length]:
                idx = int(idx)
                if idx != self.blank_idx and idx != prev:
                    string.append(self.characters[idx])
                prev = idx
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


def load_detection_model(weight_path: Path, device: torch.device) -> torch.nn.Module:
    model = Nano_detection_model()
    checkpoint = torch.load(weight_path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model

def load_recognition_model(weight_path: Path, device: torch.device) -> Tuple[torch.nn.Module, CTCLabelConverter_decode]:
    checkpoint = torch.load(weight_path, map_location="cpu")
    ckpt_chars = checkpoint.get("converter")
    if ckpt_chars is None:
        raise RuntimeError("Checkpoint missing 'converter' key with vocabulary.")
    converter = CTCLabelConverter_decode("".join(ckpt_chars[1:]), ignore_case=False)
    converter.characters = ckpt_chars
    converter.char_to_idx = {char: idx for idx, char in enumerate(converter.characters)}

    backbone = convnextv2_nano()
    model = ConvNeXtV2_BiLSTM_CTC(
        backbone=backbone,
        in_channels_c5=320,
        vocab_size=len(converter.characters),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model, converter



def run_detection_inference(
    pil_img: Image.Image,
    det_model: torch.nn.Module,
    DET_BIN_THRESH: float,
    DET_BOX_THRESH: float,
    DET_MAX_CANDIDATES: int,
    DET_UNCLIP_RATIO: float,
    DET_INPUT_SIZE: Tuple[int, int],
    det_transform,
    device: torch.device,
) -> List[Tuple[np.ndarray, float]]:
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
    detections = db_postprocess(
        prob_map,
        bin_thresh=float(DET_BIN_THRESH),
        box_thresh=float(DET_BOX_THRESH),
        max_candidates=int(DET_MAX_CANDIDATES),
        unclip_ratio=float(DET_UNCLIP_RATIO),
    )
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


def pad_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    if image.shape[1] >= target_width:
        return image
    diff = target_width - image.shape[1]
    left = diff // 2
    right = diff - left
    return cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def add_title_bar(image: np.ndarray, title: str, bar_height: int = 28) -> np.ndarray:
    h, w = image.shape[:2]
    canvas = np.full((h + bar_height, w, 3), 255, dtype=image.dtype)
    canvas[bar_height:] = image
    cv2.putText(canvas, title, (8, bar_height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def save_detection_visual(
    pil_img: Image.Image,
    gt_polys: List[np.ndarray],
    pred_polys: List[np.ndarray],
    matched_pred_indices: set,
    out_path: Path,
):
    vis = np.array(pil_img.convert("RGB"))
    if vis.ndim != 3 or vis.size == 0:
        return
    vis = np.ascontiguousarray(vis)
    for poly in gt_polys:
        if poly is None or len(poly) == 0:
            continue
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    for idx, poly in enumerate(pred_polys):
        if poly is None or len(poly) == 0:
            continue
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        color = (0, 200, 0) if idx in matched_pred_indices else (0, 180, 255)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    

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


class RecognitionEvalDataset(Dataset):
    """Flatten labels from BoxTextDataset for recognition-only evaluation."""

    def __init__(self, base_dataset: Dataset):
        self.samples: List[Tuple[torch.Tensor, str, str]] = []
        for idx in range(len(base_dataset)):
            sample = base_dataset[idx]
            if isinstance(sample, (list, tuple)):
                img_tensor = sample[0]
                labels = sample[1] if len(sample) > 1 else []
                filename = sample[-1] if isinstance(sample[-1], str) else f"idx{idx}"
            else:
                img_tensor = sample
                labels = []
                filename = f"idx{idx}"

            label_list = labels if isinstance(labels, list) else [labels]
            if not label_list:
                label_list = [""]

            multi = len(label_list) > 1
            for label_idx, label in enumerate(label_list):
                name = filename if not multi else f"{filename}#lbl{label_idx:02d}"
                self.samples.append((img_tensor.clone(), str(label), name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        return self.samples[idx]


