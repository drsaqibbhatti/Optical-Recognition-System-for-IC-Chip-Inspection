# train_dbnet.py
import os
import random
import re
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Any, Iterable, Set
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from utils.db_helper import build_db_maps
from utils.util import DBLoss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from model.dbnetDet_convnextv2 import DBNetConvNeXtV2
from model.dbnetRec_convnextv2 import ConvNeXtV2_BiLSTM_CTC
from backbone.convNextV2Block import convnextv2_nano, convnextv2_nano_dims
from typing import Dict, List, Optional, Tuple, Any
import cv2



def parse_ids_to_train(x) -> Optional[Set[int]]:
    """Normalize Ids_to_train into a set[int] or None."""
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return {int(x)}
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return None
        try:
            return {int(x)}
        except ValueError:
            return None
    if isinstance(x, (list, tuple, set)):
        out = set()
        for v in x:
            try:
                out.add(int(v))
            except (TypeError, ValueError):
                continue
        return out if out else set()
    return None


def unpack_sample_strict(sample: Any) -> Tuple[torch.Tensor, List[Any], List[Any], List[Any]]:
    """
    STRICT expected dataset output (consistent):
      (img, labels, polys, pair_ids) OR (img, labels, polys, pair_ids, ...)
    """
    if not isinstance(sample, (tuple, list)):
        raise TypeError(f"Dataset sample must be tuple/list, got: {type(sample)}")
    if len(sample) < 4:
        raise ValueError(f"Dataset sample must have >= 4 items: (img, labels, polys, pair_ids, ...). Got len={len(sample)}")

    img = sample[0]
    labels = sample[1]
    polys = sample[2]
    pair_ids = sample[3]

    labels = labels if isinstance(labels, list) else [labels]
    polys = polys if isinstance(polys, list) else [polys]
    pair_ids = pair_ids if isinstance(pair_ids, list) else [pair_ids]

    n = min(len(labels), len(polys), len(pair_ids))
    return img, labels[:n], polys[:n], pair_ids[:n]


def filter_by_ids(
    labels: List[Any],
    polys: List[Any],
    pair_ids: List[Any],
    ids_to_train: Optional[Set[int]] = None
) -> Tuple[List[Any], List[Any], List[Optional[int]]]:
    """
    Returns (labels, polys, pair_ids) filtered by ids_to_train (set[int] or None),
    and sorted by id for stable order.
    """
    triples = []
    for pid, poly, lab in zip(pair_ids, polys, labels):
        try:
            pid_int = int(pid) if pid is not None else None
        except (TypeError, ValueError):
            pid_int = None
        triples.append((pid_int, poly, lab))

    # stable sort by id (None goes last)
    triples.sort(key=lambda t: (t[0] is None, t[0] if t[0] is not None else 10**9))

    if ids_to_train is None:
        kept = triples
    else:
        kept = [t for t in triples if t[0] in ids_to_train]

    out_ids = [t[0] for t in kept]
    out_polys = [t[1] for t in kept]
    out_labels = [t[2] for t in kept]
    return out_labels, out_polys, out_ids



@dataclass
class TrainerConfig:
    imgH: int = 640
    imgW: int = 640

    text_imgH: int = 128
    text_imgW: int = 128

    
    batch_size: int = 32
    lr: float = 1e-4
    epochs: int = 2000
    num_workers: int = 1

    shrink_ratio: float = 0.4

    load_label: bool = True
    load_box_polygons: bool = True

    visualize_joint: bool = True
    visualize_joint_max: int = 70
    visualize_joint_crop_max: int = 4
    visualize_joint_dir: Optional[str] = None

    rec_crop_margin_min: float = 0.0
    rec_crop_margin: float = 0.0

    rec_visualize_training: bool = True
    
    # Real-time visualization flags
    realtime_visualization: bool = True
    visualize_every_n_batches: int = 10  # Show visualization every N batches

    patience: int = 20
    min_delta: float = 1e-8

    # recognition label filtering
    rec_placeholder_labels: Optional[List[str]] = None  # e.g. ["box"] to ignore placeholder labels
    rec_pad_preserve_aspect: bool = True




# ============================================================
# CTC converter
# ============================================================
class CTCLabelConverter:
    def __init__(self, characters: str, ignore_case: bool = True):
        if ignore_case:
            characters = characters.lower()
        self.ignore_case = ignore_case
        self.characters = ["<blank>"] + list(dict.fromkeys(characters))
        self.blank_idx = 0
        self.char_to_idx = {char: idx for idx, char in enumerate(self.characters)}

    def encode(self, texts: List[str], max_length: Optional[int] = None):
        if self.ignore_case:
            texts = [t.lower() for t in texts]

        if max_length is None:
            filtered_lengths = []
            for text in texts:
                count = sum(1 for ch in text if ch in self.char_to_idx)
                filtered_lengths.append(count)
            max_length = max(filtered_lengths) if filtered_lengths else 0

        targets = torch.zeros(len(texts), max_length, dtype=torch.long)
        lengths = torch.zeros(len(texts), dtype=torch.int32)

        for i, text in enumerate(texts):
            col = 0
            for ch in text:
                idx = self.char_to_idx.get(ch)
                if idx is None:
                    continue
                if col >= max_length:
                    break
                targets[i, col] = idx
                col += 1
            lengths[i] = col

        return targets, lengths

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


class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float("inf")

    def check(self, loss: float) -> bool:
        if loss < (self.min_loss - self.min_delta):
            self.min_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# ============================================================
# Collates (ID-aware)
# ============================================================
class CollateDet:
    def __init__(self, ids_to_train):
        self.ids_to_train = ids_to_train  # set[int] | None

    def __call__(self, batch):
        imgs = torch.stack([b[0] for b in batch], dim=0)

        polys_batch = []
        for sample in batch:
            img, labels, polys, pair_ids = unpack_sample_strict(sample)
            _, polys_f, _ = filter_by_ids(labels, polys, pair_ids, ids_to_train=self.ids_to_train)
            polys_batch.append(polys_f)

        return imgs, polys_batch



def collate_rec(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    labels = [b[1] for b in batch]
    return imgs, labels


def make_batch_targets(polys_batch, H, W, device, shrink_ratio=0.4):
    gts, masks, tms, tmask = [], [], [], []
    for polys in polys_batch:
        polys = polys if polys is not None else []
        gt, mask, thresh_map, thresh_mask = build_db_maps(polys, H, W, shrink_ratio=shrink_ratio)
        gts.append(gt)
        masks.append(mask)
        tms.append(thresh_map)
        tmask.append(thresh_mask)

    gt = torch.from_numpy(np.stack(gts, axis=0)).float().to(device)
    mask = torch.from_numpy(np.stack(masks, axis=0)).float().to(device)
    thresh_map = torch.from_numpy(np.stack(tms, axis=0)).float().to(device)
    thresh_mask = torch.from_numpy(np.stack(tmask, axis=0)).float().to(device)

    return {"gt": gt, "mask": mask, "thresh_map": thresh_map, "thresh_mask": thresh_mask}


# ============================================================
# Utils: run dirs + visualization helpers
# ============================================================
def prepare_run_directory(
    base_dir: Path,
    img_w: int,
    img_h: int,
    model_name: str,
    suffix: str = "",
    existing_root: Optional[Path] = None,
) -> Tuple[Path, str, Path]:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(ch for ch in model_name if ch.isalnum() or ch in ("-", "_")) or "Model"

    if existing_root is not None:
        run_root = existing_root
    else:
        pattern = re.compile(r"run_(\d+)$")
        max_idx = 0
        for child in base_dir.iterdir():
            if child.is_dir():
                m = pattern.match(child.name)
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
        run_root = base_dir / f"run_{max_idx + 1}"
        run_root.mkdir(parents=True, exist_ok=False)

    sub_name = suffix if suffix else safe_name
    run_dir = run_root / sub_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, safe_name, run_root


def tensor_to_uint8(img_tensor: torch.Tensor, rgb: bool) -> np.ndarray:
    arr = img_tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * 0.5) + 0.5
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)

    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    if not rgb:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def make_rec_preview(crop_tensor: torch.Tensor, gt: str, pred: str, rgb: bool) -> np.ndarray:
    img_bgr = tensor_to_uint8(crop_tensor, rgb=rgb)
    _, w, _ = img_bgr.shape
    panel = np.full((80, w, 3), 255, dtype=np.uint8)
    cv2.putText(panel, f"GT: {gt}", (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(panel, f"Pred: {pred}", (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack((img_bgr, panel))


def create_text_visualization(gt_text: str, pred_text: str, width: int = 400, height_per_text: int = 80) -> np.ndarray:
    """
    Create a stacked visualization with white background:
    - Top: Ground truth label
    - Bottom: Predicted text
    """
    total_height = height_per_text * 2
    panel = np.full((total_height, width, 3), 255, dtype=np.uint8)
    
    # Draw dividing line
    cv2.line(panel, (0, height_per_text), (width, height_per_text), (200, 200, 200), 2)
    
    # Add GT text (top half)
    cv2.putText(panel, f"GT: {gt_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 150, 0), 2, cv2.LINE_AA)
    
    # Add Pred text (bottom half)
    cv2.putText(panel, f"Pred: {pred_text}", (10, height_per_text + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 2, cv2.LINE_AA)
    
    return panel


def visualize_detection_realtime(img_tensor: torch.Tensor, pred_boxes: List[np.ndarray], gt_boxes: List[np.ndarray], 
                                  rgb: bool = True, window_name: str = "Detection Visualization") -> np.ndarray:
    """
    Create visualization of original image with predicted and ground truth boxes.
    - Green: Ground truth boxes
    - Blue: Predicted boxes
    """
    img_bgr = tensor_to_uint8(img_tensor, rgb=rgb)
    overlay = img_bgr.copy()
    
    # Draw GT boxes in green
    for box in gt_boxes:
        if box is not None and len(box) > 0:
            pts = np.array(box, dtype=np.int32)
            if pts.ndim == 2 and pts.shape[0] >= 2:
                cv2.polylines(overlay, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    
    # Draw predicted boxes in blue
    for box in pred_boxes:
        if box is not None and len(box) > 0:
            pts = np.array(box, dtype=np.int32)
            if pts.ndim == 2 and pts.shape[0] >= 2:
                cv2.polylines(overlay, [pts.reshape(-1, 1, 2)], True, (255, 0, 0), 3)
    
    # Add legend
    legend_height = 30
    legend = np.full((legend_height, overlay.shape[1], 3), 50, dtype=np.uint8)
    cv2.putText(legend, "Green: GT | Blue: Predicted", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    overlay_with_legend = np.vstack((overlay, legend))
    
    return overlay_with_legend


def create_grid_visualization(images_list, cols=2):
    """
    Create a grid of images to display multiple images in one window.
    Args:
        images_list: List of images (numpy arrays)
        cols: Number of columns in the grid
    Returns:
        Grid image combining all inputs
    """
    if not images_list:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    n_images = len(images_list)
    rows = (n_images + cols - 1) // cols  # Ceiling division
    
    # Get max dimensions
    max_h = max(img.shape[0] for img in images_list)
    max_w = max(img.shape[1] for img in images_list)
    
    # Create grid
    grid_rows = []
    for row_idx in range(rows):
        row_images = []
        for col_idx in range(cols):
            img_idx = row_idx * cols + col_idx
            if img_idx < n_images:
                img = images_list[img_idx]
                # Pad image to max dimensions
                pad_h = max_h - img.shape[0]
                pad_w = max_w - img.shape[1]
                if pad_h > 0 or pad_w > 0:
                    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(50, 50, 50))
                row_images.append(img)
            else:
                # Empty placeholder
                row_images.append(np.full((max_h, max_w, 3), 50, dtype=np.uint8))
        grid_rows.append(np.hstack(row_images))
    
    return np.vstack(grid_rows)


def resize_and_pad_rec_tensor(
    crop_tensor: torch.Tensor,
    target_h: int,
    target_w: int,
    preserve_aspect: bool,
) -> torch.Tensor:
    if target_h <= 0 or target_w <= 0:
        raise ValueError("target_h and target_w must be positive when resizing recognition crops.")

    crop_tensor = crop_tensor.unsqueeze(0)
    if not preserve_aspect:
        resized = F.interpolate(crop_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return resized.squeeze(0)

    orig_h = crop_tensor.shape[-2]
    orig_w = crop_tensor.shape[-1]
    if orig_h <= 0 or orig_w <= 0:
        return crop_tensor.squeeze(0)

    scale_h = target_h / orig_h
    scale_w = target_w / orig_w
    scale = min(scale_h, scale_w)
    scale = max(scale, 1e-6)

    scaled_h = max(1, int(round(orig_h * scale)))
    scaled_w = max(1, int(round(orig_w * scale)))

    resized = F.interpolate(crop_tensor, size=(scaled_h, scaled_w), mode="bilinear", align_corners=False).squeeze(0)

    pad_h = max(target_h - scaled_h, 0)
    pad_w = max(target_w - scaled_w, 0)
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        resized = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom))

    if resized.shape[-2] > target_h or resized.shape[-1] > target_w:
        resized = resized[:, :target_h, :target_w]

    if resized.shape[-2] < target_h or resized.shape[-1] < target_w:
        diff_h = target_h - resized.shape[-2]
        diff_w = target_w - resized.shape[-1]
        if diff_h > 0 or diff_w > 0:
            pad_top = diff_h // 2
            pad_bottom = diff_h - pad_top
            pad_left = diff_w // 2
            pad_right = diff_w - pad_left
            resized = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom))
        resized = resized[:, :target_h, :target_w]

    return resized


def plot_detection_metrics(history: List[dict], out_path: Path):
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]

    def plot_series(values: List[Optional[float]], label: str, style: str = "-"):
        xs = [ep for ep, val in zip(epochs, values) if val is not None]
        ys = [val for val in values if val is not None]
        if xs:
            plt.plot(xs, ys, style, label=label)

    plt.figure(figsize=(8, 5))

    train_exists = any(entry.get("train") for entry in history)
    val_exists = any(entry.get("val") for entry in history)

    if train_exists:
        for key, label in (("loss", "Train Loss"), ("bce", "Train BCE"), ("dice", "Train Dice"), ("l1", "Train L1")):
            values = [entry.get("train", {}).get(key) for entry in history]
            plot_series(values, label)

    if val_exists:
        for key, label in (("loss", "Val Loss"), ("bce", "Val BCE"), ("dice", "Val Dice"), ("l1", "Val L1")):
            values = [entry.get("val", {}).get(key) for entry in history]
            plot_series(values, label, style="--")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Detection Training Loss Metrics")
    if train_exists or val_exists:
        plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_recognition_loss(history: List[dict], out_path: Path):
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]

    def plot_series(values: List[Optional[float]], label: str, style: str = "-"):
        xs = [ep for ep, val in zip(epochs, values) if val is not None]
        ys = [val for val in values if val is not None]
        if xs:
            plt.plot(xs, ys, style, label=label)

    plt.figure(figsize=(8, 5))

    train_values = [entry.get("train_loss") for entry in history]
    val_values = [entry.get("val_loss") for entry in history]

    plot_series(train_values, "Train CTC Loss")
    plot_series(val_values, "Val CTC Loss", style="--")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Recognition Training Loss")
    if any(val is not None for val in train_values + val_values):
        plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_joint_visual(
    img_tensor: torch.Tensor,
    polys: List[List[List[float]]],
    samples: List[Tuple[torch.Tensor, str]],
    out_path: Path,
    rgb: bool,
    crop_max: int,
    predictions: Optional[List[str]] = None,
):
    base = tensor_to_uint8(img_tensor, rgb)
    overlay = base.copy()

    for poly in polys:
        if not poly:
            continue
        pts = np.array(poly, dtype=np.int32)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        cv2.polylines(overlay, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)

    cols = min(len(samples), crop_max)
    fig_cols = 1 + cols

    fig, axes = plt.subplots(1, fig_cols, figsize=(4 * fig_cols, 4))
    if fig_cols == 1:
        axes = [axes]

    axes[0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Image + GT")
    axes[0].axis("off")

    for i in range(cols):
        crop_tensor, label = samples[i]
        crop_img = tensor_to_uint8(crop_tensor, rgb=True)
        axes[i + 1].imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        
        # Add prediction below label if available
        if predictions and i < len(predictions):
            title_text = f"Label: {label}\nPred: {predictions[i]}"
        else:
            title_text = f"Label: {label}"
        
        axes[i + 1].set_title(title_text, fontsize=10)
        axes[i + 1].axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
# Crop helper: from polygon -> rec input
# ============================================================
def sample_rec_crop(
    img_tensor: torch.Tensor,
    poly: List[List[float]],
    label: str,
    text_imgH: int,
    text_imgW: int,
    rgb: bool,
    margin_min_ratio: float,
    margin_ratio: float,
    preserve_aspect: bool,
) -> Optional[Tuple[torch.Tensor, str]]:
    if not poly or label is None:
        return None

    img_np = img_tensor.detach().cpu().numpy()
    if img_np.ndim != 3:
        return None

    _, img_h, img_w = img_np.shape
    pts = np.array(poly, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return None

    xs, ys = pts[:, 0], pts[:, 1]
    x_min = int(np.floor(np.clip(xs.min(), 0, img_w - 1)))
    x_max = int(np.ceil(np.clip(xs.max(), 0, img_w - 1)))
    y_min = int(np.floor(np.clip(ys.min(), 0, img_h - 1)))
    y_max = int(np.ceil(np.clip(ys.max(), 0, img_h - 1)))
    if x_max < x_min or y_max < y_min:
        return None

    box_w = max(x_max - x_min + 1, 1)
    box_h = max(y_max - y_min + 1, 1)

    # random margins per side
    sides = {k: random.random() < 0.5 for k in ["left", "right", "top", "bottom"]}
    if not any(sides.values()) and random.random() < 0.5:
        sides[random.choice(list(sides.keys()))] = True

    low = min(margin_min_ratio, margin_ratio)
    high = max(margin_min_ratio, margin_ratio)

    def sample_ratio(active: bool) -> float:
        return random.uniform(low, high) if active else 0.0

    left_ratio = sample_ratio(sides["left"])
    right_ratio = sample_ratio(sides["right"])
    top_ratio = sample_ratio(sides["top"])
    bottom_ratio = sample_ratio(sides["bottom"])

    x_min_aug, x_max_aug = x_min, x_max
    y_min_aug, y_max_aug = y_min, y_max

    # expand/shrink
    left_pad = int(round(box_w * left_ratio))
    right_pad = int(round(box_w * right_ratio))
    top_pad = int(round(box_h * top_ratio))
    bottom_pad = int(round(box_h * bottom_ratio))

    x_min_aug = max(0, x_min_aug - left_pad)
    x_max_aug = min(img_w - 1, x_max_aug + right_pad)
    y_min_aug = max(0, y_min_aug - top_pad)
    y_max_aug = min(img_h - 1, y_max_aug + bottom_pad)

    crop = img_np[:, y_min_aug:y_max_aug + 1, x_min_aug:x_max_aug + 1]
    if crop.shape[1] == 0 or crop.shape[2] == 0:
        return None

    crop_tensor = torch.from_numpy(crop)
    crop_tensor = resize_and_pad_rec_tensor(crop_tensor, text_imgH, text_imgW, preserve_aspect)

    if crop_tensor.shape[0] == 1 and rgb:
        crop_tensor = crop_tensor.repeat(3, 1, 1)

    return crop_tensor, str(label)


def convert_to_rec_examples(
    img_tensor: torch.Tensor,
    polys: List[List[List[float]]],
    labels: List[str],
    text_imgH: int,
    text_imgW: int,
    rgb: bool,
    margin_min_ratio: float,
    margin_ratio: float,
    preserve_aspect: bool,
    visual_context: Optional[dict] = None,
):
    samples = []
    for poly, label in zip(polys, labels):
        s = sample_rec_crop(
            img_tensor,
            poly,
            label,
            text_imgH,
            text_imgW,
            rgb,
            margin_min_ratio,
            margin_ratio,
            preserve_aspect,
        )
        if s is not None:
            samples.append(s)

    if visual_context and visual_context.get("enabled") and samples:
        limit = visual_context.get("max", 0)
        if limit > 0 and visual_context["count"] < limit:
            out_dir = Path(visual_context["dir"])
            out_path = out_dir / f"sample_{visual_context['count']:03d}.png"
            try:
                # Get predictions if rec_model is available
                predictions = None
                if "rec_model" in visual_context and "rec_converter" in visual_context:
                    rec_model = visual_context["rec_model"]
                    rec_converter = visual_context["rec_converter"]
                    device = visual_context.get("device", "cpu")
                    
                    predictions = []
                    with torch.no_grad():
                        for crop_tensor, _ in samples:
                            crop_batch = crop_tensor.unsqueeze(0).to(device)
                            logits, logit_lengths = rec_model(crop_batch)
                            log_probs = logits.log_softmax(dim=-1)
                            pred_indices = log_probs.detach().cpu().argmax(dim=-1).permute(1, 0)
                            decoded = rec_converter.decode(pred_indices, logit_lengths.cpu())
                            predictions.append(decoded[0] if decoded else "")
                
                save_joint_visual(img_tensor, polys, samples, out_path, rgb=rgb, 
                                crop_max=visual_context.get("crop_max", 4), predictions=predictions)
                visual_context["count"] += 1
                print(f"Saved joint visualization -> {out_path}")
            except Exception as exc:
                print(f"[WARN] Failed to save joint visualization {out_path}: {exc}")

    return samples


# ============================================================
# RecognitionCropDataset (from stored full images + pairs)
# ============================================================
class RecognitionCropDataset(Dataset):
    def __init__(
        self,
        entries: List[dict],
        text_imgH: int,
        text_imgW: int,
        rgb: bool,
        margin_min_ratio: float,
        margin_ratio: float,
        preserve_aspect: bool,
        max_attempts: int = 8,
    ):
        self.entries = entries
        self.text_imgH = text_imgH
        self.text_imgW = text_imgW
        self.rgb = rgb
        self.margin_min_ratio = margin_min_ratio
        self.margin_ratio = margin_ratio
        self.preserve_aspect = preserve_aspect
        self.max_attempts = max_attempts

        self.index = []
        for ei, entry in enumerate(entries):
            for pi in range(len(entry.get("pairs", []))):
                self.index.append((ei, pi))

        if not self.index:
            raise ValueError("RecognitionCropDataset received no valid polygon/label pairs.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ei, pi = self.index[idx]
        entry = self.entries[ei]
        pair = entry["pairs"][pi]
        img_tensor = entry["img"]
        poly = pair["poly"]
        label = pair["label"]

        sample = None
        for _ in range(self.max_attempts):
            sample = sample_rec_crop(
                img_tensor, poly, label,
                self.text_imgH, self.text_imgW,
                self.rgb, self.margin_min_ratio, self.margin_ratio,
                self.preserve_aspect,
            )
            if sample is not None:
                break

        if sample is None:
            sample = sample_rec_crop(
                img_tensor, poly, label,
                self.text_imgH, self.text_imgW,
                self.rgb, 0.0, 0.0,
                self.preserve_aspect,
            )

        if sample is None:
            resized = resize_and_pad_rec_tensor(
                img_tensor,
                self.text_imgH,
                self.text_imgW,
                self.preserve_aspect,
            )
            if resized.shape[0] == 1 and self.rgb:
                resized = resized.repeat(3, 1, 1)
            sample = (resized, str(label))

        return sample

class RecognitionDirectDataset(Dataset):
    """Wrap a dataset of pre-cropped text images to return (image, label)."""

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset
        self.rgb = getattr(base_dataset, "rgb", True)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        if isinstance(sample, (list, tuple)):
            img = sample[0]
            labels = sample[1] if len(sample) > 1 else []
        else:
            img = sample
            labels = []

        if isinstance(labels, list):
            label = str(labels[0]) if labels else ""
        else:
            label = str(labels)

        return img, label



# ============================================================
# Pretrained backbone
# ============================================================
def load_pretrained_backbone(backbone, path: Optional[str], use_pretrained: bool):
    if not use_pretrained or not path:
        print("Skipping backbone pretraining load.")
        return
    if not os.path.exists(path):
        print(f"Pretrained file not found: {path}")
        return

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if not k.startswith("head.")}
    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights from {path}")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))


# ============================================================
# DET init/run
# ============================================================
def initialize_detection(
    dataset,
    config: TrainerConfig,
    device: str,
    backbone_ckpt: Optional[str],
    use_pretrained: bool,
    base_run_dir: Path,
    ids_to_train: Optional[Set[int]],
    val_dataset=None,
):
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,   # can keep >0 now
        pin_memory=True,
        drop_last=False,
        collate_fn=CollateDet(ids_to_train),   # ✅ picklable
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=CollateDet(ids_to_train),
        )


    
    
    
    
    backbone = convnextv2_nano()
    dims = list(convnextv2_nano_dims)
    
    

    load_pretrained_backbone(backbone, backbone_ckpt, use_pretrained)
    model = DBNetConvNeXtV2(backbone=backbone, in_channels_list=dims, out_channels=256, k=50).to(device)
    criterion = DBLoss(bce_scale=5.0, l1_scale=10.0, dice_scale=1.0)
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    run_dir, model_tag, run_root = prepare_run_directory(base_run_dir, config.imgW, config.imgH, type(model).__name__, suffix="Det")
    print(f"Saving detection checkpoints to {run_dir}")

    return dict(
        loader=loader,
        model=model,
        criterion=criterion,
        optim=optim,
        scaler=scaler,
        run_dir=run_dir,
        model_tag=model_tag,
        device=device,
        run_root=run_root,
        best_loss=float("inf"),
        best_ckpt_path=None,
        metrics_history=[],
        val_loader=val_loader,
    )


def run_detection_epoch(state: dict, config: TrainerConfig, epoch: int, total_epochs: int):
    model = state["model"]
    loader = state["loader"]
    criterion = state["criterion"]
    optim = state["optim"]
    scaler = state["scaler"]
    device = state["device"]
    run_dir = state["run_dir"]
    val_loader = state.get("val_loader")

    model.train()
    t0 = time.time()
    running = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "l1": 0.0}
    n = 0
    
    realtime_viz = getattr(config, "realtime_visualization", False)
    viz_interval = getattr(config, "visualize_every_n_batches", 10)

    for batch_idx, (imgs, polys_batch) in enumerate(tqdm(loader, desc=f"Det Epoch {epoch}/{total_epochs}", leave=False)):
        imgs = imgs.to(device, non_blocking=True)
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)

        H, W = imgs.shape[-2], imgs.shape[-1]
        batch_targets = make_batch_targets(polys_batch, H, W, device, shrink_ratio=config.shrink_ratio)

        optim.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            pred = model(imgs)
            loss, metrics = criterion(pred, batch_targets)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        for k in running:
            running[k] += metrics[k]
        n += 1
        
        # Real-time visualization for detection
        if realtime_viz and (batch_idx % viz_interval == 0 or batch_idx == 0):
            with torch.no_grad():
                # Cycle through all images in batch, showing one at a time in same window
                batch_size = imgs.shape[0]
                
                for idx in range(batch_size):
                    sample_img = imgs[idx].detach().cpu()
                    sample_polys = polys_batch[idx] if idx < len(polys_batch) else []
                    
                    # Get predicted boxes from probability map
                    prob_map = pred["shrink"][idx, 0].detach().cpu().numpy()
                    threshold = 0.7
                    binary = (prob_map > threshold).astype(np.uint8)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Filter and get top 3 boxes by area
                    valid_contours = [cnt for cnt in contours if len(cnt) >= 4]
                    if len(valid_contours) > 3:
                        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:3]
                    pred_boxes = [cnt.squeeze() for cnt in valid_contours]
                    
                    # Create visualization for this image
                    rgb_flag = True
                    viz_img = visualize_detection_realtime(sample_img, pred_boxes, sample_polys, rgb=rgb_flag, 
                                                           window_name="Detection")
                    
                    # Show in same window (updates with each image)
                    cv2.imshow("Window 1: Detection - Original + Boxes", viz_img)
                    cv2.waitKey(500)  # Show each image for 500ms

    for k in running:
        running[k] /= max(n, 1)

    val_metrics = None
    if val_loader is not None:
        val_running = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "l1": 0.0}
        val_n = 0
        with torch.no_grad():
            for imgs, polys_batch in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                if imgs.shape[1] == 1:
                    imgs = imgs.repeat(1, 3, 1, 1)

                H, W = imgs.shape[-2], imgs.shape[-1]
                batch_targets = make_batch_targets(polys_batch, H, W, device, shrink_ratio=config.shrink_ratio)

                pred = model(imgs)
                _, metrics = criterion(pred, batch_targets)

                for k in val_running:
                    val_running[k] += metrics[k]
                val_n += 1

        if val_n > 0:
            for k in val_running:
                val_running[k] /= val_n
        val_metrics = val_running
        # keep model in train mode so loss dictionary structure stays consistent

    dt = time.time() - t0
    msg = (f"[Det {epoch:03d}/{total_epochs}] loss={running['loss']:.7f} "
           f"bce={running['bce']:.7f} dice={running['dice']:.7f} l1={running['l1']:.7f} time={dt:.1f}s")
    if val_metrics:
        msg += (f" | val_loss={val_metrics['loss']:.7f} val_bce={val_metrics['bce']:.7f} "
                f"val_dice={val_metrics['dice']:.7f} val_l1={val_metrics['l1']:.7f}")
    print(msg)

    best_loss = state.get("best_loss", float("inf"))
    best_ckpt_path = state.get("best_ckpt_path")
    improvement_threshold = config.min_delta if hasattr(config, "min_delta") else 0.0

    history = state.setdefault("metrics_history", [])
    history.append({
        "epoch": epoch,
        "train": dict(running),
        "val": dict(val_metrics) if val_metrics else None,
    })
    metrics_path = run_dir / "det_metrics.json"
    metrics_path.write_text(json.dumps(history, indent=2))
    if len(history) % 10 == 0 or epoch == total_epochs:
        plot_detection_metrics(history, run_dir / "plots" / "det_metrics.png")

    candidate_loss = val_metrics["loss"] if val_metrics else running["loss"]

    if candidate_loss < (best_loss - improvement_threshold):
        ckpt_path = run_dir / f"best_E{epoch:03d}_L{candidate_loss:.6f}_Det.pth"
        ckpt = {"epoch": epoch, "model": model.state_dict(), "optim": optim.state_dict()}
        torch.save(ckpt, ckpt_path)
        if best_ckpt_path and best_ckpt_path.exists():
            try:
                best_ckpt_path.unlink()
            except Exception as exc:
                print(f"[WARN] Failed to remove previous best detection checkpoint {best_ckpt_path}: {exc}")
        state["best_loss"] = candidate_loss
        state["best_ckpt_path"] = ckpt_path
        print(f"[Det {epoch:03d}] Saved new best checkpoint -> {ckpt_path.name}")
    if val_metrics:
        running["val_loss"] = val_metrics["loss"]
        running["val_bce"] = val_metrics["bce"]
        running["val_dice"] = val_metrics["dice"]
        running["val_l1"] = val_metrics["l1"]
    return running


# ============================================================
# REC init/run
# ============================================================
def initialize_recognition(
    rec_dataset: Dataset,
    config: TrainerConfig,
    device: str,
    backbone_ckpt: Optional[str],
    use_pretrained: bool,
    base_run_dir: Path,
    run_root: Optional[Path] = None,
    val_dataset: Optional[Dataset] = None,
):
    loader = DataLoader(
        rec_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_rec,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_rec,
        )

    backbone = convnextv2_nano()
    dims = list(convnextv2_nano_dims)
    
    
    
    load_pretrained_backbone(backbone, backbone_ckpt, use_pretrained)

    vocab = " 0123456789abcdefghijklmnopqrstuvwxyz"
    converter = CTCLabelConverter(vocab)
    vocab_size = len(converter.characters)

    model = ConvNeXtV2_BiLSTM_CTC(backbone=backbone, in_channels_c5=dims[3], vocab_size=vocab_size).to(device)
    criterion = torch.nn.CTCLoss(reduction="mean", zero_infinity=True)
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    run_dir, model_tag, run_root = prepare_run_directory(base_run_dir, config.text_imgW, config.text_imgH,
                                                         type(model).__name__, suffix="Rec", existing_root=run_root)
    print(f"Saving recognition checkpoints to {run_dir}")

    preview_dir = run_dir / "rec_vis"
    preview_enabled = bool(getattr(config, "rec_visualize_training", True))

    return dict(
        loader=loader,
        model=model,
        criterion=criterion,
        optim=optim,
        scaler=scaler,
        converter=converter,
        run_dir=run_dir,
        model_tag=model_tag,
        device=device,
        run_root=run_root,
        preview_dir=preview_dir,
        preview_count=0,
        preview_enabled=preview_enabled,
        best_loss=float("inf"),
        best_ckpt_path=None,
        loss_history=[],
        val_loader=val_loader,
    )


def run_recognition_epoch(state: dict, config: TrainerConfig, epoch: int, total_epochs: int):
    model = state["model"]
    loader = state["loader"]
    criterion = state["criterion"]
    optim = state["optim"]
    scaler = state["scaler"]
    converter = state["converter"]
    device = state["device"]
    run_dir = state["run_dir"]
    preview_dir: Path = state["preview_dir"]
    preview_count = state["preview_count"]
    preview_enabled = state.get("preview_enabled", True)
    val_loader = state.get("val_loader")

    model.train()
    t0 = time.time()
    running_loss = 0.0
    n = 0
    
    realtime_viz = getattr(config, "realtime_visualization", False)
    viz_interval = getattr(config, "visualize_every_n_batches", 10)

    if preview_enabled:
        preview_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc=f"Rec Epoch {epoch}/{total_epochs}", leave=False)):
        imgs = imgs.to(device, non_blocking=True)

        targets_cpu, target_lengths = converter.encode(labels)
        target_lengths = target_lengths.to(torch.long)
        target_lengths_cpu = target_lengths.cpu()

        flat_targets = torch.cat([seq[:int(l)] for seq, l in zip(targets_cpu, target_lengths_cpu)], dim=0).to(device)

        optim.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits, logit_lengths = model(imgs)  # logits: (T,B,C)
            logit_lengths = logit_lengths.detach().cpu().to(torch.long)


            print(
                "[CTC lens] min logit_len:", int(logit_lengths.min()),
                "max target_len:", int(target_lengths_cpu.max()),
                "T:", int(logits.shape[0])
            )
    
            log_probs = logits.log_softmax(dim=-1)
            loss = criterion(log_probs, flat_targets, logit_lengths, target_lengths_cpu)

            # quick preview
            pred_indices = log_probs.detach().cpu().argmax(dim=-1).permute(1, 0)  # (B,T)
            decoded = converter.decode(pred_indices, logit_lengths)

            preview = list(zip(labels, decoded))[:3]
            for gt, pr in preview:
                tqdm.write(f"[Rec {epoch:03d}] GT: {gt} | Pred: {pr}")

            if preview_enabled and preview:
                for local_idx, (gt, pr) in enumerate(preview):
                    crop_tensor = imgs[local_idx].detach().cpu()
                    rgb_flag = crop_tensor.shape[0] != 1
                    stacked = make_rec_preview(crop_tensor, gt, pr, rgb=rgb_flag)
                    out_path = preview_dir / f"epoch{epoch:03d}_sample{preview_count + local_idx:05d}.png"
                    cv2.imwrite(str(out_path), stacked)
                preview_count += len(preview)
            
            # Real-time text visualization - cycle through all samples in same window
            if realtime_viz and (batch_idx % viz_interval == 0 or batch_idx == 0):
                all_preview = list(zip(labels, decoded))
                for gt_text, pred_text in all_preview:
                    text_viz = create_text_visualization(gt_text, pred_text)
                    cv2.imshow("Window 2: Text Recognition - GT vs Prediction", text_viz)
                    cv2.waitKey(500)  # Show each prediction for 500ms

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        running_loss += loss.item()
        n += 1

    avg_loss = running_loss / max(n, 1)

    val_loss = None
    if val_loader is not None:
        model.eval()
        val_running = 0.0
        val_n = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)

                targets_cpu, target_lengths = converter.encode(labels)
                target_lengths = target_lengths.to(torch.long)
                target_lengths_cpu = target_lengths.cpu()

                flat_targets = torch.cat([
                    seq[:int(l)] for seq, l in zip(targets_cpu, target_lengths_cpu)
                ], dim=0).to(device)

                logits, logit_lengths = model(imgs)
                logit_lengths = logit_lengths.detach().cpu().to(torch.long)
                log_probs = logits.log_softmax(dim=-1)
                loss = criterion(log_probs, flat_targets, logit_lengths, target_lengths_cpu)

                val_running += float(loss.item())
                val_n += 1

        val_loss = val_running / max(val_n, 1)
        model.train()

    dt = time.time() - t0
    msg = f"[Rec {epoch:03d}/{total_epochs}] loss={avg_loss:.7f} time={dt:.1f}s"
    if val_loss is not None:
        msg += f" | val_loss={val_loss:.7f}"
    print(msg)

    history = state.setdefault("loss_history", [])
    history.append({"epoch": epoch, "train_loss": avg_loss, "val_loss": val_loss})
    metrics_path = run_dir / "rec_metrics.json"
    metrics_path.write_text(json.dumps(history, indent=2))
    if len(history) % 10 == 0 or epoch == total_epochs:
        plot_recognition_loss(history, run_dir / "plots" / "rec_metrics.png")

    best_loss = state.get("best_loss", float("inf"))
    best_ckpt_path = state.get("best_ckpt_path")
    improvement_threshold = config.min_delta if hasattr(config, "min_delta") else 0.0

    candidate_loss = val_loss if val_loss is not None else avg_loss

    if candidate_loss < (best_loss - improvement_threshold):
        ckpt_path = run_dir / f"best_E{epoch:03d}_L{candidate_loss:.6f}_Rec.pth"
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "converter": converter.characters,
        }
        torch.save(ckpt, ckpt_path)
        if best_ckpt_path and best_ckpt_path.exists():
            try:
                best_ckpt_path.unlink()
            except Exception as exc:
                print(f"[WARN] Failed to remove previous best recognition checkpoint {best_ckpt_path}: {exc}")
        state["best_loss"] = candidate_loss
        state["best_ckpt_path"] = ckpt_path
        print(f"[Rec {epoch:03d}] Saved new best checkpoint -> {ckpt_path.name}")

    state["preview_count"] = preview_count
    state["last_val_loss"] = val_loss
    return {"loss": avg_loss, "val_loss": val_loss}


# ============================================================
# Build recognition entries from dataset (ID-aware)
# ============================================================
def collect_rec_entries(dataset: Dataset, config: TrainerConfig, ids_to_train: Optional[Set[int]], visual_context: Optional[dict] = None):
    rec_entries = []
    rgb_flag = getattr(dataset, "rgb", True)

    placeholder = set()
    if config.rec_placeholder_labels:
        placeholder = {str(x).strip().lower() for x in config.rec_placeholder_labels if x is not None}

    for sample in dataset:
        img, labels, polys, pair_ids = unpack_sample_strict(sample)

        labels, polys, pair_ids = filter_by_ids(labels, polys, pair_ids, ids_to_train=ids_to_train)
        if not polys or not labels:
            continue

        pairs = []
        for poly, lab, pid in zip(polys, labels, pair_ids):
            lab_str = str(lab).strip()
            if not lab_str:
                continue
            if placeholder and lab_str.lower() in placeholder:
                continue

            pairs.append({"poly": poly, "label": lab_str, "id": pid})

        if not pairs:
            continue

        stored_img = img.detach().cpu()
        rec_entries.append({"img": stored_img, "pairs": pairs})

        if visual_context:
            convert_to_rec_examples(
                stored_img,
                [p["poly"] for p in pairs],
                [p["label"] for p in pairs],
                config.text_imgH,
                config.text_imgW,
                rgb=rgb_flag,
                margin_min_ratio=config.rec_crop_margin_min,
                margin_ratio=config.rec_crop_margin,
                preserve_aspect=config.rec_pad_preserve_aspect,
                visual_context=visual_context,
            )

    return rec_entries


