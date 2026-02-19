import os
import json
import random
import math
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from ConvNxt_eval_Det import (
    build_transform as build_det_transform,
    db_postprocess,
    rescale_boxes,
)


class RecTextDataset(Dataset):
    """
    Recognition dataset:
      - If load_label=True: returns (img, text)  (1 label per image)
      - If load_label=False: returns img only

    NEW:
      - load_detection_model=True: crop ROI using detection model BEFORE resizing to (imgH,imgW)
    """

    def __init__(
        self,
        img_dir,
        json_path=None,
        imgH=32,
        imgW=100,
        rgb=False,
        load_label=False,
        exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"),
        return_filename=False,
        recursive=True,
        num_images=None,

        # ✅ NEW
        load_detection_model=False,
        detection_model_path=None,

        # optional but needed if detection_model_path is a state_dict (.pth)
        detection_model_builder=None,

        # detection preprocessing / postprocessing
        det_input_size=(640, 640),   # (H,W) used for detection forward
        det_thresh=0.30,             # DBNet bin threshold
        det_box_thresh=0.50,         # DBNet box score threshold
        det_unclip_ratio=2.0,
        det_max_candidates=1000,
        crop_margin=0.05,            # expand crop by percentage of box size
        det_device=None,             # None => auto cuda if available
        rotate_left_90=False,        # rotate final crop 90 degrees CCW if True
        show_detected_boxes=False,
        detected_box_dir=None,
    ):
        self.img_dir = img_dir
        self.rgb = rgb
        self.load_label = load_label
        self.return_filename = return_filename
        self.num_images = num_images

        # ✅ NEW
        self.load_detection_model = load_detection_model
        self.detection_model_path = detection_model_path
        self.detection_model_builder = detection_model_builder
        self.det_input_size = det_input_size
        self.det_thresh = det_thresh
        self.det_box_thresh = det_box_thresh
        self.det_unclip_ratio = det_unclip_ratio
        self.det_max_candidates = det_max_candidates
        self.crop_margin = crop_margin
        self.det_device = det_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rotate_left_90 = rotate_left_90
        self.show_detected_boxes = show_detected_boxes
        self.detected_box_dir = None
        self._detected_box_count = 0
        if self.show_detected_boxes:
            base_dir = detected_box_dir if detected_box_dir else Path(self.img_dir) / "detected_boxes"
            self.detected_box_dir = Path(base_dir)
            self.detected_box_dir.mkdir(parents=True, exist_ok=True)

        # will be loaded lazily in each worker/process
        self._det_model = None
        self._det_transform = None

        norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if rgb else transforms.Normalize((0.5,), (0.5,))
        self.transform = transforms.Compose([
            transforms.Resize((imgH, imgW), Image.BICUBIC),
            transforms.ToTensor(),
            norm
        ])

        if self.load_label and json_path is None:
            raise ValueError("json_path must be provided when load_label=True")

        if self.load_label:
            with open(json_path, "r", encoding="utf-8") as f:
                anns = json.load(f)
            if not isinstance(anns, list):
                raise ValueError("Expected combined json to be a list")

            grouped = {}
            for ann in anns:
                if not isinstance(ann, dict):
                    continue
                fname = Path(ann.get("filename", "")).name
                lab = ann.get("label", "")
                if not fname:
                    continue
                grouped.setdefault(fname, {"labels": []})
                grouped[fname]["labels"].append(lab)

            self.samples = []
            for fname, pack in grouped.items():
                img_path = os.path.join(self.img_dir, fname)
                self.samples.append((img_path, pack["labels"]))
            self.samples.sort(key=lambda x: os.path.basename(x[0]))

        else:
            exts_set = {e.lower() for e in exts}
            p = Path(self.img_dir)
            files = [str(f) for f in (p.rglob("*") if recursive else p.iterdir())
                     if f.is_file() and f.suffix.lower() in exts_set]
            files.sort()
            if len(files) == 0:
                raise ValueError(f"No images found under: {self.img_dir} (recursive={recursive}), exts={sorted(exts_set)}")
            self.samples = files

        if self.num_images is not None:
            if self.num_images <= 0:
                raise ValueError("num_images must be positive when provided")
            take = min(self.num_images, len(self.samples))
            self.samples = random.sample(self.samples, take) if take > 0 else []

        # sanity checks for detection mode
        if self.load_detection_model:
            if not self.detection_model_path:
                raise ValueError("detection_model_path must be provided when load_detection_model=True")

    def __len__(self):
        return len(self.samples)

    # ----------------------------
    # Detection model loading
    # ----------------------------
    def _get_det_model(self):
        if self._det_model is not None:
            return self._det_model

        path = self.detection_model_path
        device = torch.device(self.det_device)

        # 1) Try TorchScript first (works with only a path)
        try:
            m = torch.jit.load(path, map_location=device)
            m.eval()
            self._det_model = m
            return self._det_model
        except Exception:
            pass

        # 2) Fallback: state_dict checkpoint -> requires builder
        if self.detection_model_builder is None:
            raise RuntimeError(
                "Detection model is not TorchScript. If detection_model_path is a .pth state_dict, "
                "you MUST pass detection_model_builder=... to build the architecture before loading weights."
            )

        ckpt = torch.load(path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        m = self.detection_model_builder().to(device)
        m.load_state_dict(state, strict=False)
        m.eval()
        self._det_model = m
        return self._det_model

    # ----------------------------
    # Cropping using detection map
    # ----------------------------
    @torch.no_grad()
    def _crop_with_detection(self, pil_img: Image.Image, source_path: str) -> Image.Image:
        det_model = self._get_det_model()
        det_model.eval()

        orig_w, orig_h = pil_img.size
        det_h, det_w = self.det_input_size

        if self._det_transform is None:
            self._det_transform = build_det_transform(det_h, det_w, rgb=True)

        det_tensor = self._det_transform(pil_img.convert("RGB")).unsqueeze(0).to(self.det_device)
        pred = det_model(det_tensor)

        binary_t = None
        if isinstance(pred, dict):
            shrink_t = pred.get("shrink")
            thresh_t = pred.get("thresh")
            if shrink_t is None or thresh_t is None:
                return pil_img
            shrink_t = shrink_t[0, 0]
            thresh_t = thresh_t[0, 0]
            if "binary" in pred:
                binary_t = pred["binary"][0, 0]
            else:
                head = getattr(det_model, "head", None)
                if head is None or not hasattr(head, "step_function"):
                    return pil_img
                binary_t = head.step_function(shrink_t, thresh_t)
        elif isinstance(pred, torch.Tensor):
            if pred.ndim == 4:
                binary_t = pred[0, 0]
            elif pred.ndim == 3:
                binary_t = pred[0]
            else:
                return pil_img
        else:
            return pil_img

        prob_map = binary_t.detach().float().cpu().numpy()
        detections = db_postprocess(
            prob_map,
            bin_thresh=float(self.det_thresh),
            box_thresh=float(self.det_box_thresh),
            max_candidates=int(self.det_max_candidates),
            unclip_ratio=float(self.det_unclip_ratio),
        )

        if not detections:
            return pil_img

        det_h, det_w = self.det_input_size
        scale_x = orig_w / float(det_w)
        scale_y = orig_h / float(det_h)
        detections = rescale_boxes(detections, scale_x=scale_x, scale_y=scale_y)

        # take highest scoring detection
        best_poly, best_score = max(detections, key=lambda item: item[1])
        if best_poly.size == 0:
            return pil_img

        if self.show_detected_boxes and self.detected_box_dir is not None:
            vis_np = np.array(pil_img.convert("RGB"))
            if vis_np.ndim == 3 and vis_np.size > 0:
                vis_np = np.ascontiguousarray(vis_np)
                try:
                    pts = best_poly.astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(vis_np, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                    fname = Path(source_path).stem if source_path else f"det_{self._detected_box_count:05d}"
                    out_name = f"{fname}_{self._detected_box_count:05d}.png"
                    out_path = self.detected_box_dir / out_name
                    if not cv2.imwrite(str(out_path), cv2.cvtColor(vis_np, cv2.COLOR_RGB2BGR)):
                        print(f"[WARN] Failed to save detected box visualization for {source_path} at {out_path}")
                    else:
                        self._detected_box_count += 1
                except Exception as exc:
                    print(f"[WARN] Error saving detected box visualization for {source_path}: {exc}")

        x1 = float(best_poly[:, 0].min())
        y1 = float(best_poly[:, 1].min())
        x2 = float(best_poly[:, 0].max())
        y2 = float(best_poly[:, 1].max())

        bw = x2 - x1
        bh = y2 - y1
        mx = bw * self.crop_margin
        my = bh * self.crop_margin

        x1 = max(0.0, x1 - mx)
        y1 = max(0.0, y1 - my)
        x2 = min(float(orig_w), x2 + mx)
        y2 = min(float(orig_h), y2 + my)

        if x2 <= x1 or y2 <= y1:
            return pil_img

        return pil_img.crop((int(math.floor(x1)), int(math.floor(y1)), int(math.ceil(x2)), int(math.ceil(y2))))

    def __getitem__(self, idx):
        if self.load_label:
            img_path, labels = self.samples[idx]
        else:
            img_path = self.samples[idx]
            labels = None

        pil = Image.open(img_path)

        if self.rotate_left_90:
            pil = pil.transpose(Image.ROTATE_90)

        # ✅ NEW: crop if detection enabled
        if self.load_detection_model:
            pil = self._crop_with_detection(pil, img_path)

        # convert to desired mode for recognition
        pil = pil.convert("RGB" if self.rgb else "L")
        img = self.transform(pil)

        out = [img]

        if self.load_label:
            text = labels[0] if (labels is not None and len(labels) > 0) else ""
            out.append(text)

        if self.return_filename:
            out.append(os.path.basename(img_path))

        return out[0] if len(out) == 1 else tuple(out)
