import os
import json
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class BoxTextDataset(Dataset):
    """
    Combined JSON expected formats:
        # legacy flat entries
        [
            {"filename": "img1.png", "box": [[x,y],[x,y],[x,y],[x,y]], "label": ""},
            {"filename": "img1.png", "box": [[x,y],[x,y],[x,y],[x,y]], "label": ""}
        ]

        # grouped entries (preferred)
        [
            {
                "filename": "img1.png",
                "pairs": [
                    {"id": 0, "box": [[x,y],[x,y],[x,y],[x,y]], "label": "foo"},
                    {"id": 1, "box": [[x,y],[x,y],[x,y],[x,y]], "label": "bar"}
                ]
            }
        ]

    For each image, we group all boxes/labels and return:
        img,
        (optional) labels: list[str],
        (optional) boxes: list[list[list[float]]],
        (optional) pair ids: list[int|None],
        (optional) filename
    """

    def __init__(
        self,
        img_dir,
        json_path=None,
        imgH=32,
        imgW=100,
        rgb=False,
        load_label=False,
        load_box_polygons=False,
        exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"),
        return_filename=False,
        recursive=True,
        num_images=None,
        rotate_left=False,
        rotate_right=False,
        return_original_polygons=False,
        return_pair_ids=False,
    ):
        self.img_dir = img_dir
        self.rgb = rgb
        self.load_label = load_label
        self.load_box_polygons = load_box_polygons
        self.return_filename = return_filename
        self.num_images = num_images
        self.rotate_left = rotate_left
        self.rotate_right = rotate_right
        self.return_original_polygons = return_original_polygons
        self.return_pair_ids = return_pair_ids

        if self.rotate_left and self.rotate_right:
            raise ValueError("Only one of rotate_left or rotate_right can be True")
        if self.return_pair_ids and not self.load_box_polygons:
            raise ValueError("return_pair_ids=True requires load_box_polygons=True")

        norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if rgb else transforms.Normalize((0.5,), (0.5,))
        self.transform = transforms.Compose([
            transforms.Resize((imgH, imgW), Image.BICUBIC),
            transforms.ToTensor(),
            norm
        ])

        # If label or polygons requested -> json required
        if (self.load_label or self.load_box_polygons) and json_path is None:
            raise ValueError("json_path must be provided when load_label=True or load_box_polygons=True")

        if self.load_label or self.load_box_polygons:
            # ---- labeled (combined json) ----
            with open(json_path, "r", encoding="utf-8") as f:
                anns = json.load(f)

            if not isinstance(anns, list):
                raise ValueError("Expected combined json to be a list")

            grouped = {}
            for ann in anns:
                if not isinstance(ann, dict):
                    continue

                raw_fname = ann.get("filename", "")
                fname = Path(raw_fname).name
                if not fname:
                    continue

                pairs = ann.get("pairs")
                if isinstance(pairs, list):
                    pair_iter = pairs
                else:
                    pair_iter = [ann]

                for pair in pair_iter:
                    if not isinstance(pair, dict):
                        continue

                    box = pair.get("box")
                    lab = pair.get("label", "")

                    if self.load_box_polygons and box is None:
                        continue

                    entry = grouped.setdefault(fname, {"labels": [], "boxes": [], "ids": []})
                    entry["labels"].append(lab)
                    if self.load_box_polygons:
                        entry["boxes"].append(box)
                    entry["ids"].append(pair.get("id"))

            self.samples = []
            for fname, pack in grouped.items():
                img_path = os.path.join(self.img_dir, fname)
                boxes = pack["boxes"] if self.load_box_polygons else None
                if self.return_pair_ids:
                    self.samples.append((img_path, pack["labels"], boxes, pack["ids"]))
                else:
                    self.samples.append((img_path, pack["labels"], boxes))

            # stable ordering
            self.samples.sort(key=lambda x: os.path.basename(x[0]))

        else:
            # ---- unlabeled: just images ----
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
            # random subset without replacement; reseeded externally if deterministic subset desired
            self.samples = random.sample(self.samples, take) if take > 0 else []

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _rotate_polygons(polygons, width, height, direction):
        if not polygons:
            return polygons

        rotated = []
        for poly in polygons:
            if not poly:
                rotated.append(poly)
                continue

            rotated_poly = []
            for pt in poly:
                if pt is None or len(pt) < 2:
                    continue
                x, y = pt[0], pt[1]
                if direction == "left":
                    rotated_poly.append([y, width - 1 - x])
                else:
                    rotated_poly.append([height - 1 - y, x])
            rotated.append(rotated_poly)

        return rotated

    def __getitem__(self, idx):
        pair_ids = None
        if self.load_label or self.load_box_polygons:
            sample = self.samples[idx]
            if len(sample) == 4:
                img_path, labels, boxes, pair_ids = sample
            else:
                img_path, labels, boxes = sample
        else:
            img_path = self.samples[idx]
            labels, boxes = None, None

        img = Image.open(img_path).convert("RGB" if self.rgb else "L")
        orig_w, orig_h = img.size

        rotated_boxes = boxes
        if self.rotate_left or self.rotate_right:
            direction = "left" if self.rotate_left else "right"
            if self.load_box_polygons and boxes is not None:
                rotated_boxes = self._rotate_polygons(boxes, orig_w, orig_h, direction)
            if direction == "left":
                img = img.transpose(Image.ROTATE_90)
            else:
                img = img.transpose(Image.ROTATE_270)
            orig_w, orig_h = img.size
        else:
            if self.load_box_polygons:
                rotated_boxes = boxes

        img = self.transform(img)

        scaled_boxes = None
        if self.load_box_polygons and rotated_boxes is not None:
            # resize polygons to match the resized image
            new_h, new_w = img.shape[-2], img.shape[-1]
            scale_x = new_w / float(orig_w) if orig_w else 0.0
            scale_y = new_h / float(orig_h) if orig_h else 0.0
            scaled_boxes = []
            for poly in rotated_boxes:
                if not poly:
                    scaled_boxes.append(poly)
                    continue
                scaled_poly = []
                for pt in poly:
                    if pt is None or len(pt) < 2:
                        continue
                    x, y = pt[0], pt[1]
                    scaled_poly.append([x * scale_x, y * scale_y])
                scaled_boxes.append(scaled_poly)

        out = [img]
        if self.load_label:
            label_list = labels if labels else []
            out.append(label_list)

        if self.load_box_polygons:
            out.append(scaled_boxes if scaled_boxes is not None else rotated_boxes)   # list[polygon]
            if self.return_original_polygons:
                out.append(rotated_boxes if rotated_boxes is not None else boxes)
        if self.return_pair_ids:
            out.append(pair_ids if pair_ids is not None else [])
        if self.return_filename:
            out.append(os.path.basename(img_path))

        return out[0] if len(out) == 1 else tuple(out)
