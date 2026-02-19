import os
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class JsonTextDataset(Dataset):
    def __init__(
        self,
        img_dir,
        json_path=None,
        imgH=32,
        imgW=100,
        rgb=False,
        load_label=True,
        exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"),
        return_filename=False,
        recursive=True,
    ):
        self.img_dir = img_dir
        self.rgb = rgb
        self.load_label = load_label
        self.return_filename = return_filename

        norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if rgb else transforms.Normalize((0.5,), (0.5,))
        self.transform = transforms.Compose([
            transforms.Resize((imgH, imgW), Image.BICUBIC),
            transforms.ToTensor(),
            norm
        ])

        if self.load_label:
            if json_path is None:
                raise ValueError("json_path must be provided when load_label=True")

            with open(json_path, "r", encoding="utf-8") as f:
                anns = json.load(f)

            # store (full_image_path, label)
            self.samples = []
            for ann in anns:
                fp = os.path.join(self.img_dir, ann["filename"])
                self.samples.append((fp, ann["label"]))
        else:
            # Unlabeled: recursively collect images
            exts_set = {e.lower() for e in exts}
            p = Path(self.img_dir)

            if recursive:
                files = [str(f) for f in p.rglob("*")
                         if f.is_file() and f.suffix.lower() in exts_set]
            else:
                files = [str(f) for f in p.iterdir()
                         if f.is_file() and f.suffix.lower() in exts_set]

            files.sort()
            if len(files) == 0:
                raise ValueError(f"No images found under: {self.img_dir} (recursive={recursive}), exts={sorted(exts_set)}")

            self.samples = files

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.load_label:
            img_path, label = self.samples[idx]
        else:
            img_path = self.samples[idx]
            label = None

        img = Image.open(img_path).convert("RGB" if self.rgb else "L")
        img = self.transform(img)

        if self.load_label:
            if self.return_filename:
                return img, label, os.path.basename(img_path)
            return img, label

        if self.return_filename:
            return img, os.path.basename(img_path)
        return img
