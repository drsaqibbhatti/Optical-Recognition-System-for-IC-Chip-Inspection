# train_dbnet.py
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Any, Iterable, Set

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


from utils.DetRecDataloaderV2 import BoxTextDataset
from utils.ConvNextOCR_helper_trainV2 import run_recognition_epoch, run_detection_epoch, initialize_detection, \
    initialize_recognition, RecognitionCropDataset, RecognitionDirectDataset, collect_rec_entries, TrainerConfig, parse_ids_to_train, EarlyStopper


###################################################

# ============================================================
# USER CONTROL: Train only certain ids
#   None  -> train ALL ids (0,1,2,...)
#   int   -> train ONLY that id
#   list/set/tuple -> train ONLY those ids
# ============================================================

#=========================MUTIPLE TEXTS in the image ==================================
# For MUTIPLE TEXTS in the image 
# this is the format

    # "filename": "10.jpg",
    # "pairs": [
    #   {
    #     "id": 0,
    #     "box": [
    #       [
    #         1664.3181818181818,
    #         949.0
    #       ],
    #       [
    #         1943.863636363636,
    #         949.0
    #       ],
    #       [
    #         1943.863636363636,
    #         1119.4545454545455
    #       ],
    #       [
    #         1664.3181818181818,
    #         1119.4545454545455
    #       ]
    #     ],
    #     "label": "I01"
    #   }
    # ]
#===========================================================

#========================SINGLE TEXT in the image ==================================
# For SINGLE TEXT in the image
# this is the format

    # "filename": "1.jpg",
    # "label": "A1234567"
#===========================================================

Ids_to_train = None
Text_BoxDetection_training = True # Toggle joint text + box training vs recognition-only workflow
maintain_aspect_ratio = False # For recognition: preserve aspect ratio if there are multiple texts with varying aspect ratios in the dataset
# Make True "maintain aspect ratio" only if there are aspects radios varying a lot in the dataset, else False is fine

Pretrained_Backbone_OCR = False
Pretrained_Backbone_Fcmae = True



train_img = r"D:\Projects\Projects_Work\OCR\dataset\projects\250716_OCR\dongjuk_ring\cropped"
train_label = r"D:\Projects\Projects_Work\OCR\dataset\projects\250716_OCR\dongjuk_ring\cropped\combined_labels_poly_cropped.json"
val_img =  r"D:\Projects\Projects_Work\OCR\dataset\projects\250716_OCR\dongjuk_ring\cropped"
val_label = r"D:\Projects\Projects_Work\OCR\dataset\projects\250716_OCR\dongjuk_ring\cropped\combined_labels_poly_cropped.json"
base_dir = r"D:\Projects\Projects_Work\OCR\trainedModel\Text_Det_Rec\dongjukRing"

ocr_pretrained_ckpt = r"D:\hvs\Hyvsion_Projects\OCR\trainedModel\IIIT5K_PreTrain\run_10\Rec\best_E036_L1.869584_Rec.pth"
fcmae_backbone_ckpt = r"D:\Projects\Projects_Work\OCR\OCR_git\backbone\fcmae\convnextv2_nano_1k_224_fcmae.pt"


imgH=512 #Input image height for detection model
imgW=512 #Input image width for detection model
text_imgH=96 #Input image height for recognition model
text_imgW=320 #Input image width for recognition model
rotate_image_left = True  # Rotate training images left by 90 degrees
rotate_image_right = False  # Rotate training images right by 90 degrees
Joint_visualization = True  # Visualize detection + recognition jointly during training
Rec_visualize_training = True  # Visualize recognition training samples in real-time

# ============================================================
# Checkpoint Helpers
# ============================================================
def resume_recognition_from_checkpoint(rec_state: dict, ckpt_path: str, device: torch.device) -> None:
    if not ckpt_path:
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    model_state = ckpt.get("model")
    if model_state is None:
        print(f"[WARN] Recognition checkpoint {ckpt_path} missing 'model' state; skipping full load.")
        return

    missing, unexpected = rec_state["model"].load_state_dict(model_state, strict=False)
    print(f"Loaded OCR recognition checkpoint -> {Path(ckpt_path).name}")
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    if "optim" in ckpt:
        try:
            rec_state["optim"].load_state_dict(ckpt["optim"])
        except Exception as exc:
            print(f"[WARN] Failed to load optimizer state from {ckpt_path}: {exc}")

    if "scaler" in ckpt and "scaler" in rec_state:
        try:
            rec_state["scaler"].load_state_dict(ckpt["scaler"])
        except Exception as exc:
            print(f"[WARN] Failed to load AMP scaler state from {ckpt_path}: {exc}")

    if "converter" in ckpt and hasattr(rec_state["converter"], "characters"):
        chars = ckpt["converter"]
        rec_state["converter"].characters = chars
        rec_state["converter"].char_to_idx = {char: idx for idx, char in enumerate(chars)}

    rec_state["start_epoch"] = ckpt.get("epoch", 0)


# ============================================================
# MAIN
# ============================================================
def main():

    config = TrainerConfig(
        imgH=imgH,
        imgW=imgW,
        text_imgH=text_imgH,
        text_imgW=text_imgW,
        visualize_joint=Joint_visualization,
        rec_visualize_training=Rec_visualize_training,
        rec_pad_preserve_aspect=maintain_aspect_ratio, # where there are images with varying aspect ratios, preserve aspect ratio with padding
    )

    config.load_box_polygons = bool(Text_BoxDetection_training)
    if not Text_BoxDetection_training:
        config.visualize_joint = False
        config.imgH = config.text_imgH
        config.imgW = config.text_imgW

    ids_set = parse_ids_to_train(Ids_to_train)
    print("Ids_to_train =", Ids_to_train, "-> parsed:", ids_set)

    rec_resume_ckpt = None

    if Pretrained_Backbone_OCR and ocr_pretrained_ckpt:
        rec_resume_ckpt = ocr_pretrained_ckpt
        backbone_ckpt = None
        use_pretrained = False
    elif Pretrained_Backbone_Fcmae and fcmae_backbone_ckpt:
        backbone_ckpt = fcmae_backbone_ckpt
        use_pretrained = True
    else:
        backbone_ckpt = None
        use_pretrained = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    base_run_dir = Path(base_dir)

    # IMPORTANT: new dataloader must return pair_ids
    dataset = BoxTextDataset(
        img_dir=train_img,
        json_path=train_label,
        imgH=config.imgH,
        imgW=config.imgW,
        rgb=True,
        load_label=config.load_label,
        load_box_polygons=config.load_box_polygons,
        return_filename=False,
        return_pair_ids=bool(Text_BoxDetection_training),
        rotate_left=rotate_image_left,
        rotate_right=rotate_image_right,
    )

    val_dataset = None
    if val_img and val_label:
        val_dataset = BoxTextDataset(
            img_dir=val_img,
            json_path=val_label,
            imgH=config.imgH,
            imgW=config.imgW,
            rgb=True,
            load_label=config.load_label,
            load_box_polygons=config.load_box_polygons,
            return_filename=False,
            return_pair_ids=bool(Text_BoxDetection_training),
            rotate_left=rotate_image_left,
            rotate_right=rotate_image_right,
        )

    # ------------------------------
    # MODE 1: Det + Rec
    # ------------------------------
    if Text_BoxDetection_training and config.load_label and config.load_box_polygons:
        print("Training detection + recognition (ID-aware)")

        det_state = initialize_detection(
            dataset,
            config,
            device,
            backbone_ckpt,
            use_pretrained,
            base_run_dir,
            ids_set,
            val_dataset=val_dataset,
        )

        visual_context = None
        if config.visualize_joint:
            det_run_dir = det_state["run_dir"]
            visual_dir = Path(config.visualize_joint_dir) if config.visualize_joint_dir else det_run_dir / "joint_vis"
            visual_context = dict(enabled=True, dir=visual_dir, count=0,
                                  max=config.visualize_joint_max, crop_max=config.visualize_joint_crop_max)

        rec_entries = collect_rec_entries(dataset, config, ids_set, visual_context=visual_context)
        rec_val_dataset = None
        if val_dataset is not None:
            rec_val_entries = collect_rec_entries(val_dataset, config, ids_set, visual_context=None)
            if rec_val_entries:
                rec_val_dataset = RecognitionCropDataset(
                    rec_val_entries,
                    text_imgH=config.text_imgH,
                    text_imgW=config.text_imgW,
                    rgb=getattr(val_dataset, "rgb", True),
                    margin_min_ratio=config.rec_crop_margin_min,
                    margin_ratio=config.rec_crop_margin,
                    preserve_aspect=config.rec_pad_preserve_aspect,
                )
                print("rec_val_entries:", len(rec_val_entries))
                print("rec_val_dataset:", len(rec_val_dataset))
            else:
                print("[INFO] No recognition validation samples found after ID filtering.")

        if not rec_entries:
            print("No recognition samples found after ID filtering; skipping recognition training.")
            rec_state = None
        else:
            rec_dataset = RecognitionCropDataset(
                rec_entries,
                text_imgH=config.text_imgH,
                text_imgW=config.text_imgW,
                rgb=getattr(dataset, "rgb", True),
                margin_min_ratio=config.rec_crop_margin_min,
                margin_ratio=config.rec_crop_margin,
                preserve_aspect=config.rec_pad_preserve_aspect,
            )
            print("rec_entries:", len(rec_entries))
            print("rec_dataset:", len(rec_dataset))

            # show 5 samples
            for i in range(min(5, len(rec_dataset))):
                img, lab = rec_dataset[i]
                print(i, "label=", lab, "img shape=", tuple(img.shape), "min/max=", float(img.min()), float(img.max()))

            rec_state = initialize_recognition(
                rec_dataset,
                config,
                device,
                backbone_ckpt,
                use_pretrained,
                base_run_dir,
                run_root=det_state["run_root"],
                val_dataset=rec_val_dataset,
            )

            if rec_resume_ckpt:
                resume_recognition_from_checkpoint(rec_state, rec_resume_ckpt, device)






        det_stopper = EarlyStopper(patience=config.patience, min_delta=config.min_delta)
        rec_stopper = EarlyStopper(patience=config.patience, min_delta=config.min_delta)

        det_done = False
        rec_done = False if rec_state else True

        for epoch in range(1, config.epochs + 1):
            if not det_done:
                det_metrics = run_detection_epoch(det_state, config, epoch, config.epochs)
                monitor_det_loss = det_metrics.get("val_loss", det_metrics["loss"])
                if det_stopper.check(monitor_det_loss):
                    print(f"Early stopping DET at epoch {epoch}")
                    det_done = True

            if not rec_done and rec_state is not None:
                rec_metrics = run_recognition_epoch(rec_state, config, epoch, config.epochs)
                monitor_rec_loss = rec_metrics.get("val_loss") if rec_metrics.get("val_loss") is not None else rec_metrics["loss"]
                if rec_stopper.check(monitor_rec_loss):
                    print(f"Early stopping REC at epoch {epoch}")
                    rec_done = True

            if det_done and rec_done:
                print("Both models stopped early.")
                break

    # ------------------------------
    # MODE 2: Det only
    # ------------------------------
    elif Text_BoxDetection_training and config.load_box_polygons:
        print("Training detection only (ID-aware)")
        det_state = initialize_detection(
            dataset,
            config,
            device,
            backbone_ckpt,
            use_pretrained,
            base_run_dir,
            ids_set,
            val_dataset=val_dataset,
        )
        det_stopper = EarlyStopper(patience=config.patience, min_delta=config.min_delta)

        for epoch in range(1, config.epochs + 1):
            det_metrics = run_detection_epoch(det_state, config, epoch, config.epochs)
            monitor_det_loss = det_metrics.get("val_loss", det_metrics["loss"])
            if det_stopper.check(monitor_det_loss):
                print(f"Early stopping DET at epoch {epoch}")
                break

    # ------------------------------
    # MODE 3: Rec only (pre-cropped images, no polygons)
    # ------------------------------
    elif config.load_label:
        print("Training recognition only (direct images, polygons ignored)")

        rec_dataset = RecognitionDirectDataset(dataset)
        rec_val_dataset = RecognitionDirectDataset(val_dataset) if val_dataset is not None else None

        rec_state = initialize_recognition(
            rec_dataset,
            config,
            device,
            backbone_ckpt,
            use_pretrained,
            base_run_dir,
            val_dataset=rec_val_dataset,
        )

        if rec_resume_ckpt:
            resume_recognition_from_checkpoint(rec_state, rec_resume_ckpt, device)
        rec_stopper = EarlyStopper(patience=config.patience, min_delta=config.min_delta)

        for epoch in range(1, config.epochs + 1):
            rec_metrics = run_recognition_epoch(rec_state, config, epoch, config.epochs)
            monitor_rec_loss = rec_metrics.get("val_loss") if rec_metrics.get("val_loss") is not None else rec_metrics["loss"]
            if rec_stopper.check(monitor_rec_loss):
                print(f"Early stopping REC at epoch {epoch}")
                break
    else:
        raise ValueError("Dataset neither has labels nor polygons to train on.")


if __name__ == "__main__":
    main()
