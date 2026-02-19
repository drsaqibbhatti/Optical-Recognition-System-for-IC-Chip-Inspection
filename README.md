# Optical Recognition System for IC Chip Inspection (OCR + Vision Pipeline)

Industrial computer vision system for **IC chip inspection** that detects the target region and performs **OCR** to read printed characters/markings, designed for robust operation under real production conditions (lighting variation, blur, glare, small text).

Project page (demo + overview):  
https://drsaqibbhatti.com/projects/ic-chip-ocr.html

GitHub:  
https://github.com/drsaqibbhatti/Optical-Recognition-System-for-IC-Chip-Inspection

---

## Overview

### What this project does
- Detects the IC chip / marking region
- Enhances and normalizes the ROI (contrast/denoise/threshold)
- Performs OCR to read the printed code/ID
- Outputs:
  - predicted text
  - confidence (optional)
  - annotated visualization for verification

### Why it matters
Manual reading of chip markings is slow and error-prone. This pipeline enables:
- faster inspection
- consistent recognition quality
- easy integration into AOI/inspection workflows

---

## Key Features
- Vision-based ROI detection + OCR
- Robust preprocessing for difficult surfaces (glare/low contrast)
- Handles small text and fine strokes
- Export/deploy-friendly structure (can be integrated into a larger AOI system)
- Includes demo/visual results (see project page)

---

## Pipeline 
1. **Input**: image/frame from camera
2. **ROI Detection / Cropping**: locate the chip and text region
3. **Preprocessing**:
   - grayscale / denoise
   - contrast enhancement
   - adaptive threshold or morphological cleanup
   - optional deskew / perspective correction
4. **OCR**:
   - recognize alphanumeric markings
5. **Post-processing**:
   - regex/format rules (optional)
   - cleanup ambiguous characters (e.g., O/0, I/1)
6. **Output**:
   - recognized string + overlay image

---

## Tech Stack
- **Python**
- **OpenCV**
- **NumPy**
- OCR engine: custom OCR model with ConvNext Backbone

---
