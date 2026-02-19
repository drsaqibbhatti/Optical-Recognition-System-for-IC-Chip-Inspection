from pathlib import Path
import cv2
import numpy as np
from utils.DetRecDataloader import BoxTextDataset


def to_int_poly(points):
    """
    points: [[x,y], ...] float -> np.int32 shape (N,1,2) for cv2.polylines/fillPoly
    """
    arr = np.array(points, dtype=np.float32)
    arr = np.round(arr).astype(np.int32)
    return arr.reshape(-1, 1, 2)


def tensor_to_bgr(img_tensor, use_rgb):
    arr = img_tensor.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * 0.5) + 0.5  # undo normalization
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    if arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    if use_rgb:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def sanity_draw_polygons(
    combined_json,
    image_root,
    out_dir,
    imgH=32,
    imgW=100,
    rgb=True,
    rotate_left=False,
    rotate_right=False,
    make_mask=True,
    thickness=1,
    font_scale=0.4,
    num_images=None,
):
    combined_json = Path(combined_json)
    image_root = Path(image_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay_dir = out_dir / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = out_dir / "mask"
    if make_mask:
        mask_dir.mkdir(parents=True, exist_ok=True)

    dataset = BoxTextDataset(
        img_dir=image_root,
        json_path=combined_json,
        imgH=imgH,
        imgW=imgW,
        rgb=rgb,
        load_label=True,
        load_box_polygons=True,
        return_filename=True,
        rotate_left=False,
        rotate_right=False,
        num_images=num_images,
    )

    print(f"Loaded {len(dataset)} samples via DetBoxDataloader. Drawing polygons...")

    for sample in dataset:
        img_tensor, labels, polys, fname = sample

        overlay = tensor_to_bgr(img_tensor, rgb)
        if make_mask:
            mask = np.zeros((overlay.shape[0], overlay.shape[1]), dtype=np.uint8)

        labels = labels if isinstance(labels, list) else [labels]
        polys = polys or []

        for idx, poly in enumerate(polys):
            if not poly or len(poly) < 3:
                continue

            poly_i = to_int_poly(poly)
            cv2.polylines(overlay, [poly_i], isClosed=True, color=(0, 255, 0), thickness=thickness)

            if make_mask:
                cv2.fillPoly(mask, [poly_i], 255)

            if idx < len(labels):
                label = str(labels[idx])
                if label:
                    anchor = poly_i[0, 0]
                    text_pos = (int(anchor[0]), max(int(anchor[1]) - 4, 0))
                    cv2.putText(
                        overlay,
                        label,
                        text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 255),
                        max(1, thickness // 2),
                        cv2.LINE_AA,
                    )

        cv2.imwrite(str(overlay_dir / fname), overlay)

        if make_mask:
            cv2.imwrite(str(mask_dir / (Path(fname).stem + ".png")), mask)

    print(f"Saved overlays -> {overlay_dir}")
    if make_mask:
        print(f"Saved masks    -> {mask_dir}")


if __name__ == "__main__":
    # EDIT THESE PATHS
    COMBINED_JSON = r"D:\hvs\Hyvsion_Projects\OCR\dataset\projects\Dot_matrix_251120\combined_labels_poly.json"
    IMAGE_ROOT    = r"D:\hvs\Hyvsion_Projects\OCR\dataset\projects\Dot_matrix_251120"   # where images are
    OUT_DIR       = r"D:\hvs\Hyvsion_Projects\OCR\dataset\projects\Dot_matrix_251120\sanity_check"

    sanity_draw_polygons(
        combined_json=COMBINED_JSON,
        image_root=IMAGE_ROOT,
        out_dir=OUT_DIR,
        imgH=512,
        imgW=512,
        rgb=True,
        make_mask=True,
        thickness=1,
        font_scale=0.5,
    )
