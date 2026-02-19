# db_targets.py
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pyclipper
except Exception:
    pyclipper = None


def _clip_poly(poly, W, H):
    # poly: [[x,y], ...]
    out = []
    for x, y in poly:
        x = float(x); y = float(y)
        x = max(0.0, min(x, W - 1.0))
        y = max(0.0, min(y, H - 1.0))
        out.append([x, y])
    return out


def _poly_area(poly):
    # shoelace
    pts = np.array(poly, dtype=np.float32)
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]; y = pts[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _poly_perimeter(poly):
    pts = np.array(poly, dtype=np.float32)
    if pts.shape[0] < 2:
        return 0.0
    d = pts - np.roll(pts, -1, axis=0)
    return float(np.sqrt((d * d).sum(axis=1)).sum())


def _fill_poly(mask, poly, value=1):
    # mask: (H,W) uint8
    if cv2 is None:
        raise RuntimeError("opencv-python required for polygon rasterization.")
    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], value)


def _shrink_poly(poly, shrink_ratio=0.4):
    """
    DBNet-style shrink:
      distance = area*(1-r^2) / perimeter
      offset inward by distance
    """
    if pyclipper is None:
        return None

    area = _poly_area(poly)
    peri = _poly_perimeter(poly)
    if peri <= 1e-6 or area <= 1e-6:
        return None

    # DB formula
    distance = area * (1.0 - shrink_ratio * shrink_ratio) / (peri + 1e-6)
    if distance <= 1.0:
        return None

    pco = pyclipper.PyclipperOffset()
    # pyclipper expects integer coords; scale up
    scale = 2.0
    pco.AddPath([(int(x * scale), int(y * scale)) for x, y in poly],
                pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrunk = pco.Execute(-distance * scale)

    if not shrunk:
        return None

    # choose the largest result
    shrunk = max(shrunk, key=lambda arr: abs(pyclipper.Area(arr)))
    shrunk = [[p[0] / scale, p[1] / scale] for p in shrunk]
    if len(shrunk) < 3:
        return None
    return shrunk, float(distance)


def build_db_maps(polys, H, W, shrink_ratio=0.4, min_area=10.0):
    """
    polys: list of polygons (each polygon: list[[x,y],...]) in resized image coords.
    returns:
      gt         [1,H,W] float32 (shrink region)
      mask       [1,H,W] float32 (ignore=0, use=1)
      thresh_map [1,H,W] float32
      thresh_mask[1,H,W] float32
    """
    gt = np.zeros((H, W), dtype=np.uint8)
    mask = np.ones((H, W), dtype=np.uint8)
    thresh_map = np.zeros((H, W), dtype=np.float32)
    thresh_mask = np.zeros((H, W), dtype=np.uint8)

    if cv2 is None:
        # You can still train BCE/Dice only if you modify loss to skip thresh.
        # But for proper DB targets, install opencv-python.
        return gt[None].astype(np.float32), mask[None].astype(np.float32), thresh_map[None], thresh_mask[None].astype(np.float32)

    for poly in polys or []:
        if not poly or len(poly) < 3:
            continue

        poly = _clip_poly(poly, W, H)
        area = _poly_area(poly)
        if area < min_area:
            # ignore tiny regions
            tmp = np.zeros((H, W), dtype=np.uint8)
            _fill_poly(tmp, poly, 1)
            mask[tmp == 1] = 0
            continue

        # --- GT shrink map ---
        shrunk = None
        shrink_dist = None
        if pyclipper is not None:
            ret = _shrink_poly(poly, shrink_ratio=shrink_ratio)
            if ret is not None:
                shrunk, shrink_dist = ret

        if shrunk is None:
            # fallback: use original polygon as gt (still trains)
            shrunk = poly
            shrink_dist = 0.0

        _fill_poly(gt, shrunk, 1)

        # --- thresh supervision (simple & stable) ---
        # We supervise inside the original polygon:
        poly_mask = np.zeros((H, W), dtype=np.uint8)
        _fill_poly(poly_mask, poly, 1)
        thresh_mask = np.maximum(thresh_mask, poly_mask)

        # distance to boundary inside polygon
        dist = cv2.distanceTransform(poly_mask, distanceType=cv2.DIST_L2, maskSize=5)

        # pick a scale for threshold map; if shrink_dist computed, use it; else use max dist
        if shrink_dist is None or shrink_dist <= 1e-6:
            denom = float(dist.max()) + 1e-6
        else:
            denom = float(shrink_dist) + 1e-6

        t = 1.0 - (dist / denom)
        t = np.clip(t, 0.0, 1.0)

        # keep max across polygons
        thresh_map = np.maximum(thresh_map, t.astype(np.float32) * poly_mask.astype(np.float32))

    return (
        gt[None].astype(np.float32),
        mask[None].astype(np.float32),
        thresh_map[None].astype(np.float32),
        thresh_mask[None].astype(np.float32),
    )
