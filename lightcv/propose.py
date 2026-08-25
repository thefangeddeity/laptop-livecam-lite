"""Propose a map from evidence that has already been collected.

Nothing here is authored from scratch and nothing here is trusted. Every
sector this produces is marked `source='proposed'`, drawn dashed with a '?'
in the renderer, and expected to be dragged into place by the PM. Two
minutes of correction beats both pure authoring and pure inference -- but
only if the proposal is honest about being a proposal.

Three sources, in the order they contribute:

  1. OCCUPANCY (Phase 4). `cv_occupancy` already knows what never changes.
     The large stable region at the bottom of the frame is the floor, and
     that is the anchor everything else is positioned against.

  2. REFERENCE PHOTOS. YOLO at full resolution on a phone photo resolves
     couch / table / tv / potted plant cleanly -- which is exactly what it
     could never do on tina's live feed. The reference photo is used for
     WHAT and roughly WHERE; it is not registered to the live frame,
     because Phase 3b established that registration does not work here.

  3. REGIONS (Phase 5). The colour-mass segmentation was genuinely good at
     one thing: isolating the couch as a coherent region. That gives a
     sector OUTLINE where a detection box only gives an axis-aligned
     rectangle.

Floor heights are asserted from a class-keyed default table, never measured.
A fixed monocular camera cannot measure them, and a guessed number that
looks measured is worse than one that is openly a default.
"""

import os
import sys
import numpy as np
import cv2

from .world import World, Sector, DEFAULT_FLOOR_HEIGHT_CM
from .floor import FloorPlane

_HERE = os.path.dirname(os.path.abspath(__file__))
_CV = os.path.join(os.path.dirname(_HERE), 'cv')
if _CV not in sys.path:
    sys.path.insert(0, _CV)


def _lazy_cv():
    import cv_detect as d
    import cv_occupancy as o
    return d, o


# ---------------------------------------------------------------------------
# 1. floor, from occupancy
# ---------------------------------------------------------------------------

def floor_quad_from_occupancy(frames, bottom_frac=0.45, valid=None):
    """Find the large stable region at the bottom of the frame.

    Returns (quad_image_pts, diagnostics). The quad is the four corners of
    the floor region in IMAGE pixels, ordered near-left, near-right,
    far-right, far-left -- the order `FloorPlane` expects.

    Stability, not colour, is the signal: a floor is the thing that is still
    there in every frame. `cv_occupancy.occupancy_stats` already computes
    the per-cell variance that says so.
    """
    _, occ = _lazy_cv()
    med = occ.temporal_median(frames)
    h, w = med.shape[:2]

    # cv_occupancy returns fg_fraction / appearance_sd / stability on its
    # own coarse grid. "Floor" is the conjunction of the first and last:
    # nothing ever moves there, and it reliably looks like itself. Using
    # only one of the two admits a blown-out wall (stable, but also never
    # moved *because* it is saturated).
    stats = occ.occupancy_stats(frames, valid=valid)
    if stats is None:
        return None, {'reason': 'occupancy_stats returned nothing'}
    fg = stats['fg_fraction']
    stab = stats['stability']
    mask = ((fg <= 0.02) & (stab >= 0.85)).astype(np.uint8) * 255
    stable = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    band = np.zeros((h, w), np.uint8)
    band[int(h * (1.0 - bottom_frac)):, :] = 255
    cand = cv2.bitwise_and(stable, band)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))

    cont, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cont:
        return None, {'reason': 'no stable region in the lower frame'}
    big = max(cont, key=cv2.contourArea)
    area_frac = cv2.contourArea(big) / float(h * w)

    peri = cv2.arcLength(big, True)
    quad = None
    for eps in (0.02, 0.03, 0.05, 0.08):
        ap = cv2.approxPolyDP(big, eps * peri, True)
        if len(ap) == 4:
            quad = ap.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        box = cv2.minAreaRect(big)
        quad = cv2.boxPoints(box).astype(np.float32)

    quad = _order_floor_quad(quad)
    return quad, {'area_frac': round(float(area_frac), 4),
                  'from': 'occupancy-stability'}


def floor_candidates_from_colour(median_rgb, furniture_boxes=(), top_n=4,
                                 merge_dist=22.0, scale=0.5):
    """Rank large coherent colour masses as floor candidates.

    THIS EXISTS BECAUSE STABILITY DOES NOT WORK. Measured on tina over 240
    frames: 41.6% of occupancy cells qualify as stable and the largest
    stable component covers 30.6% of the frame -- essentially everything
    that is not the dead sensor edge. In a room where nothing moved during
    the observation, "what never changes" is the whole room, so it cannot
    separate floor from couch. (The signal that WOULD separate them is
    accumulated foot traffic -- where things actually stand over days --
    but that needs traffic, and is what `Observer` will accumulate once
    this runs live.)

    What still works is colour coherence: a floor is one large, uniformly
    coloured, contiguous mass. Candidates are returned RANKED, not chosen,
    because this genuinely cannot tell a wooden floor from a wide beige
    couch and should not pretend to. The PM picks one in the editor.
    """
    h, w = median_rgb.shape[:2]
    small = cv2.resize(median_rgb, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    low = cv2.bilateralFilter(small, 5, 40.0, 9.0)
    lab = cv2.cvtColor(low, cv2.COLOR_RGB2LAB)

    slic = getattr(getattr(cv2, 'ximgproc', None), 'createSuperpixelSLIC', None)
    if slic is None:
        return []
    sp = slic(lab, algorithm=cv2.ximgproc.SLICO, region_size=28)
    sp.iterate(4)
    labels = sp.getLabels().astype(np.int32)
    n = int(labels.max()) + 1
    flat = labels.reshape(-1)
    cnt = np.bincount(flat, minlength=n).astype(np.float32)
    cnt[cnt == 0] = 1.0
    labf = lab.reshape(-1, 3).astype(np.float32)
    mean = np.stack([np.bincount(flat, weights=labf[:, c], minlength=n) / cnt
                     for c in range(3)], axis=1)

    a = np.concatenate([labels[:, :-1].ravel(), labels[:-1, :].ravel()])
    b = np.concatenate([labels[:, 1:].ravel(), labels[1:, :].ravel()])
    m = a != b
    pairs = np.unique(np.stack([np.minimum(a[m], b[m]),
                                np.maximum(a[m], b[m])], 1), axis=0)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    d = np.linalg.norm(mean[pairs[:, 0]] - mean[pairs[:, 1]], axis=1)
    for i, j in pairs[d < merge_dist]:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    root = np.array([find(i) for i in range(n)], np.int32)
    _, root = np.unique(root, return_inverse=True)
    merged = root[labels]

    # Reject anything that is mostly inside a known furniture box, and the
    # dead sensor edge (near-black and invariant).
    fb = []
    for (bx, by, bw, bh) in furniture_boxes:
        fb.append((bx * scale, by * scale, bw * scale, bh * scale))

    out = []
    for lb in range(int(merged.max()) + 1):
        mask = (merged == lb).astype(np.uint8)
        area = int(mask.sum())
        if area < 0.02 * mask.size:
            continue
        ys, xs = np.nonzero(mask)
        mean_lum = float(lab[:, :, 0][mask > 0].mean())
        if mean_lum < 25:
            continue                        # dead sensor region
        cy = float(ys.mean()) / mask.shape[0]
        overlap = 0.0
        for (bx, by, bw, bh) in fb:
            inb = ((xs >= bx) & (xs <= bx + bw) &
                   (ys >= by) & (ys <= by + bh)).mean()
            overlap = max(overlap, float(inb))
        cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cont:
            continue
        big = max(cont, key=cv2.contourArea)
        peri = cv2.arcLength(big, True)
        quad = None
        for eps in (0.02, 0.04, 0.07, 0.10):
            ap = cv2.approxPolyDP(big, eps * peri, True)
            if len(ap) == 4:
                quad = ap.reshape(4, 2).astype(np.float32) / scale
                break
        if quad is None:
            quad = cv2.boxPoints(cv2.minAreaRect(big)).astype(np.float32) / scale
        out.append({
            'area_frac': round(area / float(mask.size), 4),
            'centroid_y_frac': round(cy, 3),
            'mean_luma': round(mean_lum, 1),
            'furniture_overlap': round(overlap, 3),
            'quad': [[float(p[0]), float(p[1])]
                     for p in _order_floor_quad(quad)],
        })
    # Prefer large, low in frame, and not explained by a furniture box.
    out.sort(key=lambda r: -(r['area_frac'] * (1.0 - r['furniture_overlap'])
                             * (0.5 + r['centroid_y_frac'])))
    return out[:top_n]


def _order_floor_quad(q):
    """near-left, near-right, far-right, far-left (image y grows downward,
    so 'near' is the larger y)."""
    q = np.asarray(q, np.float32).reshape(4, 2)
    order = q[np.argsort(-q[:, 1])]
    near, far = order[:2], order[2:]
    near = near[np.argsort(near[:, 0])]
    far = far[np.argsort(far[:, 0])]
    return np.array([near[0], near[1], far[1], far[0]], np.float32)


# ---------------------------------------------------------------------------
# 2. furniture, from a reference photo
# ---------------------------------------------------------------------------

def furniture_from_reference(photo_path, conf=0.25, max_side=1600):
    """Full-resolution detection on a reference photograph.

    Uses the wider STATIC_SCENE_LABELS set, not the live class filter: the
    point is to find couch/table/plant/tv, which the live pipeline
    deliberately never looks for.
    """
    d, _ = _lazy_cv()
    img = cv2.imread(photo_path)
    if img is None:
        raise FileNotFoundError(photo_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / float(max(h, w))
        img = cv2.resize(img, (int(w * s), int(h * s)),
                         interpolation=cv2.INTER_AREA)

    labels = dict(d.STATIC_SCENE_LABELS)
    labels[62] = 'tv'
    model = _find_model()
    det = d.OnnxDetector(model, conf=conf, classes=labels).load()
    out = det.detect(np.ascontiguousarray(img))
    return [{'label': x.cls, 'conf': round(x.confidence, 3),
             'box': [x.x, x.y, x.w, x.h]} for x in out], img.shape[:2]


def _find_model():
    for p in (os.path.join(_CV, 'models', 'yolov8n.onnx'),
              '/usr/share/hls-livecam-server/models/candidates/yolov8n.onnx',
              os.path.join(_CV, 'yolov8n.onnx')):
        if os.path.exists(p):
            return p
    raise FileNotFoundError('no yolov8n.onnx found')


# ---------------------------------------------------------------------------
# 3. sector outlines, from the regions segmentation
# ---------------------------------------------------------------------------

def region_outline(frame_rgb, box, merge_dist=20.0, scale=0.5):
    """Colour-coherent outline of whatever occupies `box`.

    A detection box is axis-aligned and always wrong at the corners. The
    Phase 5 region segmentation gives the actual silhouette, which is what a
    sector polygon should be. Falls back to the box if segmentation finds
    nothing coherent.
    """
    x, y, w, h = [int(v) for v in box]
    H, W = frame_rgb.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return _box_poly(box)
    crop = frame_rgb[y0:y1, x0:x1]

    small = cv2.resize(crop, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    low = cv2.bilateralFilter(small, 5, 40.0, 9.0)
    lab = cv2.cvtColor(low, cv2.COLOR_RGB2LAB)

    slic = getattr(getattr(cv2, 'ximgproc', None), 'createSuperpixelSLIC', None)
    if slic is None:
        return _box_poly(box)
    sp = slic(lab, algorithm=cv2.ximgproc.SLICO, region_size=20)
    sp.iterate(4)
    labels = sp.getLabels().astype(np.int32)
    n = int(labels.max()) + 1
    flat = labels.reshape(-1)
    cnt = np.bincount(flat, minlength=n).astype(np.float32)
    cnt[cnt == 0] = 1.0
    labf = lab.reshape(-1, 3).astype(np.float32)
    mean = np.stack([np.bincount(flat, weights=labf[:, c], minlength=n) / cnt
                     for c in range(3)], axis=1)

    # the mass occupying the centre of the box is the object of interest
    ch, cw = labels.shape
    centre = labels[ch // 3: 2 * ch // 3, cw // 3: 2 * cw // 3]
    if centre.size == 0:
        return _box_poly(box)
    seed = int(np.bincount(centre.reshape(-1), minlength=n).argmax())
    keep = np.linalg.norm(mean - mean[seed], axis=1) < merge_dist
    mask = keep[labels].astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cont:
        return _box_poly(box)
    big = max(cont, key=cv2.contourArea)
    if cv2.contourArea(big) < 0.08 * mask.size:
        return _box_poly(box)
    ap = cv2.approxPolyDP(big, 0.02 * cv2.arcLength(big, True), True)
    pts = ap.reshape(-1, 2).astype(np.float32) / scale
    pts[:, 0] += x0
    pts[:, 1] += y0
    return [(float(a), float(b)) for a, b in pts]


def _box_poly(box):
    x, y, w, h = box
    return [(float(x), float(y)), (float(x + w), float(y)),
            (float(x + w), float(y + h)), (float(x), float(y + h))]


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def propose_world(floor_quad, floor_map_pts, furniture, frame_rgb=None,
                  camera_height_cm=None, path=None):
    """Assemble a proposed World.

    `floor_map_pts` is the operator's assertion of the floor quad's real
    dimensions in cm -- the one number that cannot be inferred. Everything
    else follows from it.
    """
    w = World(path=path)
    fp = FloorPlane.from_correspondences(floor_quad, floor_map_pts,
                                         camera_height_cm=camera_height_cm)
    w.homography = fp.H.tolist()
    w.homography_quality = fp.quality

    w.add_sector(Sector([tuple(p) for p in floor_map_pts], 0.0, 'floor',
                        source='proposed', confidence=1.0))

    for f in furniture:
        label = f['label']
        poly_img = (region_outline(frame_rgb, f['box'])
                    if frame_rgb is not None else _box_poly(f['box']))
        # A sector's footprint is where it MEETS THE FLOOR. Projecting the
        # whole silhouette through the floor homography would smear the
        # couch's back rest across the room, so only the lower edge of the
        # outline is used as the contact line and the footprint is built
        # from it.
        poly_img = _footprint_edge(poly_img)
        below = [p for p in poly_img if fp.is_below_horizon(*p)]
        if len(below) < 3:
            continue
        mp = fp.image_to_map(below)
        h = DEFAULT_FLOOR_HEIGHT_CM.get(label, 0.0)
        w.add_sector(Sector([(float(a), float(b)) for a, b in mp], h, label,
                            source='proposed', confidence=f.get('conf', 0.0)))
    return w, fp


def _footprint_edge(poly, keep_frac=0.35):
    """Keep the lower (nearer) part of an outline -- the contact line."""
    ys = [p[1] for p in poly]
    if not ys:
        return poly
    lo, hi = min(ys), max(ys)
    cut = hi - (hi - lo) * keep_frac
    lower = [p for p in poly if p[1] >= cut]
    return lower if len(lower) >= 3 else poly
