#!/usr/bin/env python3
"""
cv_scene.py -- reference-image scene model: registration and matching.

CV Mode Phase 3. Shared by cv_scene_register.py (one-shot CLI, run manually
after the PM hands off reference photos or against a node's own camera) and
broadcast-api's /api/scene-reregister endpoint (re-run live, matching a
fresh frame against the ALREADY-STORED reference photo). One implementation
of the matching pipeline, two callers, so the two never drift apart.

The dictionary this module helps build is ASSERTED, not observed: detecting
a couch in a reference photo and warping its box into camera coordinates is
a one-time claim about where things were when the photo was taken. Nothing
here re-verifies that claim from the live feed. The staleness check
(cv_processor.py, correlating cv2.BackgroundSubtractorMOG2's learned
background against the warped reference) catches a camera pivot. It does
not catch furniture moving -- that silently desyncs the dictionary from
reality and needs a manual re-run, not an automatic one.

Multi-view depth is intentionally not implemented here. See
estimate_baseline() -- it reports whether a pair of reference photos has
enough out-of-plane parallax to make real triangulation worthwhile, and
stops there. tina stays flat-registered (homography only); a camera that
can be deliberately repositioned between shots (tanzania) is the only
plausible candidate for real multi-view geometry, and even there this
module only measures and reports, it does not build a depth map.
"""

import os
import sys
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_detect as _cvd

# ORB, not SIFT: patent status is moot either way in this OpenCV build (both
# are present), but ORB is faster and this only ever runs a handful of times
# per node, never per frame -- speed doesn't matter, but there's no reason
# to pay more for it either. Binary (Hamming-distance) descriptors also
# pair naturally with BFMatcher's fast cross-check path.
_ORB_FEATURES = 2000
_RATIO_TEST = 0.75  # Lowe's ratio test threshold -- standard, not tuned here
_RANSAC_REPROJ_THRESHOLD = 5.0  # pixels, cv2.findHomography's own default order of magnitude


def _orb_keypoints(gray):
    orb = cv2.ORB_create(nfeatures=_ORB_FEATURES)
    kp, desc = orb.detectAndCompute(gray, None)
    return kp, desc


def _match_descriptors(desc_a, desc_b):
    """Hamming BFMatcher + Lowe's ratio test. Returns filtered (a_idx, b_idx) pairs."""
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(desc_a, desc_b, k=2)
    good = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < _RATIO_TEST * n.distance:
            good.append(m)
    return good


def estimate_acuity(gray):
    """Variance of the Laplacian -- how much fine detail an image actually
    carries. Same metric cv_processor.py already uses to decide when to
    switch to edge/sharpie mode, reused here so the two stay consistent.

    Typical measured values on this fleet: a phone reference photo scores
    ~900, tina's live frame ~36. That 25x gap is not cosmetic -- it is why
    ORB descriptors computed on the two do not correspond.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def match_acuity(sharp_gray, target_acuity, max_sigma=8.0, tol=0.15):
    """Blur `sharp_gray` until its acuity approximately matches
    `target_acuity`, and return (blurred, sigma_used).

    This is acuity adaptation, and it is a required stage of registration
    rather than a tuning trick: feature descriptors encode local gradient
    structure, so an image carrying 25x more fine detail than its match
    partner produces descriptors that simply do not correspond. Degrading
    the sharper observation to the acuity of the weaker one puts both
    channels on comparable footing before any matching is attempted.

    The biological framing is apt -- two eyes of differing acuity looking
    at one scene still have to agree on where things are, and the
    agreement happens at the coarser of the two acuities, not the finer.
    Meeting in the middle is what makes cross-device registration possible
    at all on a degraded sensor.

    Binary search on sigma rather than a linear sweep: acuity falls
    monotonically with blur radius, so this converges in ~6 iterations
    instead of scanning a grid.
    """
    if estimate_acuity(sharp_gray) <= target_acuity:
        return sharp_gray, 0.0        # already at or below target

    lo, hi = 0.0, max_sigma
    best, best_sigma = sharp_gray, 0.0
    for _ in range(8):
        mid = (lo + hi) / 2.0
        if mid <= 0.05:
            break
        cand = cv2.GaussianBlur(sharp_gray, (0, 0), mid)
        a = estimate_acuity(cand)
        best, best_sigma = cand, mid
        if abs(a - target_acuity) <= tol * target_acuity:
            break
        if a > target_acuity:
            lo = mid          # still too sharp -- blur harder
        else:
            hi = mid          # overshot -- back off
    return best, best_sigma


def _spread_frac(pts, shape):
    """Fraction of frame width/height the given points span.

    A RANSAC fit whose inliers all sit in a narrow sliver is a spurious
    fit: with enough noisy correspondences, some locally-consistent subset
    can always be found, and the resulting homography is confidently
    wrong. Measured on tina: a degenerate fit put 68 'inliers' inside a
    34px-wide column. Requiring real spatial spread is what separates a
    registration from a coincidence, and a confidently misplaced scene
    model is worse than none at all.
    """
    if len(pts) == 0:
        return 0.0, 0.0
    h, w = shape[:2]
    return (float(pts[:, 0].max() - pts[:, 0].min()) / w,
            float(pts[:, 1].max() - pts[:, 1].min()) / h)


# A homography is only accepted if its inliers span at least this fraction
# of the frame in BOTH axes. Deliberately coarse -- it exists to reject
# degenerate slivers, not to grade good fits.
_MIN_SPREAD_FRAC = 0.30
_MIN_INLIERS = 12


def _quad_ok(H, ref_shape, live_shape):
    """Does H map the reference rectangle to a sane quadrilateral?

    A homography can satisfy an inlier count and still be geometrically
    nonsense -- collapsing the reference to a sliver, folding it over
    itself, or projecting it far outside the frame. Measured on tina: a
    fit with 8 well-spread inliers mapped two of the reference's corners
    to within 80px of each other, i.e. a collapsed quad. Inlier counts
    alone do not catch this; checking the shape does.

    Requires the projected quad to be convex (no self-intersection or
    fold-over) and to cover a plausible fraction of the frame.
    """
    h, w = ref_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    try:
        proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    except cv2.error:
        return False, 'corner projection failed'
    if not np.all(np.isfinite(proj)):
        return False, 'non-finite corner projection'

    # Convexity: the cross products of consecutive edges must share a sign.
    # 2-D cross product written out: NumPy 2.x removed the 2-vector form
    # of np.cross, and calling it there raises rather than returning the
    # scalar z-component this needs.
    def _cross2(u, v):
        return float(u[0] * v[1] - u[1] * v[0])

    signs = []
    for i in range(4):
        a, b, c = proj[i], proj[(i + 1) % 4], proj[(i + 2) % 4]
        signs.append(np.sign(_cross2(b - a, c - b)))
    if len(set(s for s in signs if s != 0)) > 1:
        return False, 'projected quad is not convex (folded over)'

    area = abs(cv2.contourArea(proj.astype(np.float32)))
    frame_area = float(live_shape[0] * live_shape[1])
    frac = area / frame_area
    if frac < 0.10:
        return False, f'projected quad covers only {frac*100:.0f}% of frame'
    if frac > 25.0:
        return False, f'projected quad is {frac:.0f}x the frame (wild extrapolation)'
    return True, f'quad ok ({frac*100:.0f}% of frame)'


def _fit(ref_pts, live_pts, live_shape, ref_shape=None):
    """RANSAC homography + the quality signals that decide whether it is
    trustworthy. Shared by the single-frame and burst paths."""
    n = len(ref_pts)
    if n < 4:
        return None, 0.0, n, {'reason': 'too few correspondences', 'inliers': 0,
                              'spread': (0.0, 0.0)}
    rp = np.float32(ref_pts).reshape(-1, 1, 2)
    lp = np.float32(live_pts).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(rp, lp, cv2.RANSAC, _RANSAC_REPROJ_THRESHOLD)
    if H is None:
        return None, 0.0, n, {'reason': 'no homography fit', 'inliers': 0,
                              'spread': (0.0, 0.0)}
    inliers = int(mask.sum())
    ratio = inliers / n
    spread = _spread_frac(lp[mask.ravel() == 1].reshape(-1, 2), live_shape)
    info = {'inliers': inliers, 'spread': spread, 'reason': 'ok'}
    if inliers < _MIN_INLIERS:
        info['reason'] = f'only {inliers} inliers (need {_MIN_INLIERS})'
        return None, ratio, n, info
    if spread[0] < _MIN_SPREAD_FRAC or spread[1] < _MIN_SPREAD_FRAC:
        info['reason'] = (f'degenerate: inliers span only '
                          f'{spread[0]*100:.0f}%x{spread[1]*100:.0f}% of frame')
        return None, ratio, n, info
    if ref_shape is not None:
        ok, why = _quad_ok(H, ref_shape, live_shape)
        info['quad'] = why
        if not ok:
            info['reason'] = why
            return None, ratio, n, info
    return H, ratio, n, info


def compute_homography(ref_gray, live_gray, adapt_acuity=True):
    """Match reference features against a live camera frame and fit a
    homography mapping reference pixel coords -> live camera pixel coords.

    Returns (H, inlier_ratio, n_matches). H is None when the fit is
    missing OR fails the quality gates in _fit() -- callers must treat
    None as "do not register", never as "register anyway with low
    confidence".

    With adapt_acuity (default), the sharper of the two images is first
    degraded to the other's acuity -- see match_acuity(). Without it,
    cross-device registration against a degraded sensor essentially never
    succeeds.
    """
    H, ratio, n, _ = compute_homography_ex(ref_gray, live_gray, adapt_acuity)
    return H, ratio, n


def compute_homography_ex(ref_gray, live_gray, adapt_acuity=True):
    """compute_homography plus a diagnostics dict (inliers, spread, sigma,
    and why a fit was rejected) -- what the CLI and the re-register
    endpoint report to the operator."""
    ref_used, sigma = ref_gray, 0.0
    if adapt_acuity:
        ref_used, sigma = match_acuity(ref_gray, estimate_acuity(live_gray))

    kp_ref, desc_ref = _orb_keypoints(ref_used)
    kp_live, desc_live = _orb_keypoints(live_gray)
    matches = _match_descriptors(desc_ref, desc_live)

    ref_pts = [kp_ref[m.queryIdx].pt for m in matches]
    live_pts = [kp_live[m.trainIdx].pt for m in matches]
    H, ratio, n, info = _fit(ref_pts, live_pts, live_gray.shape, ref_used.shape)
    info['acuity_sigma'] = round(sigma, 2)
    info['acuity_ref'] = round(estimate_acuity(ref_gray), 1)
    info['acuity_live'] = round(estimate_acuity(live_gray), 1)
    return H, ratio, n, info


def compute_homography_burst(ref_gray, live_grays, adapt_acuity=True):
    """Fit ONE homography from correspondences pooled across several live
    frames.

    Rationale is this project's own standing principle -- recurrence is
    the signal, not any single frame's score. A degraded sensor yields
    only a handful of matches per frame, and which features survive varies
    frame to frame; pooling across a burst accumulates enough
    correspondences for RANSAC to work with, exactly as the evidence
    accumulator pools weak detections across passes.

    The camera is static, so pooling is geometrically valid: every frame
    shares one true homography to the reference, and stacking their
    correspondences samples that same relationship repeatedly. Frames with
    a moving subject contribute a few bad correspondences, which RANSAC
    rejects as outliers.

    Returns (H, inlier_ratio, n_pooled, info).
    """
    if not live_grays:
        return None, 0.0, 0, {'reason': 'no frames supplied'}

    ref_used, sigma = ref_gray, 0.0
    if adapt_acuity:
        target = float(np.median([estimate_acuity(g) for g in live_grays]))
        ref_used, sigma = match_acuity(ref_gray, target)

    kp_ref, desc_ref = _orb_keypoints(ref_used)

    # Accumulate per REFERENCE KEYPOINT, not per raw match. Pooling raw
    # matches across frames double-counts: the camera is static, so the
    # same reference keypoint matches in frame after frame, and a flat
    # pooled list hands RANSAC N copies of one correspondence which it
    # scores as N independent inliers. Measured on tina: 469 pooled
    # matches came from just 214 distinct reference keypoints (2.2x
    # inflation, single keypoints counted up to 14 times) -- enough to
    # turn a handful of coincidences into a passing inlier count on a
    # geometrically wrong fit.
    #
    # One vote per reference keypoint, its live position taken as the
    # median across the frames it appeared in (median, not mean, so a
    # single frame with a subject occluding that point cannot drag it),
    # is the honest version: it still buys real noise reduction on the
    # position estimate without inventing independent evidence.
    seen = {}
    per_frame = []
    for lg in live_grays:
        kp_l, desc_l = _orb_keypoints(lg)
        ms = _match_descriptors(desc_ref, desc_l)
        per_frame.append(len(ms))
        for m in ms:
            seen.setdefault(m.queryIdx, []).append(kp_l[m.trainIdx].pt)

    ref_pts, live_pts, supports = [], [], []
    for qidx, positions in seen.items():
        arr = np.asarray(positions, dtype=np.float32)
        ref_pts.append(kp_ref[qidx].pt)
        live_pts.append((float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))))
        supports.append(len(positions))

    H, ratio, n, info = _fit(ref_pts, live_pts, live_grays[0].shape, ref_used.shape)
    info.update({
        'acuity_sigma': round(sigma, 2),
        'frames': len(live_grays),
        'matches_per_frame_mean': round(sum(per_frame) / len(per_frame), 1),
        'raw_matches': sum(per_frame),
        'distinct_keypoints': len(ref_pts),
        'mean_support_frames': (round(sum(supports) / len(supports), 1)
                                if supports else 0),
    })
    return H, ratio, n, info


def warp_box(box, H):
    """Transform an axis-aligned (x, y, w, h) box through a homography.

    A general homography does not map a rectangle to a rectangle, so this
    transforms all four corners and returns the axis-aligned bounding box
    of the result -- the standard, honest approximation for a "2D is fine"
    registration (per the brief: "3D or 2D, whatever, get that part done").
    """
    x, y, w, h = box
    corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    x1, y1 = warped.min(axis=0)
    x2, y2 = warped.max(axis=0)
    return (int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1)))


def detect_static_objects(reference_frame, model_path, conf=0.25):
    """Run the detector on a reference photo at its own native resolution,
    against STATIC_SCENE_LABELS (couch/chair/bed/table/plant) rather than
    the live pipeline's COCO_LABELS (human/cat/teddy-bear) -- registration
    needs a wider class set than the live HUD ever does, and the two must
    never share state (this call constructs its own OnnxDetector instance).

    "Feeder" is not a COCO class -- stock YOLOv8n has no trained category
    for a cat feeder. This will simply never appear in the returned list;
    that gap is real and is not silently masked here.

    conf defaults higher than the live pipeline's CV_DETECT_CONF (which is
    tuned as low as 0.01 on tina specifically to catch weak, noisy, live
    detections) -- a clean, full-resolution, well-lit reference photo does
    not need that permissiveness, and a lower bar here would just risk
    mislabelling furniture.

    Returns a list of dicts: {cls, confidence, box} in reference-photo
    pixel coordinates -- NOT yet warped into camera space.
    """
    detector = _cvd.OnnxDetector(
        model_path=model_path, conf=conf, classes=_cvd.STATIC_SCENE_LABELS).load()
    dets = detector.detect(reference_frame)
    return [{'cls': d.cls, 'confidence': d.confidence, 'box': d.box} for d in dets]


def estimate_baseline(ref_gray_a, ref_gray_b):
    """Scope-only check (not implementation) for whether a pair of
    reference photos has enough out-of-plane parallax to make real
    multi-view depth worthwhile later. Does NOT build a depth map or
    triangulate anything -- reports a number and a plain yes/no.

    Method: match features between the two reference photos directly (not
    against a live frame) and fit a homography, same as compute_homography.
    If a single planar homography explains the matched points well (low
    reprojection residual), the two shots are consistent with either a
    genuinely flat scene or near-zero camera movement between them --
    either way, no usable baseline for depth. A high residual with many
    inliers means a single plane does NOT explain the geometry, which is
    the signature of real parallax from a real baseline shift.

    Returns a dict: {usable, n_matches, inlier_ratio, mean_reproj_error_px}.
    """
    kp_a, desc_a = _orb_keypoints(ref_gray_a)
    kp_b, desc_b = _orb_keypoints(ref_gray_b)
    matches = _match_descriptors(desc_a, desc_b)
    n_matches = len(matches)

    if n_matches < 8:
        return {'usable': False, 'n_matches': n_matches, 'inlier_ratio': 0.0,
                'mean_reproj_error_px': None,
                'reason': 'too few matched points between the two reference photos'}

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, _RANSAC_REPROJ_THRESHOLD)
    if H is None:
        return {'usable': False, 'n_matches': n_matches, 'inlier_ratio': 0.0,
                'mean_reproj_error_px': None,
                'reason': 'no homography fit between the two reference photos'}

    inliers = int(mask.sum())
    inlier_ratio = inliers / n_matches

    projected = cv2.perspectiveTransform(pts_a, H)
    errs = np.linalg.norm(projected.reshape(-1, 2) - pts_b.reshape(-1, 2), axis=1)
    mean_err = float(errs.mean())

    # A near-perfect planar fit (low residual, most points count as RANSAC
    # inliers under the same threshold used for the actual registration)
    # means one plane explains everything seen -- no usable baseline. This
    # threshold is deliberately coarse (order-of-magnitude, not tuned): the
    # brief asks this phase to report the finding, not calibrate a metric
    # nothing downstream consumes yet.
    usable = mean_err > (_RANSAC_REPROJ_THRESHOLD * 2) and inlier_ratio < 0.85
    return {
        'usable': usable,
        'n_matches': n_matches,
        'inlier_ratio': round(inlier_ratio, 3),
        'mean_reproj_error_px': round(mean_err, 2),
        'reason': (
            'meaningful out-of-plane parallax detected' if usable else
            'a single planar homography explains the matched points well -- '
            'two shots from nearly the same spot (or a genuinely flat scene) '
            'give zero usable baseline'
        ),
    }


def rewarp_regions(regions, H):
    """Re-warp each region's ORIGINAL reference-photo-space box (`ref_box`)
    through a NEW homography, replacing `box` (the camera-space box)
    in-place on a copy. Used by /api/scene-reregister: a re-register
    recomputes the homography from a fresh live frame but does not re-run
    detection on the reference photo (that would need YOLO on the
    detection thread's critical path for what should be a rare, one-shot
    operator action) -- re-warping the already-detected boxes with the new
    geometry is the cheap, correct alternative. This is exactly why
    build_region_dict below keeps `ref_box` around instead of discarding
    it once the first `box` is computed.
    """
    out = []
    for r in regions:
        r2 = dict(r)
        r2['box'] = list(warp_box(tuple(r['ref_box']), H))
        out.append(r2)
    return out


def build_region_dict(static_objects, H, source):
    """Warp each detected reference-photo object through H into camera
    coordinates and shape it into the persisted region-dictionary entry
    format: {label, box, ref_box, source_confidence, static}. `ref_box` is
    the original, un-warped reference-photo-space box -- kept so a later
    re-register (see rewarp_regions) can recompute `box` under a new
    homography without re-running detection.

    Every entry from this phase is static=True -- everything comes from a
    static reference photo. The field is carried explicitly anyway (not
    hardcoded away) so the schema does not need reshaping if a future,
    non-static-sourced region type is added -- the schema is generalised
    even though nothing populates the other case yet.
    """
    regions = []
    for obj in static_objects:
        regions.append({
            'label': obj['cls'],
            'box': list(warp_box(obj['box'], H)),
            'ref_box': list(obj['box']),
            'source_confidence': round(obj['confidence'], 4),
            'static': True,
        })
    return {
        'source': source,
        'built_at': time.time(),
        'regions': regions,
    }
