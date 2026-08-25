#!/usr/bin/env python3
"""
cv_detect.py -- detection, tracking and Shakey-style labelling for CV Mode.

Three separable pieces, per the processing brief: a Detector interface with
no framework baked in, a Tracker that persists identity between detections,
and a renderer that draws the result over the line drawing.

The detector is deliberately pluggable and defaults to NullDetector, so the
whole path -- tracking, labelling, rendering -- runs and can be tested with
no model present at all. Dropping a model in later changes one config key,
not the pipeline.

Why detection does not run every frame: on the hardware this targets a
detector pass costs far more than the frame budget. The brief's own answer
is the right one -- run detection at a low cadence, let the tracker carry
positions between passes, and render every frame from tracker state. A
missed detection shows as a track that persists briefly, not a label that
strobes.

Coat pattern is not a detection class. No general model distinguishes a
tuxedo from a tabby -- that is a coat, not an object category. It is however
trivially separable by colour statistics on the crop the detector already
produced: a tuxedo is bimodal black-and-white with almost no saturation, a
tabby is mid-luminance and brown/orange. That runs in microseconds and needs
no training data.
"""

import time

import numpy as np
import cv2

# COCO indices that matter here. A general model gives these three; the
# labels are renamed to what the operator actually calls them.
COCO_LABELS = {
    0:  'human',
    15: 'cat',
#    24: 'handbag',      # helps find lost items
#    24: 'backpack',    # helps find lost items
#    56: 'chair',
#    57: 'couch',
#    59: 'bed',
#    76: 'scissors',    # helps find lost items
    77: 'teddy bear',
}

# Scene model registration (CV Mode Phase 3): a separate, wider class set
# used ONLY when detecting on a high-resolution reference photo at setup
# time. Never used by the live pipeline -- COCO_LABELS above is what the
# runtime detector still looks for on every frame, unchanged. Furniture
# classes get their own set specifically so registration can recognise a
# couch/chair/bed/table/plant without the live HUD ever trying to (and
# failing to, cheaply and repeatedly) reconfirm furniture on noisy video.
STATIC_SCENE_LABELS = {
    56: 'chair',
    57: 'couch',
    59: 'bed',
    60: 'dining table',
    58: 'potted plant',
}

def parse_class_filter(spec):
    """Turn a device.env CV_DETECT_CLASSES value into a {id: label} dict.

    Accepts COCO indices ("15,16"), names already known to COCO_LABELS
    ("cat,human"), or a mix. Empty/None means "use COCO_LABELS unchanged",
    so a node that has not set the key behaves exactly as before.

    This exists because narrowing what a node looks for was previously done
    by COMMENTING OUT entries in COCO_LABELS on the live machine. tina is
    configured that way right now -- cat only -- and no released version has
    ever shipped it, which means every package upgrade silently reverts a
    deliberate operator decision. Config survives upgrades; a source edit
    does not.
    """
    if not spec:
        return None
    by_name = {v.lower(): k for k, v in COCO_LABELS.items()}
    # names the operator may reasonably use that differ from our relabelling
    by_name.setdefault('person', 0)
    out = {}
    for tok in str(spec).split(','):
        tok = tok.strip()
        if not tok:
            continue
        if tok.isdigit():
            cid = int(tok)
            out[cid] = COCO_LABELS.get(cid, str(cid))
        elif tok.lower() in by_name:
            cid = by_name[tok.lower()]
            out[cid] = COCO_LABELS.get(cid, tok)
        else:
            print(f"CV DETECT: ignoring unknown class {tok!r} in "
                  f"CV_DETECT_CLASSES", flush=True)
    return out or None


LABEL_ALIAS = {
    'motion': 'motion',
}


def make_detector(kind, **kw):
    """Factory, so the pipeline never imports a specific implementation."""
    kind = (kind or 'null').strip().lower()
    if kind == 'motion':
        return MotionDetector(**{k: v for k, v in kw.items()
                                 if k in ('scale', 'threshold', 'min_area_frac',
                                          'dilate', 'max_boxes')}).load()
    if kind == 'onnx':
        return OnnxDetector(**{k: v for k, v in kw.items()
                               if k in ('model_path', 'size', 'conf', 'nms',
                                        'classes')}).load()
    return NullDetector().load()


class Detection:
    """Normalised detection, independent of whatever produced it."""

    __slots__ = ('cls', 'confidence', 'x', 'y', 'w', 'h', 'timestamp',
                 'fg_consistency', 'scene_consistency')

    def __init__(self, cls, confidence, x, y, w, h, timestamp=None,
                 fg_consistency=1.0, scene_consistency=1.0):
        self.cls = cls
        self.confidence = float(confidence)
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)
        self.timestamp = timestamp if timestamp is not None else time.time()
        # Foveal Layer (CV Mode Phase 1): fraction of this box's area that
        # overlaps the MOG2 foreground mask, 0.0-1.0. 1.0 (neutral) unless
        # cv_processor explicitly sets it -- a Detector that knows nothing
        # about the scene model, or the gate being off, must reproduce
        # today's evidence formula exactly, not a scaled-down version of it.
        self.fg_consistency = float(fg_consistency)
        # Reference-image scene model (CV Mode Phase 3): fraction of this
        # box's area that overlaps a registered STATIC region of a
        # DIFFERENT class -- "is this a plausible place for a subject,"
        # independent of fg_consistency's "did this change." 1.0 (neutral)
        # unless cv_processor sets it from a loaded region dictionary.
        self.scene_consistency = float(scene_consistency)

    @property
    def box(self):
        return (self.x, self.y, self.w, self.h)

    @property
    def centre(self):
        return (self.x + self.w // 2, self.y + self.h // 2)


class Detector:
    """Interface. Implementations must not import each other's frameworks."""

    def load(self):
        return self

    def detect(self, frame):
        raise NotImplementedError

    def close(self):
        pass


class NullDetector(Detector):
    """No model, no detections. The default.

    Exists so the tracker, labeller and renderer are exercised and testable
    before any model is chosen -- and so a node with no model file degrades
    to a plain line drawing rather than failing.
    """

    def detect(self, frame):
        return []


class MotionDetector(Detector):
    """Detector with no model: motion regions become boxes labelled 'motion'.

    Useful beyond testing. It exercises the whole path -- cadence, tracking,
    labelling, rendering -- on real video with nothing installed, and on a
    node too slow to run a real detector it still produces a usable HUD that
    says where something is happening, just not what it is.

    Frame differencing on a downscaled luminance image rather than MOG2: a
    background model has to be learned and re-learned after every lighting
    change, while a fixed camera watching for "what moved since a moment
    ago" needs neither the memory nor the warm-up.
    """

    def __init__(self, scale=0.25, threshold=18, min_area_frac=0.004,
                 dilate=9, max_boxes=6):
        self.scale = float(scale)
        self.threshold = int(threshold)
        self.min_area_frac = float(min_area_frac)
        self.dilate = int(dilate)
        self.max_boxes = int(max_boxes)
        self._prev = None

    def detect(self, frame):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)

        if self._prev is None or self._prev.shape != gray.shape:
            self._prev = gray
            return []

        diff = cv2.absdiff(gray, self._prev)
        self._prev = gray
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        if self.dilate > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.dilate | 1, self.dilate | 1))
            # Close first: a moving subject shows as scattered patches where
            # its texture happens to differ, and those belong to one object.
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            mask = cv2.dilate(mask, k)

        cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = gray.shape[0] * gray.shape[1]
        found = []
        for c in cont:
            area = cv2.contourArea(c)
            if area < self.min_area_frac * frame_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            inv = 1.0 / self.scale
            # Confidence from relative size: nothing here knows what the
            # object is, so the only honest signal is how much moved.
            conf = min(0.99, 0.4 + (area / frame_area) * 3.0)
            found.append(Detection('motion', conf, x * inv, y * inv,
                                   bw * inv, bh * inv))
        found.sort(key=lambda d: d.w * d.h, reverse=True)
        return found[:self.max_boxes]


class OnnxDetector(Detector):
    """Single-stage ONNX detector via cv2.dnn."""

    def __init__(self, model_path, size=640, conf=0.35, nms=0.45, classes=None):
        self.model_path = model_path
        self.size = int(size)
        self.conf = float(conf)
        self.nms = float(nms)
        self.classes = classes if classes is not None else COCO_LABELS
        self.net = None

    def load(self):
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return self

    def detect(self, frame):
        if self.net is None:
            return []

        h, w = frame.shape[:2]

        scale = min(self.size / w, self.size / h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))

        resized = cv2.resize(
            frame, (nw, nh), interpolation=cv2.INTER_LINEAR
        )

        canvas = np.zeros(
            (self.size, self.size, 3), dtype=frame.dtype
        )

        pad_x = (self.size - nw) // 2
        pad_y = (self.size - nh) // 2

        canvas[
            pad_y:pad_y + nh,
            pad_x:pad_x + nw
        ] = resized

        lab = cv2.cvtColor(canvas, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l_chan = clahe.apply(l_chan)
        detector_canvas = cv2.cvtColor(
            cv2.merge((l_chan, a_chan, b_chan)),
            cv2.COLOR_LAB2RGB,
        )

        detector_canvas = np.ascontiguousarray(
            detector_canvas, dtype=np.uint8
        )

        blob = cv2.dnn.blobFromImage(
            detector_canvas,
            scalefactor=1.0 / 255.0,
            size=(self.size, self.size),
            mean=(0.0, 0.0, 0.0),
            swapRB=False,
            crop=False,
            ddepth=cv2.CV_32F,
        )

        self.net.setInput(blob)
        out = np.squeeze(self.net.forward())

        if out.ndim != 2:
            return []

        if out.shape[0] < out.shape[1]:
            out = out.T


        boxes = []
        confs = []
        ids = []

        inv_scale = 1.0 / scale

        for row in out:
            if row.shape[0] < 6:
                continue

            scores = row[4:]
            cid = int(np.argmax(scores))
            score = float(scores[cid])

            if score < self.conf or cid not in self.classes:
                continue
            cx, cy, bw, bh = map(float, row[:4])

            x1 = (cx - bw / 2.0 - pad_x) * inv_scale
            y1 = (cy - bh / 2.0 - pad_y) * inv_scale
            x2 = (cx + bw / 2.0 - pad_x) * inv_scale
            y2 = (cy + bh / 2.0 - pad_y) * inv_scale

            x1 = max(0.0, min(float(w - 1), x1))
            y1 = max(0.0, min(float(h - 1), y1))
            x2 = max(0.0, min(float(w), x2))
            y2 = max(0.0, min(float(h), y2))

            bw_px = int(round(x2 - x1))
            bh_px = int(round(y2 - y1))

            if bw_px <= 0 or bh_px <= 0:
                continue

            boxes.append([
                int(round(x1)),
                int(round(y1)),
                bw_px,
                bh_px,
            ])
            confs.append(score)
            ids.append(cid)

        if not boxes:
            return []

        keep = cv2.dnn.NMSBoxes(
            boxes,
            confs,
            self.conf,
            self.nms,
        )

        if len(keep) == 0:
            return []

        dets = []

        for i in np.array(keep).flatten():
            i = int(i)
            x, y, bw_px, bh_px = boxes[i]

            dets.append(
                Detection(
                    self.classes[ids[i]],
                    confs[i],
                    x,
                    y,
                    bw_px,
                    bh_px,
                )
            )

        return dets

    def close(self):
        self.net = None

def classify_coat(frame, det):
    """Return the detector's original class unchanged.

    YOLO is authoritative. No heuristic post-classification is applied.
    """
    return det.cls


class Track:
    __slots__ = ('track_id', 'cls', 'confidence', 'box',
                 'first_seen', 'last_seen', 'state', 'misses',
                 'evidence', 'promoted')

    def __init__(self, track_id, det):
        self.track_id = track_id
        self.cls = det.cls
        self.confidence = det.confidence
        self.box = det.box
        self.first_seen = det.timestamp
        self.last_seen = det.timestamp
        self.state = 'new'
        self.misses = 0
        # Leaky-integrator evidence, not the last-seen confidence: a single
        # weak frame must not promote a track. See Tracker.update() for the
        # accumulation step.
        self.evidence = det.confidence
        self.promoted = False

    @property
    def centre(self):
        x, y, w, h = self.box
        return (x + w // 2, y + h // 2)


class Tracker:
    """Nearest-centre association. Deliberately simple.

    A Kalman/IOU tracker would be more accurate under occlusion, but the
    detector here runs at a low cadence on a fixed camera watching one or
    two slow subjects -- association by proximity is sufficient and costs
    nothing. Replaceable behind the same interface if that changes.

    Evidence accumulation (run-14): on hardware where a real detection
    scores 0.02-0.07 -- below any per-frame confidence floor that would
    also admit noise -- per-frame thresholding throws away the one signal
    that actually distinguishes a real subject from noise: recurrence.
    A weak detection landing in the same track pass after pass is real; a
    weak detection that never recurs in the same track is noise, and the
    nearest-centre association above already refuses to link detections
    that jumped somewhere else.

    Each track keeps a leaky-integrator score (`evidence`), not a moving
    average: on a match, `evidence = evidence * decay + confidence`, so
    repeated weak hits on the same track *sum* toward the promotion
    threshold rather than merely tracking the last confidence seen. On a
    miss, `evidence *= decay` only -- pure decay, no addition -- so a track
    that stops matching fades back out. The steady-state value for a
    constant confidence c recurring every pass is c / (1 - decay): with the
    default decay of 0.85 that is ~6.7x a single frame's score, which is
    exactly the "recurrence is the signal" property this exists for.

    `promoted` flips on once `evidence` crosses `promote_threshold` and
    back off if it later decays below it -- not a one-way latch -- so a
    track does not stay labelled after the subject leaves and the pass
    count needed to un-promote is the same order as to promote (no separate
    hysteresis band; kept simple per the brief, revisit if this flickers in
    practice).

    Foveal Layer (CV Mode Phase 1): `evidence_boost` scales each detection's
    contribution by its `fg_consistency` -- a detection whose box overlaps
    where the MOG2 scene model believes something is actually present
    accumulates evidence faster than one placed somewhere the model
    considers static background. With `evidence_boost=0.0` (the default) or
    `fg_consistency=1.0` (the value on every Detection until cv_processor
    sets it), the multiplier is exactly 1.0 and this is byte-identical to
    the original formula -- additive and backward compatible by
    construction, not just by intent.

    Reference-image scene model (CV Mode Phase 3): `scene_boost` is a
    second, independent multiplier on `scene_consistency`, deliberately not
    folded into `evidence_boost`/`fg_consistency`. MOG2-consistency answers
    "did this change"; scene-consistency answers "is this a plausible place
    for a subject to be" -- a cat sitting still on the couch can have LOW
    fg_consistency (settled, no longer changing) while having HIGH
    scene_consistency (it's on the couch), which is exactly the case this
    second term exists to catch. Same backward-compatible construction:
    `scene_boost=0.0` or `scene_consistency=1.0` reproduces the exact prior
    formula.
    """

    def __init__(self, max_misses=5, max_distance=180,
                 evidence_decay=0.85, promote_threshold=0.35,
                 evidence_boost=0.0, scene_boost=0.0):
        self.tracks = {}
        self._next_id = 1
        self.max_misses = int(max_misses)
        self.max_distance = int(max_distance)
        self.evidence_decay = float(evidence_decay)
        self.promote_threshold = float(promote_threshold)
        self.evidence_boost = float(evidence_boost)
        self.scene_boost = float(scene_boost)

    def update(self, detections):
        unmatched = dict(self.tracks)
        for det in detections:
            best, best_d = None, self.max_distance + 1
            cx, cy = det.centre
            for tid, tr in unmatched.items():
                if tr.cls != det.cls:
                    continue
                tx, ty = tr.centre
                d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = tid, d
            if best is not None:
                tr = self.tracks[best]
                tr.box = det.box
                tr.confidence = det.confidence
                tr.last_seen = det.timestamp
                tr.state = 'present'
                tr.misses = 0
                tr.evidence = tr.evidence * self.evidence_decay + det.confidence * (
                    1.0 + self.evidence_boost * det.fg_consistency
                        + self.scene_boost * det.scene_consistency)
                tr.promoted = tr.evidence >= self.promote_threshold
                unmatched.pop(best, None)
            else:
                tr = Track(self._next_id, det)
                self.tracks[self._next_id] = tr
                self._next_id += 1

        # Anything unmatched this pass is missing, then departed. Kept for a
        # few passes so a single failed detection does not blink a label out
        # and back with a new id. Evidence decays here too -- a track that
        # stops matching loses its promoted state at the same rate it would
        # have gained it, not instantly and not never.
        for tid, tr in list(unmatched.items()):
            tr.misses += 1
            tr.state = 'missing'
            tr.evidence *= self.evidence_decay
            tr.promoted = tr.evidence >= self.promote_threshold
            if tr.misses > self.max_misses:
                tr.state = 'departed'
                self.tracks.pop(tid, None)
        return list(self.tracks.values())


def draw_hud(canvas, tracks, ink=(220, 220, 220), capabilities=None):
    """SHAKey HUD: confidence-weighted, collision-aware target annotations."""
    out = canvas
    h, w = out.shape[:2]

    active = [
        tr for tr in tracks
        if tr.state != 'departed'
    ]

    visible = [
        tr for tr in active
        if LABEL_ALIAS.get(tr.cls, tr.cls).upper() != 'MOTION'
    ]

    # Confirmed and candidate are counted SEPARATELY. Until Phase 5 this
    # line counted every active track, so "DETECTING 3 TARGETS" was printed
    # identically whether those were three promoted entities or one entity
    # plus two sub-threshold coincidences the tracker had not committed to.
    # The boxes already distinguished the two (candidates draw as a thin
    # dim outline with no tag, below) -- the banner did not, which made the
    # one line a viewer actually reads the least honest thing on screen.
    confirmed = [tr for tr in visible if getattr(tr, 'promoted', True)]
    candidates = len(visible) - len(confirmed)
    count = len(confirmed)
    text = f"DETECTING {count} TARGET{'S' if count != 1 else ''}"
    if candidates:
        text += f" +{candidates} CANDIDATE{'S' if candidates != 1 else ''}"

    # macOS-light-inspired telemetry:
    # smaller, lighter, quieter, and using the same visual weight as the
    # floating target labels rather than the old heavy surveillance text.
    telemetry_ink = tuple(
        max(1, min(255, int(round(channel * 0.88))))
        for channel in ink
    )

    telemetry_font = cv2.FONT_HERSHEY_SIMPLEX
    telemetry_scale = 0.52
    telemetry_thickness = 1
    telemetry_spacing = 3

    tx = 18
    ty = h - 42

    # Right-aligned companion to the DETECTING banner: which CV tools are
    # actually running. Same ink, size and baseline so the two read as one
    # telemetry strip rather than two unrelated overlays.
    if capabilities:
        cap = str(capabilities)
        cap_w = 0
        for ch_ in cap:
            (cw_, _), _ = cv2.getTextSize(ch_, telemetry_font, telemetry_scale,
                                          telemetry_thickness)
            cap_w += cw_ + telemetry_spacing
        cx_ = max(18, w - 18 - cap_w)
        for ch_ in cap:
            (cw_, _), _ = cv2.getTextSize(ch_, telemetry_font, telemetry_scale,
                                          telemetry_thickness)
            cv2.putText(out, ch_, (cx_, ty), telemetry_font, telemetry_scale,
                        telemetry_ink, telemetry_thickness, cv2.LINE_AA)
            cx_ += cw_ + telemetry_spacing

    for char in text:
        (cw, ch), _ = cv2.getTextSize(
            char,
            telemetry_font,
            telemetry_scale,
            telemetry_thickness,
        )

        cv2.putText(
            out,
            char,
            (tx, ty),
            telemetry_font,
            telemetry_scale,
            telemetry_ink,
            telemetry_thickness,
            cv2.LINE_AA,
        )

        tx += cw + telemetry_spacing

    tag_font = cv2.FONT_HERSHEY_SIMPLEX
    tag_scale = 0.52
    tag_thickness = 2

    pad_x = 8
    pad_y = 6
    tag_gap = 12
    tag_height = 24

    hud_clear = (8, 8, min(w - 8, 430), 155)

    def rects_overlap(a, b, margin=4):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return not (
            ax2 + margin < bx1 or
            bx2 + margin < ax1 or
            ay2 + margin < by1 or
            by2 + margin < ay1
        )

    def clamp_rect(x1, y1, x2, y2):
        rw = x2 - x1
        rh = y2 - y1

        if rw > w - 16:
            x1, x2 = 8, w - 8
        else:
            x1 = max(8, min(w - 8 - rw, x1))
            x2 = x1 + rw

        if rh > h - 16:
            y1, y2 = 8, h - 8
        else:
            y1 = max(8, min(h - 8 - rh, y1))
            y2 = y1 + rh

        return (int(x1), int(y1), int(x2), int(y2))

    def box_anchor(box, side):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if side == 'top':
            return cx, y1
        if side == 'bottom':
            return cx, y2
        if side == 'left':
            return x1, cy
        return x2, cy

    def label_anchor(rect, side):
        x1, y1, x2, y2 = rect
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if side == 'top':
            return cx, y2
        if side == 'bottom':
            return cx, y1
        if side == 'left':
            return x2, cy
        return x1, cy

    targets = []
    # Evidence accumulated but not yet past promote_threshold: trackable,
    # not yet labelled -- motion-detector behaviour, already useful on its
    # own. Rendered as a plain outline below, not put through the tag
    # placement/collision system at all -- it has no text to place.
    plain_boxes = []

    for tr in visible:
        if not getattr(tr, 'promoted', True):
            bx, by, bw, bh = tr.box
            bx1 = max(4, min(w - 5, int(bx)))
            by1 = max(4, min(h - 5, int(by)))
            bx2 = max(bx1 + 2, min(w - 4, int(bx + bw)))
            by2 = max(by1 + 2, min(h - 4, int(by + bh)))
            plain_boxes.append((bx1, by1, bx2, by2))
            continue

        label = LABEL_ALIAS.get(tr.cls, tr.cls).upper()

        bx, by, bw, bh = tr.box

        bx1 = max(4, min(w - 5, int(bx)))
        by1 = max(4, min(h - 5, int(by)))
        bx2 = max(bx1 + 2, min(w - 4, int(bx + bw)))
        by2 = max(by1 + 2, min(h - 4, int(by + bh)))

        confidence = max(
            0.0,
            min(1.0, float(getattr(tr, 'confidence', 0.0)))
        )

        text = f"{label} {confidence:.2f}"

        (tw, th), baseline = cv2.getTextSize(
            text,
            tag_font,
            tag_scale,
            tag_thickness,
        )

        tag_w = tw + pad_x * 2
        tag_h = max(tag_height, th + baseline + pad_y * 2)

        targets.append({
            'track': tr,
            'label': text,
            'box': (bx1, by1, bx2, by2),
            'tag_w': tag_w,
            'tag_h': tag_h,
            'area': max(1, bw * bh),
            'confidence': confidence,
        })

    # Higher-confidence detections get first choice of clean placement.
    targets.sort(
        key=lambda t: (
            -t['confidence'],
            -t['area'],
            t['box'][1],
            t['box'][0],
        )
    )

    occupied = [hud_clear]
    side_order = ('top', 'right', 'bottom', 'left')

    for item in targets:
        bx1, by1, bx2, by2 = item['box']
        tw = item['tag_w']
        th = item['tag_h']

        candidates = []

        for side in side_order:
            if side == 'top':
                cx = (bx1 + bx2) // 2
                x1 = cx - tw // 2
                y1 = by1 - tag_gap - th
            elif side == 'bottom':
                cx = (bx1 + bx2) // 2
                x1 = cx - tw // 2
                y1 = by2 + tag_gap
            elif side == 'left':
                cy = (by1 + by2) // 2
                x1 = bx1 - tag_gap - tw
                y1 = cy - th // 2
            else:
                cy = (by1 + by2) // 2
                x1 = bx2 + tag_gap
                y1 = cy - th // 2

            rect = clamp_rect(x1, y1, x1 + tw, y1 + th)

            score = 0

            if rects_overlap(rect, item['box'], margin=3):
                score += 10000

            for other in occupied:
                if rects_overlap(rect, other, margin=5):
                    score += 1000

            if side in ('top', 'bottom'):
                score += abs(
                    ((rect[0] + rect[2]) // 2) -
                    ((bx1 + bx2) // 2)
                )
            else:
                score += abs(
                    ((rect[1] + rect[3]) // 2) -
                    ((by1 + by2) // 2)
                )

            candidates.append((score, side, rect))

        _, side, tag_rect = min(candidates, key=lambda x: x[0])

        item['side'] = side
        item['tag_rect'] = tag_rect
        occupied.append(tag_rect)

    def confidence_ink(confidence):
        # macOS-light-style HUD: confidence changes salience, but even
        # weak detections remain comfortably visible against the scene.
        t = max(0.0, min(1.0, confidence))

        # Gentle perceptual curve: avoid making low-confidence objects
        # visually disappear while still giving strong detections priority.
        t = t * t * (3.0 - 2.0 * t)

        # Light-theme luminance floor. The HUD should read as luminous
        # neutral UI rather than a dim surveillance overlay.
        floor = 0.68
        strength = floor + (1.0 - floor) * t

        return tuple(
            max(1, min(255, int(round(channel * strength))))
            for channel in ink
        )

    def confidence_thickness(confidence, minimum=1, maximum=3):
        t = max(0.0, min(1.0, confidence))
        return minimum + int(round((maximum - minimum) * t))

    for item in targets:
        bx1, by1, bx2, by2 = item['box']
        tag_rect = item['tag_rect']
        side = item['side']
        confidence = item['confidence']

        target_ink = confidence_ink(confidence)
        box_thickness = confidence_thickness(confidence, 1, 3)
        line_thickness = confidence_thickness(confidence, 1, 2)

        cv2.rectangle(
            out,
            (bx1, by1),
            (bx2, by2),
            target_ink,
            box_thickness,
            cv2.LINE_AA,
        )

        p1 = box_anchor(item['box'], side)
        p2 = label_anchor(tag_rect, side)

        cv2.line(
            out,
            p1,
            p2,
            target_ink,
            line_thickness,
            cv2.LINE_AA,
        )

        tx1, ty1, tx2, ty2 = tag_rect

        cv2.rectangle(
            out,
            (tx1, ty1),
            (tx2, ty2),
            (0, 0, 0),
            -1,
        )

        cv2.rectangle(
            out,
            (tx1, ty1),
            (tx2, ty2),
            target_ink,
            1,
            cv2.LINE_AA,
        )

        (tw, th), baseline = cv2.getTextSize(
            item['label'],
            tag_font,
            tag_scale,
            tag_thickness,
        )

        text_x = tx1 + (tx2 - tx1 - tw) // 2
        text_y = ty1 + (ty2 - ty1 + th) // 2

        cv2.putText(
            out,
            item['label'],
            (text_x, text_y),
            tag_font,
            tag_scale,
            target_ink,
            tag_thickness,
            cv2.LINE_AA,
        )

    # Below-threshold tracks: thin, dim outline only -- no tag, no
    # confidence number, no collision placement. Accumulated evidence
    # hasn't crossed the promotion threshold yet, so there is nothing
    # honest to label it with.
    plain_ink = tuple(max(1, int(round(c * 0.35))) for c in ink)
    for (bx1, by1, bx2, by2) in plain_boxes:
        cv2.rectangle(out, (bx1, by1), (bx2, by2), plain_ink, 1, cv2.LINE_AA)

    return out


def draw_scene_regions(canvas, regions, ink=(140, 140, 220)):
    """Static furniture labels from the reference-image scene model (CV
    Mode Phase 3) -- drawn directly from the region dictionary every frame,
    never from live inference. Deliberately distinct styling from
    draw_hud's live-tracked targets (a cooler, dimmer, thin-outline-only
    look, small caption instead of a confidence-scored tag) so a viewer can
    tell "known furniture, asserted at setup" from "something the tracker
    is actively watching right now" at a glance.
    """
    out = canvas
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1

    for r in regions:
        bx, by, bw, bh = r['box']
        x1 = max(2, min(w - 3, int(bx)))
        y1 = max(2, min(h - 3, int(by)))
        x2 = max(x1 + 2, min(w - 2, int(bx + bw)))
        y2 = max(y1 + 2, min(h - 2, int(by + bh)))

        cv2.rectangle(out, (x1, y1), (x2, y2), ink, 1, cv2.LINE_AA)

        label = str(r['label']).upper()
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        ty = y1 - 4 if y1 - 4 - th > 0 else y1 + th + 4
        cv2.putText(out, label, (x1, ty), font, scale, ink, thickness, cv2.LINE_AA)

    return out


