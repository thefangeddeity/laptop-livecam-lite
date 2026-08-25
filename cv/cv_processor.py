#!/usr/bin/env python3
"""
cv_processor.py -- stateful temporal CV pipeline for CV Mode.

RGB frame -> temporal buffer -> motion estimation (optical flow)
    -> motion-aware temporal denoise -> highlight rolloff -> tone/exposure
    correction -> controlled sharpening -> CV display frame

Stays in NumPy/OpenCV array form throughout -- no PNG/PIL round trip. That
cycle (NumPy -> bytes -> PNG -> PIL -> bytes -> NumPy) is what the cloak
render path in block_art.py does, and it measured ~190ms/frame on tanzania
against a 66.7ms budget at 15fps. CV Mode cannot afford it.

Bounded history only (HISTORY frames). A live feed has no use for stale
frames -- broadcast-api's own drain loop already discards backlog and hands
this only the newest frame, so this never sees more than it can use.

The camera is fixed (not panning), so "alignment" between frames is the
identity for the static background -- optical flow here locates *where*
motion is, to gate blending, not to warp frames into registration.

Performance note: an earlier version of this module did the motion-weighted
blend with plain NumPy float32 broadcasting (frame.astype(float32) * alpha
+ ...). It was correct but measured 130-250ms/frame on tanzania -- 3x+ over
budget. The cost was the manual float32 elementwise math, not OpenCV or the
optical flow call. Rewritten below to use a binary (not continuous) motion
mask with OpenCV's native fused ops (addWeighted, copyTo), which stay in
uint8 and run through OpenCV's own multithreaded/SIMD paths. See the
profiling harness this shipped alongside for the before/after numbers.

Per-camera tuning: every constant below that affects visible output is
read from device.env, not hardcoded -- tanzania and tina have different
cameras (and very different CPUs) and do not share good default values.
Passing no denv (or a denv missing these keys) reproduces the exact
pre-tuning defaults, so this is a no-op change for any node that hasn't
added the new keys yet.
"""

import json
import os
import threading
import time
from collections import deque

import numpy as np
import cv2

try:
    import cv_detect as _cvd
except Exception:      # detection is optional; the pipeline runs without it
    _cvd = None

try:
    import cv_notify as _cvn
except Exception:      # notification is optional; absence disables it
    _cvn = None

try:
    import cv_scene as _cvs
except Exception:      # scene model is optional; absence disables it
    _cvs = None

try:
    import cv_persist as _cvp
except Exception:      # persistence is optional; absence disables it
    _cvp = None

try:
    import cv_occupancy as _cvo
except Exception:      # occupancy/grid tooling is optional
    _cvo = None

# Flow is computed on a downscaled grayscale pair purely to build a coarse
# motion mask -- a full-resolution dense flow field is not needed to answer
# "is this pixel neighborhood moving," and it costs several times more.
_FLOW_SCALE = 0.25

# Detection thread poll interval (CV Mode Phase 2). Deliberately fast and
# NOT tied to CV_DETECT_HZ: this only decides how quickly a new display
# frame becomes available to the accumulation window, and costs a lock
# acquire plus a tuple deref -- effectively free. CV_DETECT_HZ instead
# throttles how often a completed detect() forward pass is allowed to fire
# (see CVProcessor._detection_step's _last_detect_end check), which is the
# actual expensive operation this phase is budgeting.
_DETECT_POLL_INTERVAL_S = 0.05

_DEFAULTS = {
    'CV_MOTION_THRESHOLD':    1.5,   # flow px/frame (at _FLOW_SCALE res) counted as "moving"
    'CV_CLAHE_CLIP':          2.0,   # cv2.createCLAHE clipLimit
    'CV_CLAHE_TILES':         8,     # cv2.createCLAHE tileGridSize, square (N x N)
    'CV_UNSHARP_AMOUNT':      0.6,   # unsharp mask strength
    'CV_UNSHARP_SIGMA':       1.2,   # unsharp mask Gaussian blur sigma ("radius")
    'CV_HIGHLIGHT_THRESHOLD': 200,   # L channel (0-255) above which rolloff engages
    'CV_HIGHLIGHT_CEILING':   255,   # 255 = disabled; <255 caps the L channel there, freeing
                                      # numeric headroom below white for CLAHE to work with
    'CV_HIGHLIGHT_GAMMA':     1.0,   # curve shape of the compression; 1.0 = linear
    'CV_SHADOW_LIFT':         0.0,   # 0 = off. Raises dark values so detail in
                                      # shadow survives; a dark subject against a
                                      # bright surface is otherwise crushed to a
                                      # silhouette while the rolloff only helps
                                      # the bright end.
    'CV_SHADOW_RANGE':        110,   # L value below which the lift applies

    # Adaptive edge mode. Below a sharpness threshold, enhancement stops
    # helping -- CLAHE on mush is still mush -- so the pipeline switches to
    # drawing the structure it can still find. Threshold-driven and
    # automatic, so the same config behaves correctly on a sharp camera and
    # a soft one without per-install tuning.
    'CV_EDGE_ENABLED':        1,     # 0 disables the stage entirely
    'CV_EDGE_SHARPNESS_MIN':  35.0,  # variance-of-Laplacian below which edges engage
    'CV_EDGE_SHARPNESS_OFF':  70.0,  # ...and at/above which they are fully absent
    'CV_EDGE_STRENGTH':       0.75,  # opacity of the darkest strokes at full engagement
    'CV_EDGE_SIGMA':          1.0,   # base stroke width; scaled up as the image softens
    'CV_EDGE_SIGMA_MAX':      3.5,   # cap, so a very soft frame does not draw slabs
    'CV_EDGE_PERCENTILE':     99.0,  # DoG normalisation reference; lower = heavier ink
    'CV_EDGE_GAMMA':          0.65,  # <1 lifts mid-strength edges toward full ink

    # Edge style. 'overlay' inks strokes over the photo. 'sharpie' discards
    # the photo entirely and returns a line drawing on white.
    #
    # Sharpie exists because on a genuinely defocused lens there is no fine
    # detail to recover -- enhancement either does nothing or manufactures
    # halos. What such a frame still has is large regions of distinct
    # brightness, so the drawing traces region *boundaries* rather than
    # texture. Tracing texture is what makes a blurry frame come out as
    # scattered dashes instead of strokes.
    # 'overlay' inks strokes over the photo; 'sharpie' discards the photo
    # for a line drawing; 'xdog' is the extended difference-of-Gaussians
    # stylisation below -- it keeps TONE as well as edges, which is what
    # lets a soft sensor still show something like fur texture rather than
    # collapsing it to outline. 'regions' is the cel-shaded illustration
    # path (see _draw_regions): flat masses of the scene's OWN colour with
    # a restrained edge overlay, for a sensor that resolves large-scale
    # structure and colour but no fine texture at all.
    'CV_EDGE_STYLE':          'overlay',
    'CV_XDOG_SIGMA':          3.0,   # inner Gaussian. NOT a free parameter on a
                                      # compressed feed: at sigma 0.8 the DoG
                                      # peaks at 1-2px, which is exactly the
                                      # MJPEG block-edge scale, and measured
                                      # blockiness rose to 4.13 against a 3.33
                                      # source baseline -- the stage inks the
                                      # compressor's grid rather than the room.
                                      # At sigma 3 it sits above the 8-16px
                                      # block size and blockiness returns to
                                      # baseline (3.35).
    'CV_XDOG_K':              1.6,   # outer/inner sigma ratio (DoG standard)
    'CV_XDOG_TAU':            0.985, # how much of the outer Gaussian is
                                      # subtracted. ->1 sharpens the response
                                      # and thins strokes.
    'CV_XDOG_EPS':            0.30,  # ramp centre, in units of the
                                      # normalised DoG response (see
                                      # CV_XDOG_PERCENTILE) -- so it means
                                      # the same thing on any sensor
    'CV_XDOG_PERCENTILE':     99.0,  # normalisation reference for |DoG|
    'CV_XDOG_PHI':            2.5,   # ramp steepness; low = soft grey
                                      # gradations, high = hard black/white
    'CV_XDOG_TONE':           0.55,  # 0 = pure ink on white, 1 = keep the
                                      # toned photo untouched. In between
                                      # multiplies the ink over the photo,
                                      # which is the setting that preserves
                                      # coat texture.
    # --- 'regions' renderer (CV Mode Phase 5 §1) -------------------------
    # Cel-shaded illustration of the real scene. Every fill colour is the
    # MEAN OF ACTUAL PIXELS inside that region, sampled from a denoised
    # low-frequency copy. No palette, no colour transfer, no synthesis --
    # same anti-hallucination rule as the standing super-resolution ban,
    # applied to the renderer. See _draw_regions.
    'CV_REGION_SCALE':        0.5,   # working scale. The whole stage runs here
                                      # and is upscaled once at the end: region
                                      # geometry is large-scale by definition,
                                      # so full-resolution segmentation buys
                                      # nothing and costs ~4x.
    'CV_REGION_SIGMA_COLOR':  40.0,  # bilateral colour sigma. This is what
                                      # decides whether two adjacent shades
                                      # become one mass or two.
    'CV_REGION_SIGMA_SPACE':  9.0,   # bilateral spatial sigma
    'CV_REGION_BILATERAL_D':  5,     # bilateral window. The dominant cost of
                                      # the whole stage -- 9 roughly doubles
                                      # it for no visible gain at this scale.
    'CV_REGION_SIZE':         28,    # SLIC superpixel size, in working-scale
                                      # px. Smaller = finer fragments before
                                      # merging = better boundaries, more cost.
    'CV_REGION_ITERS':        4,     # SLIC refinement iterations. 10 is the
                                      # library default and is not worth 2.5x
                                      # the cost on a feed this soft.
    'CV_REGION_MERGE_DIST':   20.0,  # merge adjacent superpixels whose mean
                                      # LAB colours are closer than this. THE
                                      # knob that decides how illustrated the
                                      # result looks. Measured on tina's own
                                      # feed: 9 -> 127 regions (reads as a
                                      # mosaic, not an illustration), 20 -> ~40,
                                      # 28 -> 22, and by 36 the frame has
                                      # collapsed to 11 regions with the whole
                                      # centre one flat grey -- the merge-into-
                                      # one-blob failure the brief warned about.
                                      # 16-22 is the usable band on this sensor.
    'CV_REGION_BANDS':        10,    # luminance bands, fallback path only
                                      # (used when cv2.ximgproc is absent)

    # --- illumination-field correction (CV Mode Phase 5 §3) --------------
    # Corrects the SPATIAL exposure error a global operator structurally
    # cannot touch. Measured on tanzania's own frame (4x4 tile means):
    #
    #   raw                tile spread 142   blown 6.59%   dark 6.27%
    #   global gamma 0.7   tile spread 123   blown 9.09%   dark 0.00%
    #   CLAHE clip 2.0     tile spread 117   blown 5.63%   dark 10.86%
    #   illum field 0.7    tile spread  84   blown 0.10%   dark  0.08%
    #
    # Both global operators trade one end of the histogram for the other,
    # exactly as the brief predicted; the spatial field fixes both at once,
    # for ~24ms at 1280x720. That is why this is classical and not learned:
    # the measured result did not justify a network. See _apply_illumination.
    'CV_ILLUM_ENABLED':       0,     # off by default, per node
    'CV_ILLUM_STRENGTH':      0.7,   # 0 = no-op, 1 = flatten the field
                                      # completely (and flat is not the goal --
                                      # a room legitimately has bright and dark
                                      # parts, and erasing that reads as fake)
    'CV_ILLUM_SIGMA_FRAC':    0.12,  # illumination-field smoothness, as a
                                      # fraction of the long edge. Too small
                                      # and the "illumination" starts tracking
                                      # objects, which produces halos.
    'CV_ILLUM_SCALE':         0.25,  # the field is smooth by construction, so
                                      # it is estimated small and upsampled
    'CV_REGION_EDGE_ALPHA':   0.45,  # how darkly the boundary overlay inks.
                                      # 0 = pure flat fills, no linework
    'CV_REGION_FG_BOUNDARY':  1,     # add the MOG2 foreground outline to the
                                      # boundary map, so a moving subject gets
                                      # its own silhouette even when it is a
                                      # poor colour match against what it is
                                      # standing in front of
    'CV_SHARPIE_LEVELS':      4,     # luminance bands; more = concentric clutter
    'CV_SHARPIE_THICK':       3,     # stroke width in px
    'CV_SHARPIE_MIN_LEN':     140,   # discard contours shorter than this
    'CV_SHARPIE_SMOOTH':      13,    # median kernel; flattens speckle pre-banding
    'CV_SHARPIE_CLOSE':       7,     # morphological close/open, for unbroken strokes
    # Detection + Shakey HUD. Runs every Nth frame, never per frame: a
    # detector pass costs far more than the frame budget on this hardware.
    # The tracker carries boxes between passes and the HUD is drawn every
    # frame from tracker state, so labels persist smoothly instead of
    # strobing at the detection cadence.
    'CV_DETECT_ENABLED':      1,     # off by default
    'CV_DETECT_BACKEND':      'onnx',   # motion | onnx | null
    'CV_DETECT_INTERVAL':     8,     # run the detector every N frames
    'CV_DETECT_MODEL':        '/usr/share/hls-livecam-server/models/candidates/yolov8n.onnx',
    'CV_DETECT_CONF':         0.12,
    'CV_DETECT_MIN_AREA':     0.004, # motion backend: ignore blobs smaller than this
    'CV_DETECT_CLASSES':      '',    # '' = every class in COCO_LABELS.
                                      # Comma-separated ids or names ("cat",
                                      # "15,0") narrow what this node looks
                                      # for -- previously done by editing
                                      # COCO_LABELS on the live machine,
                                      # which every upgrade then reverted.
    'CV_HUD_ENABLED':         1,     # draw boxes/labels when detection is on

    # Track-level evidence accumulation (run-14). Per-frame confidence
    # thresholding throws away the signal that separates a real subject
    # from noise on hardware where a genuine detection scores 0.02-0.07:
    # recurrence. Evidence is a leaky integrator per track -- on a match,
    # evidence = evidence*decay + confidence (repeated weak hits *sum*
    # toward the threshold); on a miss, evidence *= decay (pure decay, so
    # a departed subject's track stops being labelled). A track only gets
    # a label + confidence number once evidence crosses the threshold;
    # below it, it is still tracked and drawn (a plain box), just
    # unlabelled -- motion-detector behaviour. Steady-state for a constant
    # confidence c recurring every pass is c/(1-decay): with the defaults
    # below, a genuine 0.05 recurring hit settles at ~0.42, comfortably
    # promotable, while a one-off 0.02 noise hit that never recurs decays
    # away before it gets close.
    'CV_TRACK_EVIDENCE_DECAY':    0.88,
    'CV_TRACK_PROMOTE_THRESHOLD': 0.30,

    'CV_SHARPIE_SCALE':       0.5,   # compute the drawing at this scale. Strokes are
                                      # thick by design, so the detail lost is detail
                                      # the drawing discards anyway -- and contour
                                      # finding plus morphology is superlinear in
                                      # pixel count, so this is the difference between
                                      # ~150ms and ~40ms.
    # Weight of the original (toned) photo blended back into the sharpie
    # line drawing. 0 = pure line drawing on white (this file's tracked
    # default, unchanged behaviour for any node that doesn't set this).
    # 1 = pure photo, sharpie strokes invisible. tina's established
    # baseline is 0.6 (60% original / 40% sharpie) -- device.env, not
    # hardcoded, since this is exactly as per-camera as CLAHE or unsharp.
    'CV_SHARPIE_BLEND':       0.0,

    # Enhancement on moving subjects. The pipeline skips denoise and sharpen
    # wherever the motion mask is set, on the reasoning that motion blur is
    # an optical fact rather than missing detail. That holds for a fast pan;
    # it is wrong for a pet ambling across a soft, dirty lens, where the blur
    # is mostly focus and the moving subject is the whole point of watching.
    # Enabled by default: on this camera the subject matters more than the
    # theoretical risk of sharpening genuine motion blur.
    'CV_SHARPEN_MOTION':      1,     # sharpen moving regions too
    'CV_DENOISE_MOTION':      0,     # but do NOT temporally blend them (ghosting)

    # Lens-artifact subtraction. Water spots and dirt on the lens are static,
    # sharp and high-contrast while the scene behind them is soft and moving.
    # That asymmetry is what makes them separable: sample across frames that
    # contain motion, and anything high-frequency that never moves is on the
    # glass, not in the room.
    'CV_ARTIFACT_ENABLED':    0,     # off by default; opt in per node
    'CV_ARTIFACT_SAMPLES':    30,    # frames sampled to learn the mask
    'CV_ARTIFACT_MIN_MOTION': 0.4,   # mean flow required for a frame to count
    'CV_ARTIFACT_THRESHOLD':  18,    # high-pass response counted as an artifact
    'CV_ARTIFACT_DILATE':     2,     # grow the mask so stroke edges are covered
    'CV_ARTIFACT_MAX_AREA':   0.04,  # refuse a mask covering more than this
                                      # fraction of frame -- that is a wrong
                                      # answer, not a very dirty lens

    # Foveal Layer (CV Mode Phase 1). MOG2 per-pixel Gaussian-mixture scene
    # model: cheap, C++-speed, and its Mahalanobis-distance threshold means
    # a pixel that is *always* noisy learns high variance and stops being
    # surprising -- noise gets absorbed into the model instead of firing
    # the gate every frame, which is why this beats frame-differencing on a
    # degraded sensor. CV_FOVEAL_ENABLED is only the value the pipeline
    # starts with; broadcast-api's live checkbox is the actual runtime
    # control from then on (see CVProcessor.set_foveal()).
    'CV_FOVEAL_ENABLED':      0,     # off by default; opt in per node
    'CV_MOG2_HISTORY':        2000,  # frames of memory. OpenCV's own default
                                      # (500) is ~33s at 15fps -- too short
                                      # for a cat that settles in for a nap;
                                      # 2000 is ~2.2min at 15fps.
    'CV_MOG2_VAR_THRESHOLD':  16.0,  # OpenCV's own default; Mahalanobis-
                                      # distance-squared cutoff for "moving"
    'CV_MOG2_DETECT_SHADOWS': 0,     # shadow pixels come back gray (127) in
                                      # the mask when on; costs extra per-
                                      # pixel work every frame for a signal
                                      # this gate does not currently use
    'CV_MOG2_SCALE':          0.25,  # MOG2 runs on a downscaled frame, same
                                      # reasoning and same default as optical
                                      # flow's _FLOW_SCALE: measured on
                                      # tanzania (i5-10210U), full-res 1280x720
                                      # MOG2 costs ~30-38ms/frame ALONE -- 45%+
                                      # of the entire 66.7ms 15fps budget, paid
                                      # every frame with no amortization,
                                      # unlike detection. At 0.25x the same
                                      # model costs ~3.6ms, and a coarse mask
                                      # is all a padded crop box needs.
    'CV_FOVEAL_DILATE':       9,     # grow the foreground mask before
                                      # contour-finding, same rationale as
                                      # MotionDetector's own dilate: joins a
                                      # subject broken into scattered patches
    'CV_FOVEAL_MIN_AREA':     0.004, # ignore contours smaller than this
                                      # fraction of frame -- mask noise, not
                                      # a subject
    'CV_FOVEAL_PAD':          0.25,  # pad the crop box by this fraction of
                                      # its own size on each side -- a tight
                                      # crop risks cutting off a part of the
                                      # subject that was not moving (a
                                      # cat's tail outside the flagged blob)
    'CV_FOVEAL_EVIDENCE_BOOST': 0.0, # extra evidence weight for a detection
                                      # whose box overlaps the MOG2
                                      # foreground mask, 0 = no boost (today's
                                      # formula, unchanged); see Tracker in
                                      # cv_detect.py

    # Foveal Temporal Accumulation (CV Mode Phase 2). Detection now runs on
    # its own thread (see CVProcessor._detection_loop), decoupled from the
    # writer loop's display cadence -- a detector forward pass (~200-300ms
    # on tanzania, worse on tina's i3-2330M) no longer stalls schedule_frame.
    # That decoupling is what pays for accumulation: several frames of the
    # gated crop are averaged before the (now off-thread) detect() call,
    # raising SNR on a noisy sensor at zero cost to display fps.
    'CV_DETECT_HZ':            1.5,   # target detection-cycle rate, Hz.
                                      # Independent of display fps entirely.
    'CV_FOVEAL_ACCUM_FRAMES':  4,     # crop samples averaged before a
                                      # detect() pass. 1 = today's Phase 1
                                      # single-frame behaviour.
    'CV_FOVEAL_ACCUM_DIFF_MAX': 6.0,  # mean abs difference (0-255 scale) on
                                      # a downsampled crop, between this
                                      # sample and the last accepted one,
                                      # below which the sample is "aligned
                                      # enough" to accumulate. Frame
                                      # differencing on the crop itself, not
                                      # the pipeline's full-frame optical
                                      # flow -- the flow field only exists on
                                      # the non-sharpie render path, and this
                                      # gate has to work regardless of edge
                                      # style (tina runs sharpie today).

    # Reference-image scene model (CV Mode Phase 3). A toggle on every
    # node, off by default -- CV_SCENE_ENABLED=0 means any node, including
    # tanzania, is exactly what it is today. Nothing here is tina-specific;
    # registration takes the target host as an argument (cv_scene_register.py)
    # and the runtime side only ever reads whatever CV_SCENE_REFERENCE points
    # to.
    # Acuity-adaptive detection scale. Measured on this fleet: feeding a
    # detector more pixels than the optics actually resolve costs accuracy,
    # because everything above the lens's real cutoff is noise and the
    # network spends capacity on it. Downscaling first (INTER_AREA, which
    # averages rather than samples) discards that noise band before
    # inference.
    #
    # Measured through the REAL detect() path -- which applies CLAHE to the
    # letterboxed canvas before inference. An earlier version of these
    # numbers was taken from raw network output WITHOUT that CLAHE stage
    # and materially overstated the benefit; the corrected figures are
    # below and the floor here was raised because of them.
    #
    #   tanzania (acuity ~392): person 0.72 @1.0, 0.77 @0.5, 0.41 @0.12
    #                           -> downscaling HURTS a healthy lens
    #   tina     (acuity ~37):  strongest-signal mean 0.143 @0.5,
    #                           0.107 @0.35, 0.014 @0.18, 0.011 @0.12
    #                           -- and at 0.12 the detector returns nothing
    #                           at all on 20% of frames.
    #
    # So the direction is real (a good lens wants full resolution, a poor
    # one wants less) but the aggressive end is harmful on BOTH sensors.
    # SCALE_MIN is therefore 0.30, not 0.12: measured coefficient of
    # variation is lowest (~0.51) in the 0.25-0.35 band with a 100% hit
    # rate, and a persistent estimator needs consistency far more than it
    # needs one good reading.
    #
    # Not fixed by any of this: YOLO cannot see the cat on tina at any
    # scale (best hit rate 10%, at a scale that wrecks everything else).
    # That is not a resolution problem and should not be chased with one --
    # see cv_persist.py / cv_occupancy.py.
    'CV_DETECT_ACUITY_ADAPT':   1,     # 0 = always detect at full frame size
    'CV_DETECT_ACUITY_DIVISOR': 360.0, # scale = acuity / this, clamped below
    'CV_DETECT_SCALE_MIN':      0.30,  # never downscale past the stable band
    'CV_DETECT_SCALE_MAX':      1.0,   # never upsample past native
    'CV_DETECT_ACUITY_PERIOD':  30,    # re-measure acuity every N detect cycles

    # Persistent scene entities (CV Mode Phase 4). Off by default. When on,
    # promoted detections are folded into a long-lived entity store whose
    # identities survive restarts -- see cv_persist.py. This is deliberately
    # fed from the TRACKER, not from raw detections: a track has already
    # survived the evidence accumulator, so the persistence layer integrates
    # over something that recurs rather than over every frame's noise.
    # Codec block-grid correction. A lossy feed leaves a highly structured
    # artifact: measured on tina, gradient energy peaks at column mod 8 == 7
    # at 7.2x the median phase, the SAME phase in 120/120 frames, with an
    # on/off-grid excess of 5.66 at CoV 0.025. That is predictable enough to
    # model and remove rather than step over.
    #
    # Because the phase is known deterministically, the correction only
    # touches the lines the grid falls on and leaves every other pixel
    # bit-exact -- unlike a generic deblocker, which has to guess and ends up
    # smoothing away the little real detail a soft sensor still delivers.
    #
    # Off by default so it can be A/B'd against a node's known-good look.
    # Learned once from the first CV_GRID_LEARN_FRAMES frames and cached:
    # the artifact is stationary, so re-deriving it per frame would be pure
    # cost.
    'CV_GRID_CORRECT_ENABLED': 0,
    'CV_GRID_CORRECT_STRENGTH': 0.6,  # 1.0 over-corrects: it penalises a
                                       # pixel once per axis, and since real
                                       # edges are not grid-aligned roughly
                                       # 1/8 of any genuine edge is attenuated
                                       # with the grid. 0.6 measured as the
                                       # point where blockiness reaches a
                                       # healthy source's level.
    'CV_GRID_LEARN_FRAMES':     40,

    'CV_PERSIST_ENABLED':      0,
    'CV_PERSIST_PATH':         '/var/lib/hls-livecam/scene_entities.json',
    'CV_PERSIST_MIN_EVIDENCE': 0.35,  # only fold in tracks the accumulator
                                       # has actually promoted
    'CV_PERSIST_SEMANTIC_FLOOR': 0.25, # class hypotheses below this are
                                       # DROPPED before the entity store sees
                                       # them -- the entity may still exist,
                                       # it just stays unnamed.
                                       #
                                       # Deliberately INDEPENDENT of
                                       # CV_DETECT_CONF. A detection floor is
                                       # tuned for what is worth drawing on a
                                       # HUD for one frame; this floor governs
                                       # what is worth REMEMBERING for weeks.
                                       # Persistence integrates evidence, so a
                                       # detector that is wrong in a
                                       # consistent direction -- which is what
                                       # an out-of-distribution input produces
                                       # -- would otherwise be laundered into
                                       # a confident, long-lived, named
                                       # entity. A run of HUMAN 0.03 on a
                                       # scene containing only cats is exactly
                                       # that failure starting.

    'CV_SCENE_ENABLED':        0,        # off by default; opt in per node
    'CV_SCENE_REFERENCE':      '/var/lib/hls-livecam/scene_model.json',
    'CV_SCENE_EVIDENCE_BOOST': 0.0,      # second, independent evidence
                                          # multiplier alongside
                                          # CV_FOVEAL_EVIDENCE_BOOST -- see
                                          # Tracker in cv_detect.py. 0 = no
                                          # boost (today's formula, unchanged).
    'CV_SCENE_STALE_CHECK_INTERVAL_S': 60.0,  # how often the detection
                                          # thread correlates the MOG2
                                          # background against the warped
                                          # reference. Not per-frame --
                                          # background does not change fast
                                          # enough to justify the cost.
    'CV_SCENE_STALE_THRESHOLD': 40.0,    # mean abs difference (0-255,
                                          # downscaled grayscale) above which
                                          # the reference is declared stale
                                          # -- the camera was pivoted and the
                                          # registered geometry no longer
                                          # matches what MOG2 has learned.
    'CV_SCENE_REREGISTER_MIN_INLIER_RATIO': 0.5,  # /api/scene-reregister
                                          # only replaces the stored
                                          # homography if the new match
                                          # clears this bar -- otherwise the
                                          # existing registration is kept
                                          # and the request reports failure.
                                          # A confidently wrong registration
                                          # is worse than none.
}


def _read_float(denv, key):
    try:
        return float(denv.get(key, _DEFAULTS[key]))
    except (TypeError, ValueError):
        return _DEFAULTS[key]


def _read_int(denv, key):
    try:
        return int(float(denv.get(key, _DEFAULTS[key])))
    except (TypeError, ValueError):
        return _DEFAULTS[key]


def _tone_curve_lut(threshold, ceiling, gamma, shadow_lift=0.0, shadow_range=110):
    """Full tone curve: shadow lift at the dark end, highlight rolloff at the
    bright end, identity through the middle.

    The rolloff alone only ever helps a blown-out surface. A dark subject in
    the same frame -- black fur on white bedding -- is at the other end of the
    histogram and is crushed to a silhouette regardless of what the highlights
    do. The lift raises those values with a gamma curve, weighted so it fades
    out by `shadow_range` and leaves midtones alone.
    """
    lut = _highlight_rolloff_lut(threshold, ceiling, gamma).astype(np.float32)
    if shadow_lift > 0.0:
        x = np.arange(256, dtype=np.float32)
        # Gamma < 1 raises dark values; strength tapers to zero at shadow_range
        # so the curve stays continuous instead of stepping at the boundary.
        lifted = 255.0 * np.power(x / 255.0, 1.0 / (1.0 + shadow_lift))
        w = np.clip((shadow_range - x) / max(shadow_range, 1), 0.0, 1.0)
        lut = lut * (1.0 - w) + lifted * w
    return np.clip(lut, 0, 255).astype(np.uint8)


def _highlight_rolloff_lut(threshold, ceiling, gamma):
    """Soft-knee curve for the L channel: identity at and below `threshold`.
    Above it, remaps input range [threshold, 255] onto output range
    [threshold, ceiling] via t**gamma. With ceiling < 255 this genuinely
    compresses highlights -- a pixel clipped at 255 comes out at `ceiling`
    instead, leaving real numeric headroom below white for CLAHE's local
    contrast stretch to use, rather than a plateau with nothing left to
    redistribute. ceiling=255 and/or gamma=1.0 is the identity (disabled).
    Precomputed once per parameter set, not per frame: applying it is a
    single cv2.LUT call, not per-pixel math.
    """
    x = np.arange(256, dtype=np.float32)
    in_span = max(1e-6, 255.0 - threshold)
    out_span = max(0.0, ceiling - threshold)
    t = np.clip((x - threshold) / in_span, 0.0, 1.0)
    y = threshold + out_span * np.power(t, gamma)
    y = np.where(x <= threshold, x, y)
    return np.clip(y, 0, 255).astype(np.uint8)


class CVProcessor:
    """Stateful temporal enhancer. process(frame) -> (frame, metadata).

    frame is HxWx3 uint8 RGB. Retains up to HISTORY frames of history
    internally; callers do not manage state between calls.

    denv: the device.env dict (see broadcast-api's _read_device_env()).
    Optional -- omitting it, or omitting individual CV_* keys, reproduces
    the original hardcoded defaults exactly.
    """

    HISTORY = 3

    def __init__(self, denv=None):
        denv = denv or {}
        self._last_luma = None  # denoised L channel, kept by _tone_correct
        self._frames = deque(maxlen=self.HISTORY)  # raw RGB frames, newest last
        self._gray_small = deque(maxlen=self.HISTORY)  # matching downscaled gray, for flow

        self._motion_threshold = _read_float(denv, 'CV_MOTION_THRESHOLD')
        self._unsharp_amount = _read_float(denv, 'CV_UNSHARP_AMOUNT')
        self._unsharp_sigma = _read_float(denv, 'CV_UNSHARP_SIGMA')

        clip = _read_float(denv, 'CV_CLAHE_CLIP')
        tiles = _read_int(denv, 'CV_CLAHE_TILES')
        self._clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))

        self._edge_enabled = _read_int(denv, 'CV_EDGE_ENABLED') != 0
        self._edge_sharp_min = _read_float(denv, 'CV_EDGE_SHARPNESS_MIN')
        self._edge_sharp_off = _read_float(denv, 'CV_EDGE_SHARPNESS_OFF')
        self._edge_strength = _read_float(denv, 'CV_EDGE_STRENGTH')
        self._edge_sigma = _read_float(denv, 'CV_EDGE_SIGMA')
        self._edge_sigma_max = _read_float(denv, 'CV_EDGE_SIGMA_MAX')
        self._edge_percentile = _read_float(denv, 'CV_EDGE_PERCENTILE')
        self._edge_gamma = _read_float(denv, 'CV_EDGE_GAMMA')
        self._edge_style = str((denv or {}).get('CV_EDGE_STYLE',
                                                _DEFAULTS['CV_EDGE_STYLE'])).strip().lower()
        self._xdog_sigma = max(0.1, _read_float(denv, 'CV_XDOG_SIGMA'))
        self._xdog_k = max(1.01, _read_float(denv, 'CV_XDOG_K'))
        self._xdog_tau = _read_float(denv, 'CV_XDOG_TAU')
        self._xdog_eps = _read_float(denv, 'CV_XDOG_EPS')
        self._xdog_percentile = _read_float(denv, 'CV_XDOG_PERCENTILE')
        self._xdog_phi = _read_float(denv, 'CV_XDOG_PHI')
        self._xdog_tone = min(1.0, max(0.0, _read_float(denv, 'CV_XDOG_TONE')))
        self._region_scale = min(1.0, max(0.1, _read_float(denv, 'CV_REGION_SCALE')))
        self._region_sigma_color = max(1.0, _read_float(denv, 'CV_REGION_SIGMA_COLOR'))
        self._region_sigma_space = max(1.0, _read_float(denv, 'CV_REGION_SIGMA_SPACE'))
        self._region_bilateral_d = max(1, _read_int(denv, 'CV_REGION_BILATERAL_D'))
        self._region_size = max(4, _read_int(denv, 'CV_REGION_SIZE'))
        self._region_iters = max(1, _read_int(denv, 'CV_REGION_ITERS'))
        self._region_merge_dist = max(0.0, _read_float(denv, 'CV_REGION_MERGE_DIST'))
        self._region_bands = max(2, _read_int(denv, 'CV_REGION_BANDS'))
        self._region_edge_alpha = min(1.0, max(0.0, _read_float(denv, 'CV_REGION_EDGE_ALPHA')))
        self._region_fg_boundary = _read_int(denv, 'CV_REGION_FG_BOUNDARY') != 0
        self._region_warned = False
        self._illum_enabled = _read_int(denv, 'CV_ILLUM_ENABLED') != 0
        self._illum_strength = max(0.0, _read_float(denv, 'CV_ILLUM_STRENGTH'))
        self._illum_sigma_frac = max(0.01, _read_float(denv, 'CV_ILLUM_SIGMA_FRAC'))
        self._illum_scale = min(1.0, max(0.05, _read_float(denv, 'CV_ILLUM_SCALE')))
        self._sharpie_levels = max(2, _read_int(denv, 'CV_SHARPIE_LEVELS'))
        self._sharpie_thick = max(1, _read_int(denv, 'CV_SHARPIE_THICK'))
        self._sharpie_min_len = _read_int(denv, 'CV_SHARPIE_MIN_LEN')
        self._sharpie_smooth = _read_int(denv, 'CV_SHARPIE_SMOOTH')
        self._sharpie_close = _read_int(denv, 'CV_SHARPIE_CLOSE')
        self._sharpie_scale = _read_float(denv, 'CV_SHARPIE_SCALE')
        self._sharpie_blend = max(0.0, min(1.0, _read_float(denv, 'CV_SHARPIE_BLEND')))

        self._detect_enabled = _read_int(denv, 'CV_DETECT_ENABLED') != 0
        self._detect_interval = max(1, _read_int(denv, 'CV_DETECT_INTERVAL'))
        self._hud_enabled = _read_int(denv, 'CV_HUD_ENABLED') != 0
        self._frame_n = 0
        self._detector = None
        self._tracker = None
        self._notifier = None

        # Foveal Layer (CV Mode Phase 1). self._foveal_enabled is the LIVE
        # flag -- broadcast-api's writer loop calls set_foveal() every
        # iteration with the current checkbox state, the same way it
        # re-reads feed mode every iteration. CV_FOVEAL_ENABLED from
        # device.env is only the value this starts at.
        self._foveal_enabled = _read_int(denv, 'CV_FOVEAL_ENABLED') != 0
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=_read_int(denv, 'CV_MOG2_HISTORY'),
            varThreshold=_read_float(denv, 'CV_MOG2_VAR_THRESHOLD'),
            detectShadows=_read_int(denv, 'CV_MOG2_DETECT_SHADOWS') != 0)
        self._mog2_detect_shadows = _read_int(denv, 'CV_MOG2_DETECT_SHADOWS') != 0
        self._mog2_scale = _read_float(denv, 'CV_MOG2_SCALE')
        self._foveal_dilate = max(1, _read_int(denv, 'CV_FOVEAL_DILATE'))
        self._foveal_min_area = _read_float(denv, 'CV_FOVEAL_MIN_AREA')
        self._foveal_pad = max(0.0, _read_float(denv, 'CV_FOVEAL_PAD'))
        self._foveal_evidence_boost = _read_float(denv, 'CV_FOVEAL_EVIDENCE_BOOST')
        self._fg_mask = None  # HxW uint8, most recent MOG2 foreground mask

        # Foveal Temporal Accumulation (CV Mode Phase 2). process() (the
        # writer-loop thread) publishes into this single slot every call --
        # a reference swap, no detector work -- and a dedicated daemon
        # thread (_detection_loop) consumes it independently. Same
        # latest-wins idiom as broadcast-api's own _cam_frame/_cam_frame_lock,
        # not a queue: a live feed has no use for a detection backlog.
        self._detect_hz = max(0.1, _read_float(denv, 'CV_DETECT_HZ'))
        self._accum_frames = max(1, _read_int(denv, 'CV_FOVEAL_ACCUM_FRAMES'))
        self._accum_diff_max = _read_float(denv, 'CV_FOVEAL_ACCUM_DIFF_MAX')
        self._detect_slot = None  # (frame, crop_box, fg_mask) or None
        self._detect_slot_lock = threading.Lock()
        # Tracker is written by the detection thread and read by the render
        # path (_detect_and_hud, called every display frame) -- both sides
        # take this lock. Everything else the detection thread touches
        # (_accum_*) is private to that one thread and needs no lock.
        self._tracker_lock = threading.Lock()
        self._accum_box = None          # fixed crop rect for the in-progress
                                         # accumulation window -- every
                                         # sample crops to THIS rect, not its
                                         # own freshly-recomputed box, which
                                         # is what keeps samples registered
                                         # without real affine alignment.
        self._accum_buf = deque(maxlen=self._accum_frames)
        self._accum_ref_gray = None     # small downsampled gray of the last
                                         # accepted sample, for the diff gate
        self._accum_started = None      # perf_counter() when this window
                                         # opened, for the timeout fallback
        self._accum_timeout_s = 2.0 / self._detect_hz  # up to ~2 cycles
                                         # before forcing a partial-N pass,
                                         # so a subject that never fully
                                         # settles still eventually gets a
                                         # detection rather than starving
        self._last_detect_end = 0.0     # perf_counter() of the last
                                         # completed detect() pass -- throttles
                                         # new cycles (gated or fallback) to
                                         # CV_DETECT_HZ

        # Reference-image scene model (CV Mode Phase 3). CV_SCENE_ENABLED=0
        # (the default, on every node including tanzania) means everything
        # below stays at its neutral no-op state -- self._scene_regions
        # stays empty, self._scene_homography stays None, and every
        # scene_consistency computation short-circuits to the Detection
        # default (1.0, neutral) exactly as if this file did not exist.
        self._acuity_adapt = _read_int(denv, 'CV_DETECT_ACUITY_ADAPT') != 0
        self._acuity_divisor = max(1.0, _read_float(denv, 'CV_DETECT_ACUITY_DIVISOR'))
        self._detect_scale_min = _read_float(denv, 'CV_DETECT_SCALE_MIN')
        self._detect_scale_max = _read_float(denv, 'CV_DETECT_SCALE_MAX')
        self._acuity_period = max(1, _read_int(denv, 'CV_DETECT_ACUITY_PERIOD'))
        self._detect_scale = 1.0    # current adapted scale
        self._acuity_last = None    # last measured variance-of-Laplacian
        self._acuity_n = 0          # detect cycles since last measurement

        self._grid_correct = (_read_int(denv, 'CV_GRID_CORRECT_ENABLED') != 0
                              and _cvo is not None)
        self._grid_strength = _read_float(denv, 'CV_GRID_CORRECT_STRENGTH')
        self._grid_learn_frames = max(4, _read_int(denv, 'CV_GRID_LEARN_FRAMES'))
        self._grid_prior = None        # learned once, then cached
        self._grid_learn_buf = []      # raw frames held only until learned

        self._persist_enabled = (_read_int(denv, 'CV_PERSIST_ENABLED') != 0
                                 and _cvp is not None)
        self._persist_min_evidence = _read_float(denv, 'CV_PERSIST_MIN_EVIDENCE')
        self._persist_semantic_floor = _read_float(
            denv, 'CV_PERSIST_SEMANTIC_FLOOR')
        self._persist_store = None
        if self._persist_enabled:
            try:
                self._persist_store = _cvp.EntityStore(
                    {'CV_PERSIST_PATH': str((denv or {}).get(
                        'CV_PERSIST_PATH', _DEFAULTS['CV_PERSIST_PATH']))})
                self._persist_store.load()
            except Exception as exc:
                print(f"CV PERSIST: init failed ({type(exc).__name__}: {exc}) "
                      "-- persistence disabled", flush=True)
                self._persist_enabled = False
                self._persist_store = None

        self._scene_enabled = _read_int(denv, 'CV_SCENE_ENABLED') != 0
        self._scene_evidence_boost = _read_float(denv, 'CV_SCENE_EVIDENCE_BOOST')
        self._scene_stale_interval = _read_float(denv, 'CV_SCENE_STALE_CHECK_INTERVAL_S')
        self._scene_stale_threshold = _read_float(denv, 'CV_SCENE_STALE_THRESHOLD')
        self._scene_reregister_min_inlier_ratio = _read_float(
            denv, 'CV_SCENE_REREGISTER_MIN_INLIER_RATIO')
        self._scene_reference_path = str(
            (denv or {}).get('CV_SCENE_REFERENCE', _DEFAULTS['CV_SCENE_REFERENCE']))
        self._scene_regions = []        # list of {label, box, source_confidence, static}
        self._scene_homography = None   # 3x3 np.ndarray or None
        self._scene_reference_gray = None  # HxW uint8, warped-space source for staleness
        self._scene_stale = False       # surfaced via GET, three-state UI reads this
        self._scene_registered = False  # a model loaded successfully at all
        self._scene_last_stale_check = 0.0  # perf_counter() of the last staleness check
        if self._scene_enabled:
            self._load_scene_model()

        # Notification is independent of detection succeeding: build it
        # first so a detector init failure still leaves a valid (disabled)
        # notifier rather than an attribute that does not exist.
        if _cvn is not None:
            try:
                self._notifier = _cvn.make_notifier(
                    denv, log=lambda m: print(m, flush=True))
            except Exception as exc:
                print(f"NOTIFY INIT ERROR: {type(exc).__name__}: {exc}",
                      flush=True)
        if self._detect_enabled and _cvd is not None:
            backend = str((denv or {}).get('CV_DETECT_BACKEND',
                                           _DEFAULTS['CV_DETECT_BACKEND']))
            try:
                _raw_model = (denv or {}).get('CV_DETECT_MODEL')
                _effective_model = str(
                    _raw_model or _DEFAULTS['CV_DETECT_MODEL']
                )

                print(
                    "SHAKEY DETECTOR CONFIG:"
                    f" backend={backend!r}"
                    f" raw_model={_raw_model!r}"
                    f" effective_model={_effective_model!r}"
                    f" exists={__import__('os').path.exists(_effective_model)!r}",
                    flush=True,
                )

                _class_filter = _cvd.parse_class_filter(
                    (denv or {}).get('CV_DETECT_CLASSES', ''))
                if _class_filter:
                    print(f"CV DETECT: class filter active -> "
                          f"{sorted(_class_filter.values())}", flush=True)
                self._detector = _cvd.make_detector(
                    backend,
                    model_path=_effective_model,
                    conf=_read_float(denv, 'CV_DETECT_CONF'),
                    min_area_frac=_read_float(denv, 'CV_DETECT_MIN_AREA'),
                    classes=_class_filter)
                self._tracker = _cvd.Tracker(
                    evidence_decay=_read_float(denv, 'CV_TRACK_EVIDENCE_DECAY'),
                    promote_threshold=_read_float(denv, 'CV_TRACK_PROMOTE_THRESHOLD'),
                    evidence_boost=self._foveal_evidence_boost,
                    scene_boost=self._scene_evidence_boost)
            except Exception as exc:
                print(
                    f"SHAKEY DETECTOR INIT ERROR: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                self._detector = None
        self._sharpen_motion = _read_int(denv, 'CV_SHARPEN_MOTION') != 0
        self._denoise_motion = _read_int(denv, 'CV_DENOISE_MOTION') != 0

        self._artifact_enabled = _read_int(denv, 'CV_ARTIFACT_ENABLED') != 0
        self._artifact_samples = _read_int(denv, 'CV_ARTIFACT_SAMPLES')
        self._artifact_min_motion = _read_float(denv, 'CV_ARTIFACT_MIN_MOTION')
        self._artifact_threshold = _read_int(denv, 'CV_ARTIFACT_THRESHOLD')
        self._artifact_dilate = _read_int(denv, 'CV_ARTIFACT_DILATE')
        self._artifact_max_area = _read_float(denv, 'CV_ARTIFACT_MAX_AREA')
        self._artifact_mask = None      # HxW uint8, 255 where lens is dirty
        self._artifact_samples_buf = []  # luminance frames collected while learning
        self._artifact_learning = self._artifact_enabled

        hi_threshold = _read_int(denv, 'CV_HIGHLIGHT_THRESHOLD')
        hi_ceiling = _read_int(denv, 'CV_HIGHLIGHT_CEILING')
        hi_gamma = _read_float(denv, 'CV_HIGHLIGHT_GAMMA')
        self._highlight_lut = _tone_curve_lut(
            hi_threshold, hi_ceiling, hi_gamma,
            _read_float(denv, 'CV_SHADOW_LIFT'),
            _read_int(denv, 'CV_SHADOW_RANGE'))

        # Detection thread starts last, once everything it can touch
        # (_tracker, _fg_mask, _accum_*) is fully constructed. Daemon, no
        # explicit stop: this instance lives for the life of the writer-loop
        # process, and dies with it -- same convention as broadcast-api's
        # own _drain_loop/_writer_loop threads.
        if self._detect_enabled and self._detector is not None:
            threading.Thread(target=self._detection_loop, daemon=True).start()

    def set_foveal(self, enabled):
        """Live toggle for the Foveal Layer, called every writer-loop
        iteration with the checkbox's current state -- same pattern as feed
        mode itself. A plain attribute flip, not a reinit: MOG2's model
        keeps accumulating regardless of whether the gate is currently
        consulted, so toggling off and back on does not throw away the
        learned background."""
        self._foveal_enabled = bool(enabled)

    def process(self, frame):
        t = {}
        t0 = time.perf_counter()

        if self._grid_correct:
            t_grid = time.perf_counter()
            frame = self._apply_grid_correction(frame)
            t['grid'] = time.perf_counter() - t_grid

        # Ahead of the detection handoff below, deliberately: this is an
        # observation-quality stage, not a display stage, so the detector
        # and MOG2 must see the corrected frame too. Putting it after the
        # handoff would improve only what a human sees.
        if self._illum_enabled:
            t_illum = time.perf_counter()
            frame = self._apply_illumination(frame)
            t['illum'] = time.perf_counter() - t_illum

        if self._foveal_enabled:
            t_mog = time.perf_counter()
            sc = self._mog2_scale
            mog_input = frame if not (0.0 < sc < 1.0) else cv2.resize(
                frame, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            fg = self._mog2.apply(mog_input)
            if self._mog2_detect_shadows:
                # Shadow pixels come back gray (127); only genuine
                # foreground (255) should count for gating.
                _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
            if fg.shape[:2] != frame.shape[:2]:
                # Nearest, not linear: this is a binary mask, and a coarse
                # crop box is all it needs to feed -- same reasoning as the
                # existing optical-flow motion_mask upscale.
                fg = cv2.resize(fg, (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            self._fg_mask = fg
            t['mog2'] = time.perf_counter() - t_mog
        else:
            self._fg_mask = None

        # Publish to the detection thread. Cheap: crop_box is the same
        # dilate+contour pass already paid for above; frame/fg_mask are
        # reference swaps, never copies -- both are freshly allocated every
        # call (mog2.apply()/resize() never mutate a prior array in place),
        # so sharing the reference across threads is safe without a copy.
        # Runs regardless of edge style: the detection thread must not care
        # whether the display path is about to take the sharpie early return
        # below.
        crop_box = (self._foveal_crop_box(frame.shape)
                    if self._foveal_enabled and self._fg_mask is not None
                    else None)
        with self._detect_slot_lock:
            self._detect_slot = (frame, crop_box, self._fg_mask)

        # Sharpie discards the photo, so everything that exists to improve the
        # photo is wasted work: optical flow, the temporal blend and the
        # unsharp pass are ~70ms of the frame and none of it survives into a
        # line drawing. Tone still runs -- the drawing needs its equalised
        # luminance, and without equalisation the bands all land inside the
        # blown-out end of the histogram.
        if self._edge_enabled and self._edge_style == 'sharpie':
            toned = self._tone_correct(frame)
            t['tone'] = time.perf_counter() - t0
            t5 = time.perf_counter()
            out = self._draw_sharpie(toned)
            t['edges'] = time.perf_counter() - t5
            t6 = time.perf_counter()
            out, tracks = self._detect_and_hud(frame, out)
            t['detect'] = time.perf_counter() - t6
            t['total'] = time.perf_counter() - t0
            return out, {
                'motion': 0.0,
                'sharpness': 0.0,
                'tracks': len(tracks),
                'edge_weight': 1.0,
                'artifact_mask': self._artifact_mask is not None,
                'artifact_learning': False,
                'timings_ms': {k: round(v * 1000, 3) for k, v in t.items()},
                'history': len(self._frames),
            }

        small_gray = cv2.cvtColor(
            cv2.resize(frame, None, fx=_FLOW_SCALE, fy=_FLOW_SCALE, interpolation=cv2.INTER_AREA),
            cv2.COLOR_RGB2GRAY,
        )
        t['downscale_gray'] = time.perf_counter() - t0

        motion_mean = 0.0
        moving_mask = None  # HxW uint8, 255 where moving, else 0
        t1 = time.perf_counter()
        if self._gray_small:
            flow = cv2.calcOpticalFlowFarneback(
                self._gray_small[-1], small_gray, None,
                pyr_scale=0.5, levels=2, winsize=13,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
            )
            mag_small = cv2.magnitude(flow[..., 0], flow[..., 1])
            motion_mean = float(mag_small.mean())
            _, moving_small = cv2.threshold(mag_small, self._motion_threshold, 255, cv2.THRESH_BINARY)
            moving_mask = cv2.resize(moving_small.astype(np.uint8),
                                      (frame.shape[1], frame.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
        t['flow'] = time.perf_counter() - t1

        t2 = time.perf_counter()
        denoised = self._temporal_denoise(frame, moving_mask)
        t['denoise'] = time.perf_counter() - t2

        t3 = time.perf_counter()
        toned = self._tone_correct(denoised)
        t['tone'] = time.perf_counter() - t3

        t4 = time.perf_counter()
        sharpened = self._sharpen(toned, moving_mask)
        t['sharpen'] = time.perf_counter() - t4

        # Sharpness is measured on the *denoised* luminance, not the raw
        # frame. Sensor noise is high-frequency, so a variance-of-Laplacian
        # taken before denoising reads grain as detail and a blurry, noisy
        # frame scores as sharp -- exactly backwards. Measuring after the
        # temporal blend removes that confound. (Tenengrad scores better in
        # comparative focus studies but is documented as having weak noise
        # immunity, which is the wrong trade for an old webcam.)
        # Sampling uses the denoised luminance the tone stage kept, and runs
        # only while learning -- once the mask exists this costs nothing.
        if self._artifact_learning:
            self._collect_artifact_sample(self._last_luma, motion_mean)
        if self._artifact_enabled and self._artifact_mask is not None:
            toned = self._subtract_artifacts(toned)
            sharpened = self._sharpen(toned, moving_mask)

        t5 = time.perf_counter()
        sharpness = self._measure_sharpness(self._last_luma)
        edged, edge_weight = self._draw_edges(sharpened, sharpness)
        t['edges'] = time.perf_counter() - t5

        t6 = time.perf_counter()
        edged, hud_tracks = self._detect_and_hud(frame, edged)
        t['detect'] = time.perf_counter() - t6

        self._frames.append(frame)
        self._gray_small.append(small_gray)

        # SHAKey HUD is already drawn on `edged` inside _detect_and_hud
        # (deliberately, immediately before it returns, so the HUD cannot be
        # consumed by the enhancement/edge pipeline). Found while
        # restructuring this method for Phase 2: this path used to call
        # _cvd.draw_hud() a SECOND time here, double-rendering every frame
        # in the non-sharpie path (the sharpie path never had this bug --
        # it only calls _detect_and_hud). Pre-existing, not introduced by
        # this phase; removed rather than left in place since it was found
        # in code this phase already had to touch.

        t['total'] = time.perf_counter() - t0
        metadata = {
            'motion': motion_mean,
            'sharpness': round(sharpness, 2),
            'edge_weight': round(edge_weight, 3),
            'artifact_mask': self._artifact_mask is not None,
            'artifact_learning': self._artifact_learning,
            'tracks': len(hud_tracks),
            'timings_ms': {k: round(v * 1000, 3) for k, v in t.items()},
            'history': len(self._frames),
        }
        return edged, metadata

    def _measure_sharpness(self, luma):
        """Variance of the Laplacian on a quarter-scale luminance image.

        Takes the L channel the tone stage has already extracted, rather
        than converting and downscaling a colour frame again -- measured at
        10.2ms per frame when it did its own full-resolution colour resize,
        paid on *every* frame including sharp ones where no edges are drawn.
        Downscaling a single channel that already exists is a fraction of
        that.

        Quarter scale also suppresses, via INTER_AREA averaging, the sensor
        grain that would otherwise inflate the score. Absolute values are
        therefore lower than a full-resolution figure, which is why the
        thresholds are configuration rather than constants.
        """
        small = cv2.resize(luma, None, fx=_FLOW_SCALE, fy=_FLOW_SCALE,
                           interpolation=cv2.INTER_AREA)
        return float(cv2.Laplacian(small, cv2.CV_64F).var())

    def _edge_weight(self, sharpness):
        """0 when the image is sharp enough to leave alone, 1 when it is
        thoroughly soft, ramped between. A ramp rather than a hard switch:
        a binary cutoff makes edges snap in and out as the score dithers
        across the threshold, which is far more distracting than the edges
        themselves."""
        lo, hi = self._edge_sharp_min, self._edge_sharp_off
        if hi <= lo:
            return 1.0 if sharpness <= lo else 0.0
        if sharpness >= hi:
            return 0.0
        if sharpness <= lo:
            return 1.0
        return float((hi - sharpness) / (hi - lo))

    def _draw_edges(self, frame, sharpness):
        """XDoG-style line overlay, engaged only as the image goes soft.

        Difference-of-Gaussians rather than Canny: it yields tonal strokes
        that vary with edge strength instead of uniform binary hairlines,
        which is what reads as a drawing rather than a mask. Flow-based
        (coherent) line drawing is the better-looking method but costs
        several times more, and this has to run on the weakest node in the
        fleet.

        Stroke width scales with how soft the image is, per the brief: a
        heavily blurred frame has no fine structure left to trace, so a
        hairline would follow noise. Returns (frame, weight_applied).
        """
        if not self._edge_enabled:
            return frame, 0.0

        # 'regions' is a rendering CHOICE, not a softness remedy, so it is
        # dispatched ahead of the sharpness ramp -- an operator who selects
        # it wants it on every frame, not only on the frames the ramp
        # judges soft enough. (Sharpie gets the same treatment via its own
        # early return in process().) Everything below this line is the
        # engage-as-the-image-degrades path.
        if self._edge_style == 'regions':
            return self._draw_regions(frame, self._fg_mask), 1.0

        w = self._edge_weight(sharpness)
        if w <= 0.0:
            # Sharp feed: absent entirely, and costing nothing beyond the
            # metric itself. This is the tanzania case.
            return frame, 0.0

        if self._edge_style == 'sharpie':
            return self._draw_sharpie(frame), w
        if self._edge_style == 'xdog':
            return self._draw_xdog(frame), w

        sigma = min(self._edge_sigma * (1.0 + 2.0 * w), self._edge_sigma_max)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        g1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
        g2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma * 1.6)
        dog = cv2.subtract(g1, g2)

        # Normalise against a high percentile, not the maximum. On a noisy
        # frame the max is a single outlier pixel, so dividing by it scales
        # every genuine stroke down to nothing -- which is exactly why the
        # first version drew almost nothing on a soft, grainy feed. A
        # percentile puts real edges at full range and lets the outliers
        # clip, which is what they should do.
        ref = float(np.percentile(dog, self._edge_percentile))
        if ref <= 1e-6:
            return frame, 0.0
        strokes = cv2.convertScaleAbs(dog, alpha=255.0 / ref)

        # Gamma below 1 lifts mid-strength edges toward full ink instead of
        # leaving the drawing dominated by only the few strongest contours.
        if self._edge_gamma != 1.0:
            lut = np.clip(255.0 * (np.arange(256) / 255.0) ** self._edge_gamma,
                          0, 255).astype(np.uint8)
            strokes = cv2.LUT(strokes, lut)

        # Darken along strokes, scaled by engagement.
        amount = self._edge_strength * w
        ink = cv2.cvtColor(strokes, cv2.COLOR_GRAY2RGB)
        return cv2.addWeighted(frame, 1.0, ink, -amount, 0), w

    def _foveal_crop_box(self, frame_shape):
        """Largest MOG2 foreground cluster, dilated/padded, as a crop box
        the detector should look at instead of the full frame. Returns
        (x, y, w, h) in full-frame pixels, or None if nothing survives
        CV_FOVEAL_MIN_AREA -- callers fall back to a full-frame pass.
        """
        h, w = frame_shape[:2]
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._foveal_dilate | 1, self._foveal_dilate | 1))
        mask = cv2.dilate(self._fg_mask, k)
        cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cont:
            return None
        largest = max(cont, key=cv2.contourArea)
        if cv2.contourArea(largest) < self._foveal_min_area * (h * w):
            return None

        bx, by, bw, bh = cv2.boundingRect(largest)
        pad_x = int(bw * self._foveal_pad)
        pad_y = int(bh * self._foveal_pad)
        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(w, bx + bw + pad_x)
        y2 = min(h, by + bh + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2 - x1, y2 - y1)

    def _set_fg_consistency(self, dets, fg_mask):
        """Fraction of each detection's box covered by the MOG2 foreground
        mask, 0.0-1.0 -- the evidence accumulator's "is this position
        plausible" signal. Left at the Detection default (1.0, neutral) if
        the box is degenerate or falls entirely outside the frame.

        Takes fg_mask explicitly rather than reading self._fg_mask: this
        runs on the detection thread, which may still be finishing a cycle
        (accumulation can span several display frames) after the display
        thread has already moved self._fg_mask on to something newer --
        the mask passed here is the one that was current when this
        detection cycle's frame was published, which is the honest match.
        """
        fh, fw = fg_mask.shape[:2]
        for d in dets:
            x1 = max(0, min(fw, d.x))
            y1 = max(0, min(fh, d.y))
            x2 = max(0, min(fw, d.x + d.w))
            y2 = max(0, min(fh, d.y + d.h))
            if x2 <= x1 or y2 <= y1:
                continue
            region = fg_mask[y1:y2, x1:x2]
            d.fg_consistency = float((region > 0).mean())

    def _draw_xdog(self, frame):
        """Extended difference-of-Gaussians (Winnemoller).

        Plain DoG answers "is there an edge here" and discards the rest.
        XDoG keeps a continuous response and pushes it through a soft ramp,
        so a surface with fine low-contrast structure -- fur, fabric weave --
        comes out as graded tone rather than either flat grey or a hard
        outline. On a soft sensor that band is exactly where the remaining
        detail lives, and a hard threshold throws it away.

        The response is NORMALISED against a high percentile before the
        ramp. That is not cosmetic: measured on tina, the raw DoG response
        spans median 0.0088 to p99 0.0205, so absolute thresholds tuned on
        any other image ink nothing at all here -- an earlier version of
        this function used eps=-0.05 and left 0.0% of the frame inked,
        making the whole stage a silent no-op. Normalising first makes eps
        and phi mean the same thing on a good sensor and a dying one, which
        is the same reason the overlay path normalises by percentile.

        Nothing here invents detail: the ramp is monotonic in the DoG
        response, so it can only redistribute contrast the optics actually
        delivered -- unlike sharpening, which manufactures halos.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        g1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=self._xdog_sigma)
        g2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=self._xdog_sigma * self._xdog_k)
        d = g1 - self._xdog_tau * g2

        ref = float(np.percentile(np.abs(d), self._xdog_percentile))
        if ref <= 1e-6:
            return frame
        dn = d / ref                      # ~[-1, 1] for the bulk of the frame

        # Soft ramp in normalised units: 1 (no ink) above eps, tanh-graded
        # below it. phi sets how abruptly ink arrives.
        ink = np.where(dn >= self._xdog_eps,
                       1.0,
                       1.0 + np.tanh(self._xdog_phi * (dn - self._xdog_eps)))
        ink = np.clip(ink, 0.0, 1.0).astype(np.float32)

        if self._xdog_tone <= 0.0:
            return cv2.cvtColor((ink * 255.0).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # Multiply the ink over the photo so colour and shading survive;
        # tone mixes back toward the untouched frame.
        ink3 = cv2.cvtColor((ink * 255.0).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        inked = cv2.multiply(frame, ink3, scale=1.0 / 255.0)
        return cv2.addWeighted(inked, 1.0 - self._xdog_tone,
                               frame, self._xdog_tone, 0.0)

    def _apply_illumination(self, frame):
        """Correct the spatial illumination field, not the global tone.

        The failure this addresses is photometric and POSITIONAL: on
        tanzania the centre-right is blown while the left is dark, but the
        frame mean is a perfectly ordinary 151. Every global operator --
        gamma, a single CLAHE pass, the existing highlight/shadow chain --
        sees a correctly exposed image and has no representation for "too
        bright HERE, too dark THERE". Measured tile spread on that frame is
        142 luma levels; see the CV_ILLUM_* block for the comparison table.

        L(x,y) is estimated as a heavily low-passed luminance -- the smooth
        component a global curve cannot express -- and each pixel is scaled
        toward a flat target by that field.

        This does NOT invent detail, and the distinction matters: 6.6% of
        that frame is clipped at white, and clipped is gone. Those pixels
        are rescaled along with everything else, never reconstructed. A
        model that "recovers" texture inside a blown region is fabricating
        it, which is the standing no-super-resolution rule -- the reason
        this stage is a division by a smooth field and nothing more.
        """
        if self._illum_strength <= 0.0:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0].astype(np.float32)
        sc = self._illum_scale
        small = cv2.resize(L, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        sig = max(1.0, self._illum_sigma_frac * max(small.shape))
        field = cv2.GaussianBlur(small, (0, 0), sigmaX=sig, sigmaY=sig)
        field = cv2.resize(field, (L.shape[1], L.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        target = float(field.mean())
        gain = (target / np.maximum(field, 1.0)) ** self._illum_strength
        lab[:, :, 0] = np.clip(L * gain, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def _region_labels(self, low):
        """Segment the low-frequency colour image into coherent masses.

        Returns an int32 label image, labels 0..n-1.

        NOT seeded from the edge map. The first version of this method was
        (edges -> close -> connected components of the non-edge area ->
        watershed) and it collapsed the entire frame into ONE region on
        tina: a soft sensor's contours do not enclose anything, so "the set
        of pixels that are not an edge" is a single connected mass and the
        flood has nothing to flood between. That is the same failure the
        occupancy work hit in Phase 4, where edge-based region proposal
        produced one whole-frame blob and luminance banding fixed it.

        So regions are grown from COLOUR COHERENCE instead, which is the
        property this sensor actually still has. SLIC superpixels give
        compact, colour-uniform fragments with no dependence on closed
        contours; adjacent fragments are then merged wherever their mean
        LAB colours are close, which is what turns a few hundred fragments
        into the handful of large masses the renderer wants. The edge map
        is still used -- but only to ink boundaries at the end, which is
        the one job it can do reliably here.
        """
        lab = cv2.cvtColor(low, cv2.COLOR_RGB2LAB)
        sh, sw = lab.shape[:2]

        slic = getattr(getattr(cv2, 'ximgproc', None),
                       'createSuperpixelSLIC', None)
        if slic is None:
            # Fallback with no contrib dependency: band the luminance and
            # take connected components per band. Cruder -- two touching
            # surfaces of equal brightness and different colour become one
            # region -- but it is a real segmentation, not a silent
            # degradation to "no regions at all".
            if not self._region_warned:
                print('CV REGIONS: cv2.ximgproc.createSuperpixelSLIC '
                      'unavailable; using luminance-band fallback',
                      flush=True)
                self._region_warned = True
            step = max(1, 256 // max(2, self._region_bands))
            bands = (lab[:, :, 0] // step).astype(np.int32)
            out = np.zeros((sh, sw), np.int32)
            nxt = 0
            for b in np.unique(bands):
                n, cc = cv2.connectedComponents(
                    (bands == b).astype(np.uint8))
                if n <= 1:
                    continue
                m = cc > 0
                out[m] = cc[m] + nxt
                nxt += n
            return out

        sp = slic(lab, algorithm=cv2.ximgproc.SLICO,
                  region_size=self._region_size)
        sp.iterate(self._region_iters)
        labels = sp.getLabels().astype(np.int32)
        n = int(labels.max()) + 1
        if n <= 1:
            return labels

        # Mean LAB per superpixel, vectorised.
        flat = labels.reshape(-1)
        cnt = np.bincount(flat, minlength=n).astype(np.float32)
        cnt[cnt == 0] = 1.0
        labf = lab.reshape(-1, 3).astype(np.float32)
        mean = np.empty((n, 3), np.float32)
        for c in range(3):
            mean[:, c] = np.bincount(flat, weights=labf[:, c],
                                     minlength=n) / cnt

        # Adjacency: every horizontally or vertically neighbouring pair of
        # differing labels. Vectorised -- no per-pixel Python.
        a = np.concatenate([labels[:, :-1].ravel(), labels[:-1, :].ravel()])
        b = np.concatenate([labels[:, 1:].ravel(), labels[1:, :].ravel()])
        m = a != b
        a, b = a[m], b[m]
        if a.size == 0:
            return labels
        pairs = np.unique(np.stack([np.minimum(a, b), np.maximum(a, b)], 1),
                          axis=0)

        # Merge neighbours whose mean colours are close. Union-find over a
        # few hundred nodes -- cheap enough to leave in Python.
        d = np.linalg.norm(mean[pairs[:, 0]] - mean[pairs[:, 1]], axis=1)
        parent = np.arange(n)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j in pairs[d < self._region_merge_dist]:
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

        root = np.array([find(i) for i in range(n)], dtype=np.int32)
        _, root = np.unique(root, return_inverse=True)
        return root[labels].astype(np.int32)

    def _draw_regions(self, frame, fg_mask=None):
        """Cel-shaded illustration of the real scene.

        The premise this exists for: a lens like tina's still delivers
        usable low-frequency colour and large-scale structure while
        resolving no fine texture at all. Sharpie and XDoG both answer that
        by throwing the colour away and drawing lines. This answers it by
        keeping the colour and throwing the *texture* away -- a couch
        becomes one coherent grey mass, the floor one brown mass, a plant
        one green mass, and a cat keeps its real black/grey/white inside a
        legible silhouette.

        HARD CONSTRAINT -- never invent colour or content. Every fill is
        the arithmetic mean of the actual pixels inside that region, taken
        from a smoothed low-frequency copy so sensor grain is not
        reproduced as fake detail. There is no palette, no colour transfer
        and no synthesis anywhere in this method. That is the same rule as
        the standing no-deblurring/no-super-resolution ban: invented detail
        becomes invented entities downstream, and they look authoritative.

        This is a DISPLAY stage only. It knows nothing about what a region
        contains and never reports one -- semantics come from the
        persistent scene model later, or never.
        """
        sc = self._region_scale
        small = (frame if sc >= 0.999 else
                 cv2.resize(frame, None, fx=sc, fy=sc,
                            interpolation=cv2.INTER_AREA))
        sh, sw = small.shape[:2]

        # 1. Low-frequency colour. Bilateral rather than a plain blur: the
        #    fills must not bleed across the boundaries between masses, or
        #    every region ends up tinted by its neighbours.
        low = cv2.bilateralFilter(small, self._region_bilateral_d,
                                  self._region_sigma_color,
                                  self._region_sigma_space)

        # 2. Regions, from colour coherence (see _region_labels).
        labels = self._region_labels(low)
        n = int(labels.max()) + 1
        if n <= 1:
            # Nothing separable. Returning the smoothed colour is honest:
            # it is what the sensor gave us.
            return cv2.resize(low, (frame.shape[1], frame.shape[0]),
                              interpolation=cv2.INTER_LINEAR)

        # 3. Fill each region with the mean of its OWN pixels.
        flat = labels.reshape(-1)
        cnt = np.bincount(flat, minlength=n).astype(np.float32)
        cnt[cnt == 0] = 1.0
        lowf = low.reshape(-1, 3).astype(np.float32)
        mean = np.empty((n, 3), np.float32)
        for c in range(3):
            mean[:, c] = np.bincount(flat, weights=lowf[:, c],
                                     minlength=n) / cnt
        filled = mean[labels].astype(np.uint8)

        # 4. Restrained boundary overlay. Drawn where the REGION LABEL
        #    changes, so the linework lands exactly where the fills change
        #    -- an independent DoG pass would put strokes inside a flat
        #    mass, which is the texture this renderer just discarded.
        if self._region_edge_alpha > 0.0:
            ridge = np.zeros((sh, sw), np.uint8)
            ridge[:, :-1] |= (labels[:, :-1] != labels[:, 1:]).astype(np.uint8) * 255
            ridge[:-1, :] |= (labels[:-1, :] != labels[1:, :]).astype(np.uint8) * 255

            # A moving subject is often a poor colour match against
            # whatever it is standing in front of, so its boundary can be
            # the weakest one in the frame. MOG2 already knows where it is
            # -- borrow that outline rather than trying to win it back
            # from colour alone.
            if self._region_fg_boundary and fg_mask is not None:
                fg_small = cv2.resize(fg_mask, (sw, sh),
                                      interpolation=cv2.INTER_NEAREST)
                k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                ridge = cv2.max(ridge, cv2.morphologyEx(
                    fg_small, cv2.MORPH_GRADIENT, k3))

            ink = cv2.cvtColor(ridge, cv2.COLOR_GRAY2RGB)
            filled = cv2.addWeighted(filled, 1.0, ink,
                                     -self._region_edge_alpha, 0)

        if filled.shape[:2] != frame.shape[:2]:
            filled = cv2.resize(filled, (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
        return filled

    # Styles this build can actually render. Anything else falls through to
    # the overlay path, so the HUD must report overlay -- not the string
    # somebody put in device.env. tina is configured CV_EDGE_STYLE=xdog on a
    # build with no xdog branch; a HUD that echoed the config would have
    # claimed XDOG while rendering DoG-overlay, which is exactly the
    # config-intent-versus-reality gap this line exists to close.
    _KNOWN_EDGE_STYLES = ('sharpie', 'xdog', 'regions')

    def _apply_grid_correction(self, frame):
        """Learn the codec grid once, then correct every frame at its known
        positions. Runs at the FRONT of the pipeline: this is a transport
        artifact, not scene content, so everything downstream -- denoise,
        tone, edges, MOG2, the detector -- should see the corrected frame
        rather than each having to cope with the grid separately.
        """
        if self._grid_prior is None:
            if len(self._grid_learn_buf) < self._grid_learn_frames:
                self._grid_learn_buf.append(frame)
                return frame          # uncorrected while still learning
            try:
                self._grid_prior = _cvo.estimate_grid_prior(self._grid_learn_buf)
            except Exception as exc:
                print(f"CV GRID: learn failed ({type(exc).__name__}: {exc}) "
                      "-- correction disabled", flush=True)
                self._grid_correct = False
                self._grid_learn_buf = []
                return frame
            self._grid_learn_buf = []   # release the frames
            if self._grid_prior is None:
                print("CV GRID: no block grid detected on this source -- "
                      "correction inactive", flush=True)
                self._grid_correct = False
                return frame
            print(f"CV GRID: learned period={self._grid_prior['period']} "
                  f"col_phase={self._grid_prior['col_phase']} "
                  f"excess={self._grid_prior['col_excess']} / "
                  f"row_phase={self._grid_prior['row_phase']} "
                  f"excess={self._grid_prior['row_excess']} "
                  f"strength={self._grid_strength}", flush=True)
        return _cvo.deblock_grid(frame, self._grid_prior, self._grid_strength)

    def _capability_line(self):
        """Right-aligned HUD summary of the CV stages actually running.

        Reads live state, never configuration intent: a stage that failed to
        initialise, or that this build does not implement, does not appear.
        """
        style = (self._edge_style if self._edge_style in self._KNOWN_EDGE_STYLES
                 else 'overlay')
        bits = []
        if self._edge_enabled:
            bits.append(style.upper())
        if self._grid_correct and self._grid_prior is not None:
            bits.append('DEBLOCK')
        if self._foveal_enabled:
            bits.append('FOVEAL')
        if getattr(self, '_scene_enabled', False) and self._scene_registered:
            bits.append('SCENE STALE' if self._scene_stale else 'SCENE 2D')
        if getattr(self, '_persist_enabled', False) and self._persist_store is not None:
            n = len(self._persist_store.persistent())
            bits.append(f'PERSISTENT 2D ({n})' if n else 'PERSISTENT 2D')
        line = ' \u00b7 '.join(bits)
        # Acuity reads as a qualifier on everything above it rather than
        # another item in the list, so it joins with "AT".
        if self._acuity_adapt and self._acuity_last is not None:
            at = f'AT ACUITY {self._detect_scale:.2f}'
            line = f'{line} {at}' if line else at
        return line

    def _detect_and_hud(self, source_frame, canvas):
        """Render-only: draws SHAKey telemetry from the tracker's current
        state. Detection itself runs asynchronously on its own thread (see
        _detection_loop/_detection_step below) -- this never calls detect()
        and never blocks the writer loop on it."""
        self._frame_n += 1
        tracks = []
        if self._tracker is not None:
            with self._tracker_lock:
                tracks = [
                    t for t in self._tracker.tracks.values()
                    if t.state != 'departed'
                ]
        if self._hud_enabled:
            canvas = _cvd.draw_hud(canvas, tracks,
                                   capabilities=self._capability_line())
            # HUD labels for furniture come from the region dictionary, not
            # live inference (CV Mode Phase 3) -- the live detector never
            # spends a pass trying to reconfirm a couch. Suppressed while
            # stale: a mislabelled, out-of-date box is worse than none.
            if self._scene_enabled and self._scene_regions and not self._scene_stale:
                canvas = _cvd.draw_scene_regions(canvas, self._scene_regions)
        return canvas, tracks

    # ── reference-image scene model (CV Mode Phase 3) ───────────────────
    def _load_scene_model(self):
        """Load the region dictionary + reference photo from
        CV_SCENE_REFERENCE. Missing, unparseable, or structurally invalid
        -> stays fully disabled (self._scene_regions empty,
        self._scene_homography None), same backward-compatible fallback
        every other signal in this pipeline uses. Logged either way so an
        operator can tell "off by choice" from "tried and failed" --
        NOT prefixed SHAKEY, that naming was retired.
        """
        path = self._scene_reference_path
        try:
            with open(path) as f:
                model = json.load(f)
            H = np.array(model['homography'], dtype=np.float64)
            if H.shape != (3, 3):
                raise ValueError(f"homography has shape {H.shape}, expected (3, 3)")
            regions = model.get('regions', [])

            ref_path = os.path.join(os.path.dirname(path), 'scene_reference.jpg')
            ref_bgr = cv2.imread(ref_path)
            if ref_bgr is None:
                raise ValueError(f"could not read reference photo at {ref_path}")

            self._scene_homography = H
            self._scene_regions = regions
            self._scene_registered = True
            self._scene_stale = False
            # Warped once, at load time -- the staleness check runs
            # periodically and reuses this rather than re-warping per check.
            self._scene_reference_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
            print(
                f"CV SCENE: loaded {len(regions)} region(s) from {path} "
                f"(source={model.get('source', '?')}, "
                f"inlier_ratio={model.get('inlier_ratio', '?')})",
                flush=True,
            )
        except Exception as exc:
            print(
                f"CV SCENE: no usable model at {path} "
                f"({type(exc).__name__}: {exc}) -- running un-primed",
                flush=True,
            )
            self._scene_homography = None
            self._scene_regions = []
            self._scene_registered = False

    def _set_scene_consistency(self, dets):
        """Overlap fraction against a registered STATIC region of a
        DIFFERENT class, 0.0-1.0 -- the "is this a plausible place for a
        subject" signal. A cat detection overlapping a stored couch region
        scores high; one with no overlapping region stays at the Detection
        default (1.0, neutral) -- unlabelled space is NOT penalised, since
        COCO has no 'wall' class to positively confirm emptiness, and
        walking across open floor is normal and must not be suppressed.
        """
        if not self._scene_regions:
            return
        for d in dets:
            dx1, dy1, dx2, dy2 = d.x, d.y, d.x + d.w, d.y + d.h
            d_area = max(1, d.w * d.h)
            best = 0.0
            for r in self._scene_regions:
                if r['label'] == d.cls or not r.get('static', True):
                    continue
                rx, ry, rw, rh = r['box']
                ix1, iy1 = max(dx1, rx), max(dy1, ry)
                ix2, iy2 = min(dx2, rx + rw), min(dy2, ry + rh)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                overlap = ((ix2 - ix1) * (iy2 - iy1)) / d_area
                best = max(best, overlap)
            if best > 0.0:
                d.scene_consistency = min(1.0, best)

    def _suppress_static_region_detections(self, dets):
        """Drop detections whose class matches an overlapping region's
        label -- the detector re-confirming furniture as itself is not
        interesting signal; a couch is not a subject. Distinct from
        _set_scene_consistency (which boosts DIFFERENT-class detections
        near a region): this is the other of the two consumers the region
        dictionary exists to serve."""
        if not self._scene_regions:
            return dets
        kept = []
        for d in dets:
            dx1, dy1, dx2, dy2 = d.x, d.y, d.x + d.w, d.y + d.h
            d_area = max(1, d.w * d.h)
            suppressed = False
            for r in self._scene_regions:
                if r['label'] != d.cls or not r.get('static', True):
                    continue
                rx, ry, rw, rh = r['box']
                ix1, iy1 = max(dx1, rx), max(dy1, ry)
                ix2, iy2 = min(dx2, rx + rw), min(dy2, ry + rh)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                if ((ix2 - ix1) * (iy2 - iy1)) / d_area > 0.5:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(d)
        return kept

    def _check_scene_staleness(self):
        """Periodic (CV_SCENE_STALE_CHECK_INTERVAL_S), not per-frame:
        correlate the current MOG2 background against the registered
        reference (warped once at load time) via downscaled grayscale mean
        absolute difference. Divergence past threshold marks the reference
        stale -- camera pivots are caught here; furniture moving is not
        (see cv_scene.py's module docstring) and needs a manual re-run.
        Runs on the detection thread, called from _detection_loop.
        """
        if not self._scene_registered or self._mog2 is None:
            return
        now = time.perf_counter()
        if now - self._scene_last_stale_check < self._scene_stale_interval:
            return
        self._scene_last_stale_check = now

        bg = self._mog2.getBackgroundImage()
        if bg is None:
            return
        bg_gray = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
        ref = self._scene_reference_gray
        if ref.shape != bg_gray.shape:
            ref = cv2.resize(ref, (bg_gray.shape[1], bg_gray.shape[0]))
        small_bg = cv2.resize(bg_gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        small_ref = cv2.resize(ref, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        diff = float(cv2.absdiff(small_bg, small_ref).mean())

        was_stale = self._scene_stale
        self._scene_stale = diff > self._scene_stale_threshold
        if self._scene_stale and not was_stale:
            print(
                f"CV SCENE: reference now STALE (diff={diff:.1f} > "
                f"threshold={self._scene_stale_threshold:.1f}) -- "
                "camera likely pivoted; falling back to un-primed behaviour "
                "until a re-register succeeds",
                flush=True,
            )
        elif was_stale and not self._scene_stale:
            print(f"CV SCENE: reference no longer stale (diff={diff:.1f})", flush=True)

    # ── detection thread (CV Mode Phase 2) ──────────────────────────────
    def _detection_loop(self):
        """Background daemon thread: consumes the latest published
        (frame, crop_box, fg_mask) at a fast, cheap poll rate and advances
        the accumulation state machine. How often an actual detect()
        forward pass fires is governed inside _detection_step by
        CV_DETECT_HZ, not by this loop's poll rate."""
        while True:
            with self._detect_slot_lock:
                slot = self._detect_slot
            if slot is not None:
                try:
                    self._detection_step(*slot)
                except Exception as exc:
                    print(
                        f"SHAKEY DETECT THREAD ERROR: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
            if self._scene_enabled:
                try:
                    self._check_scene_staleness()
                except Exception as exc:
                    print(f"CV SCENE: staleness check error: "
                          f"{type(exc).__name__}: {exc}", flush=True)
            time.sleep(_DETECT_POLL_INTERVAL_S)

    def _reset_accum(self):
        self._accum_box = None
        self._accum_buf.clear()
        self._accum_ref_gray = None
        self._accum_started = None

    def _acuity_scale(self, frame):
        """Detection input scale for this frame, from its own measured
        acuity. Re-measured every CV_DETECT_ACUITY_PERIOD cycles rather
        than per pass: a lens does not change, and the only thing that
        moves this number is lighting.

        Returns 1.0 (no rescale) when adaptation is off, so this is a
        no-op for any node that has not opted in.
        """
        if not self._acuity_adapt:
            return 1.0
        if self._acuity_last is None or self._acuity_n >= self._acuity_period:
            g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
            # Measured on a downscaled copy: variance-of-Laplacian is a
            # relative metric here, and this keeps the cost negligible.
            g = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            self._acuity_last = float(cv2.Laplacian(g, cv2.CV_64F).var())
            self._acuity_n = 0
            self._detect_scale = max(
                self._detect_scale_min,
                min(self._detect_scale_max,
                    self._acuity_last / self._acuity_divisor))
            print(f"CV ACUITY: varLap={self._acuity_last:.1f} "
                  f"-> detect scale {self._detect_scale:.2f}", flush=True)
        self._acuity_n += 1
        return self._detect_scale

    def _detect_input(self, img):
        """Apply the adapted scale: downscale with INTER_AREA (which
        averages away the noise band the optics never resolved), then
        restore the original dimensions so every downstream coordinate --
        crop offsets, fg_consistency, tracker positions -- keeps working
        in unchanged full-frame pixels. Returning a smaller image would
        mean translating boxes back, for no benefit: the detector
        letterboxes to its own fixed input size regardless.
        """
        s = self._acuity_scale(img)
        if s >= 0.999:
            return img
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_LINEAR)

    def _detection_step(self, frame, crop_box, fg_mask):
        """One poll-tick of the detection thread's state machine.

        Ungated (crop_box is None -- Foveal off, or MOG2 found nothing this
        instant): throttled full-frame fallback pass, same reasoning
        Phase 1 established for keeping a still-but-present subject
        eligible for re-verification rather than decaying toward
        unlabelled purely because it stopped moving. Throttled to
        CV_DETECT_HZ so a static room does not fire detect() on every
        50ms poll tick.

        Gated: accumulates crops from a FIXED rectangle (self._accum_box,
        set once when the window opens, not re-derived per sample) --
        that fixed-rectangle discipline is the alignment mechanism: every
        sample in a window is pixel-registered by construction, without
        needing real affine registration. A sample is only accepted into
        the average if the region's mean frame-to-frame difference is
        below CV_FOVEAL_ACCUM_DIFF_MAX (a moving cat would smear).
        """
        now = time.perf_counter()

        if crop_box is None:
            self._reset_accum()
            if now - self._last_detect_end < 1.0 / self._detect_hz:
                return
            dets = self._detector.detect(self._detect_input(frame))
            self._finish_detect_cycle(frame, dets, None, fg_mask)
            return

        if self._accum_box is None:
            if now - self._last_detect_end < 1.0 / self._detect_hz:
                return
            self._accum_box = crop_box
            self._accum_started = now

        cx, cy, cw, ch = self._accum_box
        crop = frame[cy:cy + ch, cx:cx + cw]
        if crop.size == 0:
            self._reset_accum()
            return

        small = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if self._accum_ref_gray is None or float(
                np.abs(gray - self._accum_ref_gray).mean()) <= self._accum_diff_max:
            self._accum_buf.append(crop)
            self._accum_ref_gray = gray

        timed_out = (now - self._accum_started) >= self._accum_timeout_s
        if len(self._accum_buf) < self._accum_frames and not timed_out:
            return  # still collecting

        if self._accum_buf:
            stack = np.stack(list(self._accum_buf), axis=0)
            averaged = stack[0] if len(self._accum_buf) == 1 else \
                np.mean(stack, axis=0).astype(np.uint8)
        else:
            # Timed out without a single aligned sample (subject never
            # settled at all within the window) -- detect on the raw
            # current crop rather than starve indefinitely. Worst case
            # degrades to Phase 1's single-frame behaviour, not a hang.
            averaged = crop

        dets = self._detector.detect(self._detect_input(averaged))
        for d in dets:
            d.x += cx
            d.y += cy
        self._finish_detect_cycle(frame, dets, self._accum_box, fg_mask)
        self._reset_accum()

    def _finish_detect_cycle(self, source_frame, dets, crop_box, fg_mask):
        """Common tail for both the gated and ungated-fallback paths:
        fg_consistency, coat classification, tracker update, promotion
        notification, logging. Runs on the detection thread; tracker
        access is locked because the render path reads it every display
        frame from a different thread."""
        if self._foveal_enabled and fg_mask is not None:
            self._set_fg_consistency(dets, fg_mask)

        if self._scene_enabled and self._scene_regions and not self._scene_stale:
            dets = self._suppress_static_region_detections(dets)
            self._set_scene_consistency(dets)

        for d in dets:
            d.cls = _cvd.classify_coat(source_frame, d)

        with self._tracker_lock:
            self._tracker.update(dets)
            promoted = [tr for tr in self._tracker.tracks.values()
                        if getattr(tr, 'promoted', False)]
            n_tracks = len(self._tracker.tracks)

        # Notify on promotion only. The evidence accumulator has already
        # decided this is real; the notifier needs the edge, and dedupes
        # per track itself. Enqueue-and-return -- the send runs on its own
        # thread, so a slow SMTP server cannot reach the detection thread.
        if self._notifier is not None:
            for tr in promoted:
                self._notifier.on_promotion(tr, source_frame)

        # Raw detector output logged SEPARATELY from tracker state, because
        # the two answer different questions and were previously conflated.
        # A HUD showing seven targets can mean the detector fired seven
        # times, or that the evidence accumulator is holding stale tracks
        # alive -- and the fix is different in each case. This line reports
        # what YOLO actually returned this pass, before association,
        # evidence promotion, or decay touched it.
        print(
            "CV RAW DETECT:"
            f" n={len(dets)}"
            f" detail={sorted(((d.cls, round(d.confidence, 3)) for d in dets), key=lambda x: -x[1])}",
            flush=True,
        )
        print(
            "SHAKEY DETECT:"
            f" detections={len(dets)}"
            f" tracks={n_tracks}"
            f" crop={crop_box}"
            f" accum={len(self._accum_buf)}"
            f" detail={[f"{d.cls}:{d.confidence:.3f}"
                        f"/fg={d.fg_consistency:.2f}" for d in dets]}",
            flush=True,
        )
        self._last_detect_end = time.perf_counter()

    def _draw_sharpie(self, frame):
        """Line drawing on white: quantise luminance, outline the bands.

        Reuses the luminance the tone stage already produced -- it has been
        denoised and CLAHE-equalised, which is exactly the input this wants,
        and re-deriving it would double the cost of the most expensive part.
        Equalisation matters: without it the bands all land inside the
        blown-out end of the histogram and the drawing loses the bright half
        of the scene.
        """
        h, w_px = frame.shape[:2]
        g = self._last_luma if self._last_luma is not None else \
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        sc = self._sharpie_scale
        if 0.1 < sc < 1.0:
            g = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        gh, gw = g.shape[:2]

        k = self._sharpie_smooth | 1
        g = cv2.medianBlur(g, k if k <= 9 else 9)

        step = max(1, 256 // self._sharpie_levels)
        ink = np.zeros((gh, gw), np.uint8)
        close_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._sharpie_close | 1, self._sharpie_close | 1))

        for lv in range(1, self._sharpie_levels):
            band = ((g >= lv * step).astype(np.uint8)) * 255
            # Close then open: joins a band broken by noise into one region,
            # then drops the specks that would otherwise each get outlined.
            band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, close_k)
            band = cv2.morphologyEx(band, cv2.MORPH_OPEN, close_k)
            cont, _ = cv2.findContours(band, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for c in cont:
                if cv2.arcLength(c, False) >= self._sharpie_min_len:
                    cv2.drawContours(ink, [c], -1, 255, self._sharpie_thick)

        if ink.shape[:2] != (h, w_px):
            ink = cv2.resize(ink, (w_px, h), interpolation=cv2.INTER_NEAREST)

        canvas = np.full_like(frame, 255)
        canvas[ink > 0] = (25, 25, 25)
        if self._sharpie_blend > 0.0:
            canvas = cv2.addWeighted(frame, self._sharpie_blend, canvas, 1.0 - self._sharpie_blend, 0)
        return canvas

    # ── lens-artifact subtraction ───────────────────────────────────────
    def relearn_artifacts(self):
        """Discard the current mask and start sampling again."""
        self._artifact_mask = None
        self._artifact_samples_buf = []
        self._artifact_learning = self._artifact_enabled

    def _collect_artifact_sample(self, luma, motion_mean):
        """Sample luminance while the scene is moving.

        Motion is the requirement, not an optimisation: the method separates
        lens from scene by exploiting that the scene changes and the glass
        does not. Sampling a still scene would bake furniture into the mask.
        """
        if not self._artifact_learning or luma is None:
            return
        if motion_mean < self._artifact_min_motion:
            return
        self._artifact_samples_buf.append(luma.copy())
        if len(self._artifact_samples_buf) >= self._artifact_samples:
            self._build_artifact_mask()

    def _build_artifact_mask(self):
        """Median across samples, high-passed, thresholded.

        Median rather than mean: a mean is dragged by whatever passed through
        frame, while a median discards anything not present in most samples.
        What survives is what was there the whole time -- the glass.
        """
        self._artifact_learning = False
        stack = np.stack(self._artifact_samples_buf, axis=0)
        self._artifact_samples_buf = []

        median = np.median(stack, axis=0).astype(np.uint8)
        # High-pass: artifacts are small and sharp; the scene's persistent
        # structure is broad. Subtracting a blurred copy keeps only the fine
        # detail that survived the median.
        low = cv2.GaussianBlur(median, (0, 0), sigmaX=4.0)
        high = cv2.absdiff(median, low)
        _, mask = cv2.threshold(high, self._artifact_threshold, 255, cv2.THRESH_BINARY)

        if self._artifact_dilate > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * self._artifact_dilate + 1, 2 * self._artifact_dilate + 1))
            mask = cv2.dilate(mask, k)

        # A mask covering a large fraction of the frame means the separation
        # failed -- a static camera on a still scene, or a threshold far too
        # low. Repairing that much of every frame would do more damage than
        # the dirt. Refuse it and leave the feed alone.
        area = float((mask > 0).sum()) / mask.size
        if area > self._artifact_max_area:
            self._artifact_mask = None
            return
        self._artifact_mask = mask

    def _subtract_artifacts(self, frame):
        """Fill masked pixels from their surroundings.

        A blur-and-composite rather than cv2.inpaint: inpainting solves a PDE
        per region and is far too expensive per frame on the hardware this
        targets, while the mask is sparse specks whose neighbourhoods are
        genuinely representative. One blur and one masked copy.
        """
        if self._artifact_mask is None:
            return frame
        filled = cv2.GaussianBlur(frame, (0, 0), sigmaX=3.0)
        result = frame.copy()
        cv2.copyTo(filled, self._artifact_mask, result)
        return result

    def _temporal_denoise(self, frame, moving_mask):
        """Temporally average against up to two prior frames, then restore
        the crisp current frame wherever the binary motion mask says a
        pixel is moving -- static background gets real noise reduction, a
        moving subject is composited back in untouched, never blended."""
        if not self._frames:
            return frame

        if len(self._frames) >= 2:
            # tina's tuned weights, carried over from its live install:
            # less temporal smearing than the HLSLS default 0.44/0.28/0.28.
            avg = cv2.addWeighted(frame, 0.60, self._frames[-1], 0.20, 0)
            avg = cv2.addWeighted(avg, 1.0, self._frames[-2], 0.20, 0)
        else:
            avg = cv2.addWeighted(frame, 0.5, self._frames[-1], 0.5, 0)

        if self._denoise_motion or moving_mask is None or not moving_mask.any():
            return avg

        # Still composite the crisp frame back over motion here: temporal
        # blending a moving subject ghosts it, which is a different and much
        # more visible failure than leaving it un-denoised.
        result = avg.copy()
        cv2.copyTo(frame, moving_mask, result)
        return result

    def _tone_correct(self, frame):
        """Highlight rolloff, then CLAHE, both on the luminance channel only
        -- fixes exposure/contrast without shifting color balance. Rolloff
        runs first so CLAHE's local histogram equalization has real headroom
        to work with in a tile that's mostly blown out, instead of a flat
        plateau of saturated values with nothing left to redistribute."""
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        # Kept for the sharpness metric: this is the denoised luminance,
        # before CLAHE stretches local contrast. Measuring after CLAHE would
        # report the enhancement rather than the underlying focus, and the
        # decision to draw edges has to be made about the source image.
        self._last_luma = l
        l = cv2.LUT(l, self._highlight_lut)
        l2 = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)

    def _sharpen(self, frame, moving_mask):
        """Unsharp mask via OpenCV's fused addWeighted, skipped wherever the
        motion mask is set. Motion blur there is a real optical effect from
        the exposure window, not missing detail -- sharpening it would
        amplify noise, not the subject."""
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=self._unsharp_sigma)
        amount = self._unsharp_amount
        sharpened = cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)

        # With CV_SHARPEN_MOTION the moving subject is sharpened like
        # everything else. That is the point on this camera: the cat is what
        # someone is watching, and excluding it left the subject the softest
        # thing in frame.
        if self._sharpen_motion or moving_mask is None or not moving_mask.any():
            return sharpened

        result = sharpened.copy()
        cv2.copyTo(frame, moving_mask, result)
        return result
