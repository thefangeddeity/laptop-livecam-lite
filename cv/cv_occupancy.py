#!/usr/bin/env python3
"""
cv_occupancy.py -- static structure from temporal evidence alone
(CV Mode Phase 4 §5).

No classifier is involved anywhere in this module, by design. It answers
only "what tends to stay spatially stable", which is a question a degraded
sensor can still answer, and deliberately not "what is this", which it
cannot.

Everything is computed on a long-run temporal median rather than live
frames. On a static camera the median is simply a better image than any
frame it was built from: independent noise falls as sqrt(N), so a few
hundred frames materially raises the contrast of exactly the large,
low-frequency region boundaries that survive a bad lens. It cannot restore
spatial frequencies the optics never passed -- blur is deterministic, not
noise -- and nothing here pretends otherwise.

MOG2 is used strictly as a change/occupancy sensor. No notion of what a
moving thing might be is encoded here; that decision belongs to the
persistence layer, which has the temporal evidence to make it.
"""

import numpy as np
import cv2

# Everything runs on a heavily downsampled grid. Structure at this scale
# is what a degraded lens can actually support, and it keeps the whole
# module off the frame budget -- this is long-run bookkeeping, not a
# per-frame stage.
GRID_W, GRID_H = 80, 45


def validity_mask(frames, max_frames=120, sd_floor=1.0, dark_luma=40.0):
    """Per-pixel sensor validity, derived from long-run temporal statistics.

    A pixel that never changes across a live scene is not reporting the
    room -- it is dead. Measured on tina, whose sensor film has peeled:
    x=[0,201] over the full height, 12.3% of the frame, temporal sd
    exactly 0.00 and luminance exactly 0. Nothing there is data.

    That matters because every stage downstream treats invariance as
    evidence of STRUCTURE. Left unmasked, the dead strip is the single
    most temporally stable thing in the frame, so it is promoted to the
    most confident persistent entity in the room -- which is precisely
    what happened before this existed: the top-ranked static entity on
    tina was a 224x720 block of nothing.

    Returns a bool array, True where the pixel is usable. Derived rather
    than configured, so it needs no per-node calibration and tracks a
    sensor that degrades further.
    """
    if not frames:
        return None
    sel = frames if len(frames) <= max_frames else [
        frames[i] for i in np.linspace(0, len(frames) - 1, max_frames).astype(int)]
    L = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in sel]).astype(np.float32)
    sd = L.std(axis=0)
    mean = L.mean(axis=0)
    # Dark AND invariant only. Saturation is deliberately NOT treated as
    # death: a blown-out highlight is an exposure problem, not a broken
    # sensor, and it sits on a real object. Including a `mean > 250` clause
    # here masked 15.4% of tina's frame and 26% of the couch itself -- the
    # very surface the scene model exists to find -- because that couch is
    # a bright clipped white.
    dead = (sd < sd_floor) & (mean < dark_luma)
    # Grow slightly: the boundary of a dead region is a hard synthetic edge
    # and the few pixels beside it are contaminated by it.
    dead = cv2.dilate(dead.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return ~dead


def estimate_grid_prior(frames, max_frames=60, period=8):
    """Learn the compression grid's signature so perception can DISCOUNT it
    rather than avoid it.

    The block grid a lossy codec leaves behind is not noise in the usual
    sense -- it is a highly structured, largely predictable artifact, and
    that makes it a candidate for being modelled and subtracted rather than
    dodged by choosing filter scales that step over it.

    Measured on tina's MJPEG feed (120 frames):
      * phase is deterministic -- gradient energy peaks at column mod 8 == 7,
        7.2x the median phase, and the SAME phase wins in 120/120 frames;
      * amplitude is stationary -- on/off-grid excess ratio 5.66 with a
        coefficient of variation of 0.025 across the sequence;
      * but it is NOT content-independent: on-grid energy correlates with
        local image detail at r=+0.58, which is expected since quantisation
        error scales with what is being quantised.

    That last point decides the model. A fixed subtractable grid image would
    be wrong; what is genuinely constant is WHERE the grid falls and roughly
    how much it inflates gradient energy there. So this returns a per-column
    and per-row weight in (0, 1] that says "discount gradients at these
    positions by this much", leaving amplitude to scale with local content
    naturally.

    Returns None when no grid is detectable -- an uncompressed or
    high-bitrate source should not be penalised for a grid it does not have.
    """
    if not frames or len(frames) < 2:
        return None
    sel = frames if len(frames) <= max_frames else [
        frames[i] for i in np.linspace(0, len(frames) - 1, max_frames).astype(int)]
    G = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in sel]

    def axis_prior(axis):
        prof = np.mean([np.abs(np.diff(g, axis=axis)).mean(axis=1 - axis)
                        for g in G], axis=0)
        by_phase = np.array([prof[c::period].mean() for c in range(period)])
        med = float(np.median(by_phase))
        if med <= 1e-6:
            return None, 0.0, 0
        peak = int(by_phase.argmax())
        excess = float(by_phase[peak] / med)
        if excess < 1.5:
            return None, excess, peak      # no meaningful grid on this axis
        w = np.ones(len(prof) + 1, np.float32)
        # Weight is the inverse of the phase's own excess, so a phase that
        # is 7x the median contributes ~1/7 as much evidence. Clamped so a
        # grid line never becomes negative evidence.
        for c in range(period):
            w[c::period] = float(np.clip(med / by_phase[c], 0.05, 1.0))
        return w, excess, peak

    col_w, col_exc, col_ph = axis_prior(1)
    row_w, row_exc, row_ph = axis_prior(0)
    if col_w is None and row_w is None:
        return None
    h, w_ = G[0].shape
    if col_w is None:
        col_w = np.ones(w_, np.float32)
    if row_w is None:
        row_w = np.ones(h, np.float32)
    return {
        'period': period,
        'col_weight': col_w[:w_],
        'row_weight': row_w[:h],
        'col_excess': round(col_exc, 3), 'col_phase': col_ph,
        'row_excess': round(row_exc, 3), 'row_phase': row_ph,
    }


def apply_grid_prior(gradient_map, prior, strength=0.6):
    """Discount a gradient/edge map at the learned grid positions.

    Deliberately multiplicative and separable: the artifact is a property of
    position, so it is applied as a positional weight and never as a
    subtraction of assumed content. Nothing is invented and nothing is
    removed that was not already attributable to the grid's location.

    `strength` exists because a full discount over-corrects. The weights are
    per-axis, so a pixel on a row grid line AND a column grid line takes the
    penalty twice -- at full strength that measured out as blockiness 5.62 ->
    1.03 but only 38% of total gradient energy retained. Real edges are not
    grid-aligned, so roughly 1/8 of any genuine edge sits on a grid line and
    is attenuated with it; a partial discount keeps that collateral small
    while still removing most of the bias. strength=0 disables, 1.0 is the
    full inverse-excess weighting.
    """
    if prior is None or strength <= 0.0:
        return gradient_map
    s = float(np.clip(strength, 0.0, 1.0))
    cw = 1.0 - s * (1.0 - prior['col_weight'])
    rw = 1.0 - s * (1.0 - prior['row_weight'])
    out = gradient_map.astype(np.float32)
    out = out * cw[None, :out.shape[1]]
    out = out * rw[:out.shape[0], None]
    return out


def deblock_grid(frame, prior, strength=0.6):
    """Attenuate the codec's block grid IN THE IMAGE, at known positions.

    A generic deblocking filter has to guess where block edges are and
    ends up smoothing everything, which on a soft sensor removes the little
    real detail that survived. Because estimate_grid_prior() has already
    established the phase deterministically (tina: column mod 8 == 7 in
    120/120 frames), this only touches the lines the grid actually falls on
    and leaves every other pixel bit-exact.

    On a grid line the value is pulled toward the mean of its two immediate
    neighbours. A real edge that happens to cross a grid line is only
    partially affected -- its neighbours carry the same edge, so their mean
    is close to the true value and the pull is small. A pure block
    discontinuity has dissimilar neighbours, so the pull is large. The
    correction is therefore self-limiting on genuine content.

    Per-phase pull is scaled by how anomalous that phase measured, so a
    phase barely above median is barely touched.
    """
    if prior is None or strength <= 0.0:
        return frame
    s = float(np.clip(strength, 0.0, 1.0))
    period = prior['period']

    # Cost discipline: an earlier version converted the whole frame to
    # float32 and indexed it per phase, which measured 92 ms/frame -- 74%
    # of tina's entire 125 ms budget at 8fps, for a correction that touches
    # about an eighth of the pixels. This version converts nothing globally
    # and only materialises the lines it actually modifies.
    out = frame.copy()

    def pull(axis, weights):
        n = out.shape[1] if axis == 1 else out.shape[0]
        for c in range(period):
            idx = np.arange(c, n, period)
            idx = idx[(idx > 0) & (idx < n - 1)]
            if idx.size == 0:
                continue
            w = float(np.clip(1.0 - weights[idx].mean(), 0.0, 1.0)) * s
            if w <= 0.01:
                continue          # this phase is not anomalous; leave it alone
            if axis == 1:
                a_ = out[:, idx - 1].astype(np.int16)
                b_ = out[:, idx + 1].astype(np.int16)
                cur = out[:, idx].astype(np.int16)
                out[:, idx] = (cur + (((a_ + b_) // 2 - cur) * w)
                               ).clip(0, 255).astype(np.uint8)
            else:
                a_ = out[idx - 1, :].astype(np.int16)
                b_ = out[idx + 1, :].astype(np.int16)
                cur = out[idx, :].astype(np.int16)
                out[idx, :] = (cur + (((a_ + b_) // 2 - cur) * w)
                               ).clip(0, 255).astype(np.uint8)

    pull(1, prior['col_weight'])
    pull(0, prior['row_weight'])
    return out


def _grid(mask):
    """Downsample a full-frame bool mask onto the occupancy grid."""
    if mask is None:
        return np.ones((GRID_H, GRID_W), bool)
    m = cv2.resize(mask.astype(np.uint8), (GRID_W, GRID_H),
                   interpolation=cv2.INTER_AREA)
    return m > 0


def temporal_median(frames, max_frames=300):
    """Long-run median. Median rather than mean so a subject that parks in
    frame for a while cannot drag the estimate of what is behind it."""
    if not frames:
        return None
    sel = frames if len(frames) <= max_frames else [
        frames[i] for i in np.linspace(0, len(frames) - 1, max_frames).astype(int)]
    return np.median(np.stack(sel).astype(np.float32), axis=0).astype(np.uint8)


def occupancy_stats(frames, var_threshold=150.0, scale=0.25, valid=None):
    """Per-cell long-run statistics on the GRID_W x GRID_H grid.

    Returns dict of HxW float arrays:
      fg_fraction     -- fraction of observations where this cell changed.
                         Its COMPLEMENT is the interesting signal: cells
                         that never change are structure. Free space is
                         where things move; occupied space is where they
                         do not.
      appearance_sd   -- temporal standard deviation of cell brightness.
                         Low = stable appearance.
      stability       -- fraction of observations within 1 sd of the
                         cell's own median: "how reliably does this cell
                         look like itself".
    """
    if not frames:
        return None
    mog = cv2.createBackgroundSubtractorMOG2(
        history=max(50, len(frames)), varThreshold=var_threshold,
        detectShadows=False)

    fg_hits = np.zeros((GRID_H, GRID_W), np.float32)
    lum = []
    for f in frames:
        small = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        m = mog.apply(small)
        m = cv2.resize(m, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
        fg_hits += (m > 0).astype(np.float32)
        g = cv2.cvtColor(cv2.resize(f, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_RGB2GRAY)
        lum.append(g.astype(np.float32))

    n = float(len(frames))
    L = np.stack(lum)
    vg = _grid(valid)
    med = np.median(L, axis=0)
    sd = L.std(axis=0)
    within = (np.abs(L - med) <= np.maximum(sd, 1e-6)).mean(axis=0)
    # Invalid cells are reported as maximally unstable so no downstream
    # stage can mistake dead silicon for a stable surface.
    stability = within.astype(np.float32)
    stability[~vg] = 0.0
    return {
        'fg_fraction': fg_hits / n,
        'appearance_sd': sd,
        'stability': stability,
        'median_luma': med.astype(np.float32),
        'valid': vg,
        'n_frames': int(n),
    }


def candidate_regions(stats, median_img, max_regions=12,
                      min_area_frac=0.006, max_area_frac=0.60,
                      max_fg_fraction=0.35, bands=6):
    """Propose static-structure regions from stability and boundaries only.

    Segmentation is by LUMINANCE BANDING of the temporal median, not by
    edge detection. That choice is forced by the hardware: on a degraded
    lens there is no reliable fine detail, and an edge-based segmenter
    either finds nothing or -- as measured during development -- floods
    into a single region spanning the whole frame, which then "explains"
    every object trivially. What such an image still has is large areas of
    distinct brightness, so banding traces the boundaries that actually
    survive. This is the same reasoning sharpie mode already rests on.

    Both bounds matter. `max_area_frac` is not tidiness: a region covering
    most of the frame contains every object by construction and would make
    any containment-based evaluation pass without the layer having learned
    anything. A proposer that can emit one is not a proposer.

    Returns boxes in FULL-FRAME pixel coordinates.
    """
    if stats is None or median_img is None:
        return []
    h, w = median_img.shape[:2]
    cells = GRID_W * GRID_H

    vg = stats.get('valid')
    if vg is None:
        vg = np.ones_like(stats['stability'], bool)
    # Percentile over VALID cells only -- a large dead area would otherwise
    # drag the threshold down and let genuinely unstable regions through.
    valid_stab = stats['stability'][vg]
    thresh = np.percentile(valid_stab, 35) if valid_stab.size else 0.0
    stable = ((stats['stability'] > thresh) &
              (stats['fg_fraction'] < max_fg_fraction) & vg)

    g = cv2.cvtColor(cv2.resize(median_img, (GRID_W, GRID_H),
                                interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    g = cv2.medianBlur(g, 3)
    lo, hi = float(g.min()), float(g.max())
    if hi - lo < 1.0:
        return []
    band = np.clip(((g - lo) / (hi - lo) * bands).astype(np.int32), 0, bands - 1)

    out = []
    for b in range(bands):
        mask = ((band == b) & stable).astype(np.uint8)
        if mask.sum() == 0:
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        nlab, labels, st, _ = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, nlab):
            x, y, bw, bh, area = st[i]
            frac = area / float(cells)
            if frac < min_area_frac or frac > max_area_frac:
                continue
            # A real object can legitimately fill half the frame -- tina's
            # couch does. What must still be rejected is the degenerate
            # blob that spans essentially the WHOLE frame in both axes and
            # therefore "contains" every object trivially. Size alone
            # cannot separate those two cases; spanning both dimensions
            # can.
            if bw >= 0.92 * GRID_W and bh >= 0.92 * GRID_H:
                continue
            sel = (labels[y:y+bh, x:x+bw] == i)
            sx, sy = w / float(GRID_W), h / float(GRID_H)
            out.append({
                'box': (int(x * sx), int(y * sy),
                        max(1, int(bw * sx)), max(1, int(bh * sy))),
                'grid_box': (int(x), int(y), int(bw), int(bh)),
                'area_cells': int(area),
                'area_frac': round(frac, 4),
                'band': b,
                'mean_fg_fraction': float(stats['fg_fraction'][y:y+bh, x:x+bw][sel].mean()),
                'mean_stability': float(stats['stability'][y:y+bh, x:x+bw][sel].mean()),
            })
    out.sort(key=lambda r: r['area_cells'], reverse=True)
    return out[:max_regions]
