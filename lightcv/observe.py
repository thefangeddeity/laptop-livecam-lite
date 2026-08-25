"""The bridge: camera observations -> writes into the map.

This is the only place the camera is allowed to touch the world, and it is
deliberately thin. Everything upstream of it -- enhancement, artifact
learning, grid correction, illumination field, acuity adaptation, MOG2, the
detector, the evidence accumulator -- is carried over from HLSLS unchanged
and still does its job of producing the best possible frame and the most
trustworthy tracks.

What changed is only what happens after. In HLSLS a track WAS the display,
so when its evidence decayed the picture faded. Here a track is a witness:
it reports a ground-contact position, the map records it, and the map then
owns that fact regardless of what the track does next. A track that decays
out of existence does not erase what it told us.
"""

import time

from .world import DEFAULT_FLOOR_HEIGHT_CM


class Observer:
    """Turns tracker output into map writes.

    `min_evidence` gates which tracks are allowed to write at all. It is the
    accumulator's own promotion signal, reused: a track that never
    accumulated enough evidence to be worth labelling on a HUD is not worth
    writing into durable world state either.
    """

    def __init__(self, world, floor, min_evidence=0.30,
                 semantic_floor=0.25):
        self.world = world
        self.floor = floor
        self.min_evidence = float(min_evidence)
        # Same rule as HLSLS Phase 4 §6: below this, a detection establishes
        # that SOMETHING is there, never WHAT. It writes a thing of kind
        # 'unknown' rather than a named one.
        self.semantic_floor = float(semantic_floor)
        self.written = 0
        self.skipped_above_horizon = 0
        self.skipped_weak = 0

    def _height_under(self, box):
        """Floor height of the sector the box's contact point falls in.

        Chicken-and-egg, resolved in two steps: project onto the floor plane
        first to find which sector that lands in, then re-project using that
        sector's height. One iteration is enough -- the correction moves the
        point along the view ray, and sectors are much larger than the
        residual.
        """
        p0 = self.floor.contact_to_map(box, 0.0)
        if p0 is None:
            return 0.0
        s = self.world.sector_at(*p0)
        return s.floor_height if s else 0.0

    def observe_track(self, track, t=None):
        """Fold one track into the map. Returns True if the world changed."""
        ev = getattr(track, 'evidence', None)
        if ev is not None and ev < self.min_evidence:
            self.skipped_weak += 1
            return False

        conf = float(getattr(track, 'confidence', 0.0))
        kind = getattr(track, 'cls', 'unknown')
        if conf < self.semantic_floor:
            kind = 'unknown'

        box = tuple(track.box)
        h = self._height_under(box)
        p = self.floor.contact_to_map(box, h)
        if p is None:
            # Above the horizon: there is no floor position for this, and a
            # fabricated one would place a thing behind the camera.
            self.skipped_above_horizon += 1
            return False

        self.written += 1
        return self.world.observe(p[0], p[1], kind=kind, confidence=conf,
                                  t=t if t is not None else time.time())

    def observe_tracks(self, tracks, t=None):
        changed = False
        for tr in tracks:
            if getattr(tr, 'state', 'present') == 'departed':
                continue
            changed = self.observe_track(tr, t=t) or changed
        return changed
