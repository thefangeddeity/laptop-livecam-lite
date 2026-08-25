"""The world model: sectors with floor heights, and things standing on them.

Doom's model, and it is the right one here for the reason Doom picked it --
it is 2D with heights, not 3D. A sector is a polygon with a floor height. A
thing is an (x, y) with a facing. A cat on the couch is a thing whose
position falls inside a sector whose floor is at 40cm; it is *on* a raised
sector, not floating in a reconstructed volume. There is no depth model, no
point cloud, and nothing to reconstruct -- which is why this ran on a 386.

THE CORE INVERSION (Phase 6 §4). In HLSLS the display was a view of live
evidence: the evidence accumulator decayed, so the render faded. Here the
map owns the truth and the camera is something that occasionally writes to
it. Concretely:

  * Nothing in this module decays, times out, or fades.
  * A thing stays exactly where it is until an observation moves it.
  * If the feed dies, the last known world is still the world. Frozen, not
    fading. `World.observe()` simply stops being called.
  * `World.revision` increments only on an actual change, so the renderer
    can redraw on change rather than per frame.

Decay still exists upstream, in the tracker's association logic. It just no
longer reaches the display.

Map coordinates are centimetres in the room's own frame, x to the right and
y away from the camera. They are not pixels and not normalised: heights are
in the same unit as positions, so "the couch floor is 40" and "the cat is
35cm from the couch's front edge" are directly comparable.
"""

import json
import os
import time
import uuid

# Asserted defaults, not measurements. Same discipline as the Phase 3 region
# dictionary: the map is ASSERTED, and the PM corrects it. Nothing in this
# pipeline can measure a floor height from a single fixed camera, and a
# number that looks measured but was guessed is worse than one that is
# openly a default.
DEFAULT_FLOOR_HEIGHT_CM = {
    'floor': 0.0,
    'rug': 2.0,
    'bed': 55.0,
    'couch': 40.0,
    'chair': 45.0,
    'dining table': 74.0,
    'table': 74.0,
    'tv': 60.0,
    'potted plant': 0.0,
    'shelf': 0.0,
}


def _new_id(prefix):
    return f'{prefix}-{uuid.uuid4().hex[:10]}'


def point_in_polygon(x, y, poly):
    """Standard ray-cast. Sectors are small and few; this is not hot."""
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            xint = (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            if x < xint:
                inside = not inside
        j = i
    return inside


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


class Sector:
    """A polygon with a floor height. The room floor is one of these."""

    __slots__ = ('sector_id', 'label', 'polygon', 'floor_height',
                 'source', 'confidence', 'locked')

    def __init__(self, polygon, floor_height=0.0, label='floor',
                 sector_id=None, source='proposed', confidence=0.0,
                 locked=False):
        self.sector_id = sector_id or _new_id('s')
        self.label = label
        self.polygon = [(float(x), float(y)) for x, y in polygon]
        self.floor_height = float(floor_height)
        # 'proposed'  -- derived automatically, never confirmed
        # 'confirmed' -- the PM looked at it and kept or corrected it
        self.source = source
        self.confidence = float(confidence)
        # A locked sector is never replaced by a new proposal run.
        self.locked = bool(locked)

    @property
    def area(self):
        return polygon_area(self.polygon)

    def contains(self, x, y):
        return point_in_polygon(x, y, self.polygon)

    def to_dict(self):
        return {'sector_id': self.sector_id, 'label': self.label,
                'polygon': [list(p) for p in self.polygon],
                'floor_height': self.floor_height, 'source': self.source,
                'confidence': self.confidence, 'locked': self.locked}

    @classmethod
    def from_dict(cls, d):
        return cls(d['polygon'], d.get('floor_height', 0.0),
                   d.get('label', 'floor'), d.get('sector_id'),
                   d.get('source', 'proposed'), d.get('confidence', 0.0),
                   d.get('locked', False))


class Thing:
    """Something standing somewhere. Drawn as a token, not a box.

    `x, y` are map centimetres. `facing` is degrees clockwise from +y (away
    from camera); it is only ever set when motion actually implies it, and
    stays None otherwise rather than being invented.
    """

    __slots__ = ('thing_id', 'kind', 'label', 'x', 'y', 'facing',
                 'last_seen', 'observations', 'confidence', 'sector_id')

    def __init__(self, x, y, kind='unknown', label=None, thing_id=None,
                 facing=None, last_seen=None, observations=1,
                 confidence=0.0, sector_id=None):
        self.thing_id = thing_id or _new_id('t')
        self.kind = kind
        self.label = label
        self.x = float(x)
        self.y = float(y)
        self.facing = facing
        self.last_seen = last_seen if last_seen is not None else time.time()
        self.observations = int(observations)
        self.confidence = float(confidence)
        self.sector_id = sector_id

    @property
    def display_name(self):
        return self.label or self.kind.upper()

    def to_dict(self):
        return {'thing_id': self.thing_id, 'kind': self.kind,
                'label': self.label, 'x': self.x, 'y': self.y,
                'facing': self.facing, 'last_seen': self.last_seen,
                'observations': self.observations,
                'confidence': self.confidence, 'sector_id': self.sector_id}

    @classmethod
    def from_dict(cls, d):
        return cls(d['x'], d['y'], d.get('kind', 'unknown'), d.get('label'),
                   d.get('thing_id'), d.get('facing'), d.get('last_seen'),
                   d.get('observations', 1), d.get('confidence', 0.0),
                   d.get('sector_id'))


class World:
    """Sectors, things, and the floor homography that puts camera
    observations into map coordinates.

    This object IS the display state. It is written by observation and read
    by the renderer, and it never changes on its own -- no timer, no decay,
    no expiry. Every mutating method returns True if it actually changed
    something, and bumps `revision` when it does.
    """

    def __init__(self, path=None, units='cm'):
        self.path = path
        self.units = units
        self.sectors = {}
        self.things = {}
        self.homography = None       # 3x3 list-of-lists, image px -> map cm
        self.homography_quality = None
        self.revision = 0
        self.updated = None
        # Association radius in cm. A new observation within this distance
        # of an existing thing of the same kind updates it rather than
        # creating a second one -- the map-space analogue of the tracker's
        # max_distance, but expressed in room units instead of pixels.
        self.assoc_radius_cm = 60.0
        # Movement below this is treated as jitter and does NOT bump the
        # revision, so a stationary cat does not force a redraw every pass.
        self.still_radius_cm = 8.0

    # ---- sectors -------------------------------------------------------

    def add_sector(self, sector):
        self.sectors[sector.sector_id] = sector
        self._touch()
        return sector.sector_id

    def sector_at(self, x, y):
        """Highest-floor sector containing the point, or None.

        Highest wins because that is what standing on something means: a
        cat inside both the room floor polygon and the couch polygon is on
        the couch. This one line is the whole reason the model needs no
        depth reasoning.
        """
        best = None
        for s in self.sectors.values():
            if s.contains(x, y):
                if best is None or s.floor_height > best.floor_height:
                    best = s
        return best

    def floor_sector(self):
        """The largest sector at height 0 -- the room floor."""
        cands = [s for s in self.sectors.values() if s.floor_height <= 0.01]
        return max(cands, key=lambda s: s.area) if cands else None

    # ---- observation ---------------------------------------------------

    def observe(self, x, y, kind='unknown', confidence=0.0, facing=None,
                t=None):
        """Fold one map-space observation into the world.

        Returns True if the world changed. This is the ONLY way a thing
        moves. There is deliberately no counterpart that removes or fades a
        thing with the passage of time.
        """
        t = time.time() if t is None else t
        best, best_d = None, self.assoc_radius_cm
        for th in self.things.values():
            if th.kind != kind:
                continue
            d = ((th.x - x) ** 2 + (th.y - y) ** 2) ** 0.5
            if d < best_d:
                best, best_d = th, d

        if best is None:
            th = Thing(x, y, kind=kind, confidence=confidence, facing=facing,
                       last_seen=t)
            s = self.sector_at(x, y)
            th.sector_id = s.sector_id if s else None
            self.things[th.thing_id] = th
            self._touch(t)
            return True

        moved = best_d > self.still_radius_cm
        best.observations += 1
        best.last_seen = t
        best.confidence = confidence
        if facing is not None:
            facing_changed = best.facing != facing
            best.facing = facing
        else:
            facing_changed = False
        if moved:
            best.x, best.y = float(x), float(y)
            s = self.sector_at(x, y)
            sid = s.sector_id if s else None
            sector_changed = sid != best.sector_id
            best.sector_id = sid
        else:
            sector_changed = False

        if moved or sector_changed or facing_changed:
            self._touch(t)
            return True
        return False

    def forget(self, thing_id):
        """Explicit removal. There is no automatic equivalent, on purpose:
        a thing leaves the map when something says it left, not because a
        clock ran out."""
        if self.things.pop(thing_id, None) is not None:
            self._touch()
            return True
        return False

    # ---- persistence ---------------------------------------------------

    def _touch(self, t=None):
        self.revision += 1
        self.updated = time.time() if t is None else t

    def to_dict(self):
        return {
            'version': 1, 'units': self.units, 'revision': self.revision,
            'updated': self.updated,
            'assoc_radius_cm': self.assoc_radius_cm,
            'still_radius_cm': self.still_radius_cm,
            'homography': self.homography,
            'homography_quality': self.homography_quality,
            'sectors': [s.to_dict() for s in self.sectors.values()],
            'things': [t.to_dict() for t in self.things.values()],
        }

    def save(self, path=None):
        """Write-then-rename with fsync. The map is the only durable state
        this system has; a torn write loses the room."""
        p = path or self.path
        if not p:
            raise ValueError('no path to save to')
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        tmp = p + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        return p

    @classmethod
    def load(cls, path):
        w = cls(path=path)
        if not os.path.exists(path):
            return w
        with open(path) as f:
            d = json.load(f)
        w.units = d.get('units', 'cm')
        w.revision = d.get('revision', 0)
        w.updated = d.get('updated')
        w.assoc_radius_cm = d.get('assoc_radius_cm', 60.0)
        w.still_radius_cm = d.get('still_radius_cm', 8.0)
        w.homography = d.get('homography')
        w.homography_quality = d.get('homography_quality')
        for sd in d.get('sectors', []):
            s = Sector.from_dict(sd)
            w.sectors[s.sector_id] = s
        for td in d.get('things', []):
            t = Thing.from_dict(td)
            w.things[t.thing_id] = t
        return w

    def bounds(self):
        """(min_x, min_y, max_x, max_y) over all sector geometry."""
        pts = [p for s in self.sectors.values() for p in s.polygon]
        if not pts:
            return (0.0, 0.0, 100.0, 100.0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))
