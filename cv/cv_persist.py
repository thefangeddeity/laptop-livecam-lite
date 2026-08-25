#!/usr/bin/env python3
"""
cv_persist.py -- persistent scene entities (CV Mode Phase 4).

An entity's identity is the entity. A bounding box is one OBSERVATION of
it, in one coordinate system, at one moment. That distinction is the whole
point of this module and it is load-bearing: get it wrong and camera
motion, a replacement camera, externally-supplied photographs, and any
future depth representation all become rewrites rather than additions.

So every observation records the coordinate system it was made in
(`region.space`, currently always 'image'), and nothing outside this
module may treat an entity's current box as "what the entity is". The box
is the most recent place we saw it.

Why persistence carries semantic weight of its own: a region that holds
still for fourteen hours is part of the room, whether or not any
classifier can name it. Recurrence is evidence. This is the same principle
the run-14 evidence accumulator already applies within a few seconds --
this module applies it across hours and restarts, which is why it is a
separate mechanism with its own time constants rather than a longer decay
on the existing one. A leaky integrator tuned to drop a departed cat in
seconds cannot also express "present for two weeks".

Deliberately NOT in this module: any notion of what an entity *is*. The
detector's class, when there is one worth trusting, is attached as a
hypothesis. Below the semantic floor it is not attached at all -- see
Observation.klass and the module note on objectness in observe().
"""

import json
import os
import time
import uuid

# Promotion ladder. An entity climbs it by recurring, never by scoring
# well once.
STATE_TRANSIENT = 'transient'    # seen, not yet worth remembering
STATE_RECURRING = 'recurring'    # seen repeatedly in a consistent place
STATE_PERSISTENT = 'persistent'  # part of the room
STATES = (STATE_TRANSIENT, STATE_RECURRING, STATE_PERSISTENT)

# Kinematic character, orthogonal to the promotion ladder: a cat that
# lives on the couch is persistent AND dynamic.
KIND_UNKNOWN = 'unknown'
KIND_STATIC = 'static'
KIND_DYNAMIC = 'dynamic'


class Region:
    """A box in an explicitly-named coordinate space.

    `space` exists so that a future depth-capable or multi-camera node can
    add a new space without any consumer silently misreading old records
    as being in the new one. Today it is always 'image'.
    """

    __slots__ = ('space', 'x', 'y', 'w', 'h')

    def __init__(self, x, y, w, h, space='image'):
        self.space = space
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)

    @property
    def centre(self):
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def area(self):
        return max(0, self.w) * max(0, self.h)

    def iou(self, other):
        if self.space != other.space:
            raise ValueError(
                f"refusing to compare regions across coordinate spaces: "
                f"{self.space!r} vs {other.space!r}")
        ax2, ay2 = self.x + self.w, self.y + self.h
        bx2, by2 = other.x + other.w, other.y + other.h
        ix1, iy1 = max(self.x, other.x), max(self.y, other.y)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def to_dict(self):
        return {'space': self.space, 'x': self.x, 'y': self.y,
                'w': self.w, 'h': self.h}

    @staticmethod
    def from_dict(d):
        return Region(d['x'], d['y'], d['w'], d['h'], d.get('space', 'image'))


class Observation:
    """One sighting. Immutable by convention.

    `klass` is a detector hypothesis and is allowed to be None -- which is
    the correct value below the semantic floor, where the class head is
    noise with labels attached. `objectness` is always meaningful: it says
    something occupied this region, which is exactly the claim a weak
    detector can actually support.
    """

    __slots__ = ('t', 'region', 'objectness', 'klass', 'klass_confidence',
                 'appearance')

    def __init__(self, region, objectness, t=None, klass=None,
                 klass_confidence=0.0, appearance=None):
        self.t = time.time() if t is None else float(t)
        self.region = region
        self.objectness = float(objectness)
        self.klass = klass
        self.klass_confidence = float(klass_confidence)
        self.appearance = appearance   # small dict of cheap features, or None

    def to_dict(self):
        return {'t': self.t, 'region': self.region.to_dict(),
                'objectness': round(self.objectness, 4),
                'klass': self.klass,
                'klass_confidence': round(self.klass_confidence, 4),
                'appearance': self.appearance}

    @staticmethod
    def from_dict(d):
        return Observation(Region.from_dict(d['region']), d['objectness'],
                           t=d['t'], klass=d.get('klass'),
                           klass_confidence=d.get('klass_confidence', 0.0),
                           appearance=d.get('appearance'))


class Entity:
    """A thing the system believes exists, independent of any one sighting.

    The identity is `id` -- a UUID, deliberately not derived from position,
    so that an entity keeps its identity when it moves, when the camera is
    nudged, or when a future coordinate space replaces the current one.

    `user_label` attaches HERE, to the entity, never to a box. That is what
    makes "STATIC OBJECT #7 is the life-size Mo'ai" survive the box drifting
    by ten pixels, the camera being replaced, or the region being recomputed
    from a different observation space.

    `machine_class` and `user_label` are kept separate on purpose. A user
    calling something "grandma's chair" must not destroy the detector's
    "chair", and a detector later learning "chair" must not overwrite the
    user's name for it.
    """

    __slots__ = ('id', 'created', 'last_seen', 'observation_count', 'state',
                 'kind', 'machine_class', 'machine_class_confidence',
                 'user_label', '_recent', '_sum_cx', '_sum_cy', '_sum_w',
                 '_sum_h', '_sumsq_cx', '_sumsq_cy', '_fg_hits')

    def __init__(self, first_obs=None, entity_id=None):
        self.id = entity_id or f"e-{uuid.uuid4().hex[:12]}"
        self.created = time.time()
        self.last_seen = self.created
        self.observation_count = 0
        self.state = STATE_TRANSIENT
        self.kind = KIND_UNKNOWN
        self.machine_class = None
        self.machine_class_confidence = 0.0
        self.user_label = None
        # Bounded recent buffer for temporal inference; the running sums
        # below are the compact summary that survives it. Region statistics
        # must not require replaying every observation ever made.
        self._recent = []
        self._sum_cx = self._sum_cy = 0.0
        self._sum_w = self._sum_h = 0.0
        self._sumsq_cx = self._sumsq_cy = 0.0
        self._fg_hits = 0
        if first_obs is not None:
            self.observe(first_obs)

    # ---- observation ------------------------------------------------
    def observe(self, obs, recent_cap=64):
        """Fold one sighting in.

        Note what is NOT done here: no promotion decision, and no class
        assignment from a weak detection. Promotion is the tracker's job
        (it needs the whole population to judge), and class attachment is
        gated by the caller against the semantic floor -- this method
        trusts that obs.klass is already None when the detector was below
        the floor.
        """
        self.observation_count += 1
        self.last_seen = obs.t
        cx, cy = obs.region.centre
        self._sum_cx += cx
        self._sum_cy += cy
        self._sumsq_cx += cx * cx
        self._sumsq_cy += cy * cy
        self._sum_w += obs.region.w
        self._sum_h += obs.region.h
        if obs.objectness > 0:
            self._fg_hits += 1
        self._recent.append(obs)
        if len(self._recent) > recent_cap:
            del self._recent[0:len(self._recent) - recent_cap]
        # A class hypothesis only ever upgrades on strictly better evidence.
        if obs.klass is not None and obs.klass_confidence > self.machine_class_confidence:
            self.machine_class = obs.klass
            self.machine_class_confidence = obs.klass_confidence

    # ---- region statistics (the compact summary) --------------------
    @property
    def region_statistics(self):
        n = max(1, self.observation_count)
        mean_cx = self._sum_cx / n
        mean_cy = self._sum_cy / n
        var_cx = max(0.0, self._sumsq_cx / n - mean_cx * mean_cx)
        var_cy = max(0.0, self._sumsq_cy / n - mean_cy * mean_cy)
        return {
            'space': self._recent[-1].region.space if self._recent else 'image',
            'mean_centre': (round(mean_cx, 1), round(mean_cy, 1)),
            'centre_sd': (round(var_cx ** 0.5, 1), round(var_cy ** 0.5, 1)),
            'mean_size': (round(self._sum_w / n, 1), round(self._sum_h / n, 1)),
            'observations': self.observation_count,
        }

    @property
    def positional_sd(self):
        """Single scalar for how much this entity's position wanders --
        the statistic that separates furniture from a cat."""
        rs = self.region_statistics
        return (rs['centre_sd'][0] ** 2 + rs['centre_sd'][1] ** 2) ** 0.5

    @property
    def age_seconds(self):
        return max(0.0, self.last_seen - self.created)

    def current_region(self):
        """Most recent observed region. NOT the entity's identity -- callers
        must not cache this as though it were."""
        return self._recent[-1].region if self._recent else None

    # ---- labelling ---------------------------------------------------
    def set_user_label(self, label):
        """Attach a human name to the ENTITY. Does not touch machine_class:
        the two semantic layers coexist."""
        self.user_label = label

    @property
    def display_name(self):
        if self.user_label:
            return self.user_label
        if self.machine_class:
            return self.machine_class
        short = self.id.split('-')[-1][:4].upper()
        return f"{'STATIC' if self.kind == KIND_STATIC else 'OBJECT'} {short}"

    # ---- serialisation -----------------------------------------------
    def to_dict(self, include_recent=False):
        d = {
            'id': self.id, 'created': self.created, 'last_seen': self.last_seen,
            'observation_count': self.observation_count, 'state': self.state,
            'kind': self.kind, 'machine_class': self.machine_class,
            'machine_class_confidence': round(self.machine_class_confidence, 4),
            'user_label': self.user_label,
            'region_statistics': self.region_statistics,
            '_sums': [self._sum_cx, self._sum_cy, self._sum_w, self._sum_h,
                      self._sumsq_cx, self._sumsq_cy, self._fg_hits],
        }
        if include_recent:
            d['recent'] = [o.to_dict() for o in self._recent]
        return d

    @staticmethod
    def from_dict(d):
        e = Entity(entity_id=d['id'])
        e.created = d['created']
        e.last_seen = d['last_seen']
        e.observation_count = d['observation_count']
        e.state = d['state']
        e.kind = d['kind']
        e.machine_class = d.get('machine_class')
        e.machine_class_confidence = d.get('machine_class_confidence', 0.0)
        e.user_label = d.get('user_label')
        s = d.get('_sums') or [0, 0, 0, 0, 0, 0, 0]
        (e._sum_cx, e._sum_cy, e._sum_w, e._sum_h,
         e._sumsq_cx, e._sumsq_cy, e._fg_hits) = s
        e._recent = [Observation.from_dict(o) for o in d.get('recent', [])]
        return e


# Promotion thresholds. These are counts and seconds, NOT a decay rate --
# the whole reason this is separate from the run-14 evidence accumulator.
DEFAULTS = {
    'CV_PERSIST_ENABLED':        0,
    'CV_PERSIST_PATH':           '/var/lib/hls-livecam/scene_entities.json',
    'CV_PERSIST_MATCH_IOU':      0.30,   # association threshold
    'CV_PERSIST_RECENT_CAP':     64,     # bounded per-entity buffer
    'CV_PERSIST_RECUR_COUNT':    5,      # -> recurring
    'CV_PERSIST_PERSIST_COUNT':  50,     # -> persistent (with age below)
    'CV_PERSIST_PERSIST_AGE_S':  600.0,  # both count AND age required
    'CV_PERSIST_STATIC_SD':      12.0,   # positional sd below this = static
    'CV_PERSIST_FORGET_S':       604800.0,  # 7d unseen and never persistent
    'CV_PERSIST_CHECKPOINT_N':   50,     # write after N observations...
    'CV_PERSIST_CHECKPOINT_S':   300.0,  # ...or N seconds, whichever first
}


class EntityStore:
    """Association, promotion, and checkpointed persistence.

    Association is by IoU in the observation's own coordinate space. That
    is a deliberate limitation and it is stated rather than hidden: it
    works because these cameras are static. A node that gains camera
    motion needs a motion model here, and the entity/region split exists
    precisely so that change lands in this class instead of rippling
    through every consumer.
    """

    def __init__(self, cfg=None):
        c = dict(DEFAULTS)
        if cfg:
            for k, v in cfg.items():
                if k in c:
                    c[k] = type(c[k])(v) if not isinstance(c[k], str) else v
        self.cfg = c
        self.entities = {}
        self._since_checkpoint = 0
        self._last_checkpoint = time.time()

    # ---- association + promotion -------------------------------------
    def observe_many(self, observations):
        """Fold a batch of sightings in, returning the entities touched."""
        touched = []
        for obs in observations:
            touched.append(self._observe_one(obs))
        self._promote()
        self._maybe_checkpoint()
        return touched

    def _observe_one(self, obs):
        best, best_iou = None, 0.0
        for e in self.entities.values():
            cur = e.current_region()
            if cur is None or cur.space != obs.region.space:
                continue
            i = cur.iou(obs.region)
            if i > best_iou:
                best, best_iou = e, i
        if best is not None and best_iou >= self.cfg['CV_PERSIST_MATCH_IOU']:
            best.observe(obs, recent_cap=self.cfg['CV_PERSIST_RECENT_CAP'])
            self._since_checkpoint += 1
            return best
        e = Entity(obs)
        self.entities[e.id] = e
        self._since_checkpoint += 1
        return e

    def _promote(self):
        for e in self.entities.values():
            n, age = e.observation_count, e.age_seconds
            if (n >= self.cfg['CV_PERSIST_PERSIST_COUNT']
                    and age >= self.cfg['CV_PERSIST_PERSIST_AGE_S']):
                e.state = STATE_PERSISTENT
            elif n >= self.cfg['CV_PERSIST_RECUR_COUNT']:
                e.state = STATE_RECURRING
            else:
                e.state = STATE_TRANSIENT
            # Kinematic character is independent of the promotion ladder.
            if e.observation_count >= self.cfg['CV_PERSIST_RECUR_COUNT']:
                e.kind = (KIND_STATIC
                          if e.positional_sd <= self.cfg['CV_PERSIST_STATIC_SD']
                          else KIND_DYNAMIC)

    def forget_stale(self, now=None):
        """Drop transient entities never seen again. Persistent entities and
        anything a human has named are never forgotten automatically -- a
        user label is a statement that this thing matters."""
        now = time.time() if now is None else now
        cutoff = self.cfg['CV_PERSIST_FORGET_S']
        drop = [k for k, e in self.entities.items()
                if e.state != STATE_PERSISTENT and e.user_label is None
                and (now - e.last_seen) > cutoff]
        for k in drop:
            del self.entities[k]
        return len(drop)

    # ---- queries ------------------------------------------------------
    def persistent(self):
        return [e for e in self.entities.values() if e.state == STATE_PERSISTENT]

    def static_regions(self):
        """What the scene model consumes: persistent, non-moving entities."""
        return [e for e in self.entities.values()
                if e.state == STATE_PERSISTENT and e.kind == KIND_STATIC]

    def label(self, entity_id, text):
        e = self.entities.get(entity_id)
        if e is None:
            return False
        e.set_user_label(text)
        self.checkpoint()      # a human action is always worth a write
        return True

    # ---- persistence ---------------------------------------------------
    def _maybe_checkpoint(self):
        """Write on events and counts, never per observation."""
        if (self._since_checkpoint >= self.cfg['CV_PERSIST_CHECKPOINT_N']
                or (time.time() - self._last_checkpoint)
                >= self.cfg['CV_PERSIST_CHECKPOINT_S']):
            self.checkpoint()

    def checkpoint(self):
        path = self.cfg['CV_PERSIST_PATH']
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = {
                'version': 1,
                'written': time.time(),
                'entities': [e.to_dict(include_recent=True)
                             for e in self.entities.values()],
            }
            # Write-then-rename: a power loss mid-write leaves the previous
            # checkpoint intact rather than a truncated file that would
            # read as "no entities" and silently discard the room.
            tmp = path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            self._since_checkpoint = 0
            self._last_checkpoint = time.time()
            return True
        except Exception as exc:
            print(f"CV PERSIST: checkpoint failed: {type(exc).__name__}: {exc}",
                  flush=True)
            return False

    def load(self):
        path = self.cfg['CV_PERSIST_PATH']
        try:
            with open(path) as f:
                payload = json.load(f)
            self.entities = {}
            for d in payload.get('entities', []):
                e = Entity.from_dict(d)
                self.entities[e.id] = e
            print(f"CV PERSIST: loaded {len(self.entities)} entities from {path}",
                  flush=True)
            return True
        except FileNotFoundError:
            return False
        except Exception as exc:
            print(f"CV PERSIST: could not load {path} "
                  f"({type(exc).__name__}: {exc}) -- starting empty", flush=True)
            return False


# ── the semantic floor (CV Mode Phase 4 §6) ────────────────────────────
#
# Below a node's detector confidence floor, YOLO's class head is a noise
# source with labels attached: on tina, raw per-class maxima sit at
# 0.001-0.045 for EVERY class including ones nothing in the room could
# produce, and the ordering between them tracks class prior and whatever
# texture the blur happened to leave, not evidence. CV_DETECT_CONF=0.01
# does not lower a bar; it removes one.
#
# That matters more here than anywhere else in the pipeline, because
# persistence is an evidence AMPLIFIER. A detector that is wrong in the
# same direction every time -- and a systematically biased detector is
# exactly what an out-of-distribution input produces -- would have its
# error integrated into a confident, long-lived, named entity. The system
# would then present a fabricated memory with all the authority of a
# real one.
#
# So the floor is enforced structurally rather than by convention: this is
# the ONLY sanctioned way to turn a detection into an Observation, and it
# physically cannot carry a class below the floor.

def observation_from_detection(region, confidence, floor, klass=None,
                               t=None, appearance=None):
    """Build an Observation, dropping the class hypothesis below `floor`.

    Above the floor: the class is carried as a hypothesis, still subject
    to the entity's own "only upgrade on better evidence" rule.

    Below it: `objectness` survives, `klass` does not. "Something occupied
    this region" is a claim a weak detector genuinely supports. "That
    something was a cat" is not.
    """
    keep = klass if (klass is not None and confidence >= floor) else None
    return Observation(
        region=region,
        objectness=float(confidence),
        t=t,
        klass=keep,
        klass_confidence=float(confidence) if keep is not None else 0.0,
        appearance=appearance,
    )
