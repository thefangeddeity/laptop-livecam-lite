# laptop-livecam-lite

A fork of `laptop-livecam` that stops trying to make a degraded camera
produce a good picture, and instead uses it to maintain a **map**.

Runs on tina only. No packaging, no versioning, no AUR, no deb. If this goes
nowhere, nothing has to be unwound.

## What carried over unchanged

The entire observation stack, in `cv/`: enhancement, artifact learning, grid
correction, illumination field, acuity adaptation, MOG2, the detector, the
evidence accumulator. None of that work is discarded — the detector still
needs the best possible frame, and this fork does nothing to help it get one
that LiveCam was not already doing.

`cv/cv_processor.py` additionally carries tina's own local tuning: the
temporal-denoise weights `0.60 / 0.20 / 0.20` from its live install, rather
than LiveCam's default `0.44 / 0.28 / 0.28`.

## What changed

Only what happens *after* detection.

### The world model — sectors with floor heights

Doom's model, chosen for the reason Doom chose it: it is 2D with heights,
not 3D.

- **Sector** — a polygon with a floor height. The room floor is height 0.
  The couch is a sector at ~40. Same primitive throughout.
- **Thing** — an (x, y) in map centimetres plus an optional facing, drawn as
  a token.

A cat on the couch is a thing whose position falls inside a sector whose
floor is at 40cm. `World.sector_at()` returns the *highest* sector
containing the point, which is the whole of the reasoning — no depth model,
no reconstruction, no special cases.

### The map is state, not a view of live evidence

This is the core requirement and it is enforced structurally:

- Nothing in `lightcv/world.py` decays, expires, or fades.
- A thing stays where it is until an observation moves it.
- Feed dies → the last known world is still the world. **Frozen, not
  fading.** `World.observe()` simply stops being called.
- `World.revision` increments only on a real change, and `RenderCache`
  redraws only when it does.

The evidence accumulator still decays — it is still what associates tracks
frame to frame. It just no longer reaches the display.

### Floor-plane homography

Phase 3b died trying to register a photograph against a live frame, and was
right to: a room is not a plane. A **floor** is a plane, which is the one
case a homography models exactly. Four floor corners, and every detection's
bottom edge maps to a floor position.

Two honesties in `lightcv/floor.py`:

- A 4-point solve is exact, so its residuals are ~0 *by construction*. That
  is reported as `exact_solve: true` with a note, not as a quality score.
- A thing standing on something raised does not touch the floor plane.
  `_raise_to_height` corrects for it — but only when the camera's ground
  position is known, and it **skips the correction rather than applying it
  against the wrong origin** when that is missing.

## Layout

    lightcv/world.py     sectors, things, the map as state
    lightcv/floor.py     floor-plane homography and projection
    lightcv/render.py    top-down renderer + render-on-change cache
    lightcv/propose.py   build a map proposal from collected evidence
    lightcv/observe.py   the one place the camera writes to the map
    bin/lightcv-api      map server + editor  (stdlib http.server)
    bin/lightcv-propose  emit a proposal from captured frames
    bin/lightcv-selftest end-to-end checks
    web/map.html         map view + drag-to-correct editor
    cv/                  the carried-over observation stack

## Running

    python3 bin/lightcv-selftest         # verify the model
    python3 bin/lightcv-propose          # propose from captured frames
    python3 bin/lightcv-api              # serve on :5100

Then open `http://tina:5100/`, trace the four floor corners over the camera
backdrop, set the room's real width and depth in cm, and save.

## What is asserted, not measured

Stated plainly because the distinction has bitten this project before:

- **Floor heights** come from a class-keyed default table. A fixed monocular
  camera cannot measure them.
- **Room dimensions** and the **camera height / ground position** are
  operator input.
- **Every proposed sector** is a guess from a detector run on a phone photo,
  drawn dashed with a `?` until confirmed.

## What did not work

`floor_quad_from_occupancy` was written first, on the premise that the large
stable region at the bottom of the frame is the floor. Measured on tina:
41.6% of occupancy cells qualify as stable and the largest stable component
covers 30.6% of the frame. In a room where nothing moved during the
observation, *everything* is stable, so it cannot separate floor from couch —
and on tina's geometry the bottom of the frame is couch, not floor. The
function is kept for a camera where that premise holds; `propose` uses
`floor_candidates_from_colour` instead, and ranks candidates rather than
choosing.

The signal that *would* separate them is accumulated foot traffic — where
things actually stand, over days. That is exactly what `Observer` writes into
the map, so the map should get better at knowing its own floor the longer it
runs.

## Licence

GPL-3.0-or-later, the same as `laptop-livecam` — see [LICENSE](LICENSE).
Everything under `cv/` is carried over from that project unchanged in
substance, so it cannot be under anything else.

Detector weights are **not** redistributed here; see [doc/README.md](doc/README.md).
