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

## Camera-missing fallback (pilot, running on tina)

On Linux the microphone rides in `/cam` with the picture. tina's USB camera
drops off the bus, and when it did, the room's audio went with it: no frame,
nothing written to the v4l2 loopback, no publisher, so no `/cam`, no HLS
listening, no `roomaudio` and no two-way.

`hlsls/broadcast-api.camera-fallback.patch` keeps `/cam` up instead. While the
camera is missing or stalled (no frame for `CAMERA_FRESH_S`, the same 3 s after
which the panel's camera lamp goes down), the writer loop publishes a black
frame on the live microphone. The drain loop keeps reopening the camera, and
the picture returns by itself. Both transitions are logged
(`camera missing ...`, `camera back`).

It is a patch on hls-livecam-server's `broadcast-api` (as deployed on tina and
tanzania on 2026-10-05, upstream 50b06ad):

    sudo patch /usr/local/bin/broadcast-api < hlsls/broadcast-api.camera-fallback.patch
    sudo systemctl restart broadcast-api

`patch -R` with the same file takes it back out.

## Owning what tina actually runs (`hlsls/`)

Until 2026-10-09 every artifact that ran tina lived **only** on the filesystem.
The viewer at `/var/www/hls-livecam` was hand-installed and tracked nowhere, so
when `vendor/hls.min.js` went missing the viewer silently fell back to native
HLS and there was no source of truth to compare against. `hlsls/` now mirrors
the deployed node:

    hlsls/bin/broadcast-api          -> /usr/local/bin/broadcast-api
    hlsls/web/                       -> /var/www/hls-livecam/
    hlsls/etc/nginx/hls-livecam.conf -> /etc/nginx/conf.d/
    hlsls/etc/systemd/*              -> /etc/systemd/system/
    hlsls/deploy.sh                     the only path from tree to box

### Anonymization: three axes, all required

This repo is **public**. It is anonymized along three separate axes; none
substitutes for another, and all three apply to anything added here.

**1. Personal attribution.** No maintainer name in comments. `hls-livecam`
commit `de7c732` replaced the maintainer name with `dev` throughout; this tree goes further
and removes the *quotation* too, because a quoted line is still one person's
words even under an anonymous label. `(dev: "X")` becomes `(requirement: X)`.
The rationale each comment records is unchanged.

**2. Node identity.** The tailnet name and the machines on it stay out of the
tree. Files ending `.in` carry `@TAILNET_HOST@` and `deploy.sh` fills them from
`hlsls/node.env` (gitignored), aborting if a placeholder survives. Node
nicknames are written as "a sibling node". The viewer already followed this
convention with `https://<node>.ts.net`.

**3. Camera-derived material.** Frames of a family living space never land
here -- see the `.gitignore` entries, which exist because two `git add -A`
sweeps brought such files in before the rules were written.

`web/cams/cams.json` lists real machines by label and tailnet IP, so it is
runtime state on the box and is never shipped from here, not even as a default.
`web/cams/cams.example.json` shows its shape.


`deploy.sh` copies only files this repo owns. It never uses `--delete` and
never touches runtime state (`broadcast.txt`, `buzz.txt`, `cams.json`), because
the running system writes those. `--dry-run` shows the diff; an unchanged
deploy is a true no-op and does not bounce nginx.

`/etc/hls-livecam/device.env` is the **admin tier** -- node-specific hardware
values -- and is deliberately *not* deployed. `etc/hls-livecam/device.env.example`
tracks its shape only.

### Ports

tina redirects `:8080` -> `https://<node>.<tailnet>.ts.net:8443`, so there
is no plaintext fallback: if the Tailscale cert lapses the viewer is simply
down. tanzania still serves plain HTTP on `:8080` with no TLS and no redirect.
The two nodes are deliberately not alike here; decide before copying either.

### `lightcv-audiocheck` -- prove the mic carries *sound*

    hlsls/bin/lightcv-audiocheck [device] [seconds]

tina's panel showed "HLS AUDIO LIVE", a green Mic lamp and a moving IN meter
while carrying **no room audio at all**. The meter was tracking a DC offset
from an empty mic jack, which `speechnorm=e=12.5` then expanded until it
resembled signal. Every indicator we had could prove bytes were moving, not
that they were sound.

This measures the audible band (300 Hz-4 kHz) separately from sub-40 Hz energy
and calls a floating input what it is. On tina it reports:

    sub-40Hz (DC)         -35.7 dB
    audible 300-4k        -82.5 dB
    verdict: DEAD INPUT -- audible band silent, energy is DC offset

**tina has no working microphone.** Its HDA codec exposes exactly one input pin
(`0x1a`, the external pink 1/8" jack, which reads `Mic Jack: off`) and no
internal mic pin at all. The USB camera (`0c45:6366`) is video-only -- the
kernel has only ever bound `uvcvideo` to it, never `snd-usb-audio`, and no
second sound card has ever appeared. Room audio on this node therefore requires
physically plugging a mic into the pink jack, or adding a USB mic. No software
change can recover it, and the planned "invert which leg owns ALSA" fix would
have kept the stream up and still silent.
