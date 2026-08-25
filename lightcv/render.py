"""Top-down map renderer.

Draws the world, not the camera. Sectors are filled polygons shaded by floor
height; things are tokens standing on them.

CHEAP BY CONSTRUCTION, and not by optimisation. The HLSLS renderers cost
what they cost because they touch every pixel of a 1280x720 photograph every
frame -- the regions renderer measured 394-659ms on tina. This one fills a
few dozen polygons at whatever size the viewer asked for, and only when
`World.revision` changes. A still room costs nothing at all: the last render
is still correct, because the map is state rather than a view of a decaying
signal.

`RenderCache.frame_for()` is the entry point that enforces that: it returns
the cached image unless the revision moved.
"""

import numpy as np
import cv2

# Height shading. Floor is darkest, raised sectors progressively lighter, so
# "higher" reads immediately without a legend. Deliberately a small palette
# of neutrals: the fills encode HEIGHT, not the object's real colour, and
# using plausible object colours here would imply a measurement that was
# never made.
_FLOOR_RGB = (34, 38, 46)
_HIGH_RGB = (150, 158, 172)
_EDGE_RGB = (90, 98, 112)
_LABEL_RGB = (190, 198, 210)

_THING_RGB = {
    'cat': (240, 176, 74),
    'human': (236, 92, 92),
    'unknown': (150, 150, 160),
}


class MapRenderer:
    def __init__(self, width=900, height=640, margin_px=40, bg=(18, 20, 25)):
        self.width = int(width)
        self.height = int(height)
        self.margin = int(margin_px)
        self.bg = bg

    # ---- coordinate transform -----------------------------------------

    def _fit(self, world):
        """Map cm -> pixels, preserving aspect ratio and y-up.

        y is flipped: map +y runs away from the camera, and a plan view
        reads correctly with 'away' at the top.
        """
        x0, y0, x1, y1 = world.bounds()
        w = max(1e-6, x1 - x0)
        h = max(1e-6, y1 - y0)
        sx = (self.width - 2 * self.margin) / w
        sy = (self.height - 2 * self.margin) / h
        s = min(sx, sy)
        ox = (self.width - w * s) / 2.0 - x0 * s
        oy = (self.height - h * s) / 2.0 - y0 * s

        def to_px(p):
            X = p[0] * s + ox
            Y = self.height - (p[1] * s + oy)
            return (int(round(X)), int(round(Y)))

        return to_px, s

    def _height_colour(self, h, hmax):
        f = 0.0 if hmax <= 0 else max(0.0, min(1.0, h / hmax))
        # gamma < 1 so the first step off the floor is clearly visible
        f = f ** 0.65
        return tuple(int(round(a + (b - a) * f))
                     for a, b in zip(_FLOOR_RGB, _HIGH_RGB))

    # ---- drawing -------------------------------------------------------

    def render(self, world, show_labels=True, show_heights=True):
        img = np.full((self.height, self.width, 3), self.bg, np.uint8)
        if not world.sectors:
            cv2.putText(img, 'NO MAP', (self.margin, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, _LABEL_RGB, 2,
                        cv2.LINE_AA)
            return img

        to_px, scale = self._fit(world)
        hmax = max((s.floor_height for s in world.sectors.values()), default=0.0)

        # Painter's order: low floors first, so a raised sector sits visually
        # on top of the floor it stands on.
        for s in sorted(world.sectors.values(), key=lambda s: s.floor_height):
            if len(s.polygon) < 3:
                continue
            pts = np.array([to_px(p) for p in s.polygon], np.int32)
            cv2.fillPoly(img, [pts], self._height_colour(s.floor_height, hmax))
            # A proposed sector is drawn dashed-thin, a confirmed one solid.
            # Same honesty rule as the Phase 5 HUD: the viewer must be able
            # to tell asserted-and-unchecked from checked.
            solid = (s.source == 'confirmed')
            cv2.polylines(img, [pts], True, _EDGE_RGB, 2 if solid else 1,
                          cv2.LINE_AA)
            if show_labels:
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                txt = s.label.upper()
                if show_heights and s.floor_height > 0.01:
                    txt += f'  {s.floor_height:.0f}'
                mark = '' if solid else ' ?'
                cv2.putText(img, txt + mark, (cx - 30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _LABEL_RGB, 1,
                            cv2.LINE_AA)

        for t in world.things.values():
            px = to_px((t.x, t.y))
            col = _THING_RGB.get(t.kind, _THING_RGB['unknown'])
            r = max(6, int(round(14 * scale / max(scale, 0.06))))
            r = max(7, min(16, r))
            cv2.circle(img, px, r, col, -1, cv2.LINE_AA)
            cv2.circle(img, px, r, (12, 14, 18), 2, cv2.LINE_AA)
            if t.facing is not None:
                a = np.deg2rad(t.facing)
                tip = (int(px[0] + np.sin(a) * r * 2.0),
                       int(px[1] - np.cos(a) * r * 2.0))
                cv2.line(img, px, tip, col, 2, cv2.LINE_AA)
            cv2.putText(img, t.display_name, (px[0] + r + 4, px[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            if t.sector_id and t.sector_id in world.sectors:
                s = world.sectors[t.sector_id]
                if s.floor_height > 0.01:
                    cv2.putText(img, f'on {s.label}',
                                (px[0] + r + 4, px[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _LABEL_RGB, 1,
                                cv2.LINE_AA)

        cv2.putText(img, f'rev {world.revision}', (10, self.height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _EDGE_RGB, 1, cv2.LINE_AA)
        return img


class RenderCache:
    """Render on change, never per frame.

    This is where §4's inversion becomes a cost saving as well as a
    correctness property: if nothing wrote to the map, the previous image is
    still exactly right, so there is nothing to recompute.
    """

    def __init__(self, renderer=None):
        self.renderer = renderer or MapRenderer()
        self._rev = None
        self._img = None
        self.renders = 0
        self.hits = 0

    def frame_for(self, world, force=False):
        if not force and self._img is not None and self._rev == world.revision:
            self.hits += 1
            return self._img
        self._img = self.renderer.render(world)
        self._rev = world.revision
        self.renders += 1
        return self._img
