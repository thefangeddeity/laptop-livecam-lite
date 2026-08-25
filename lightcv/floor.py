"""Floor-plane homography: image pixels -> map centimetres.

WHY THIS WORKS WHERE PHASE 3B FAILED. Phase 3b tried to register a
photograph of the room against a live frame and died on parallax, which was
the correct outcome: a room is not a plane, so no single homography can map
it. A FLOOR is a plane. That is the one case a homography models exactly,
with no approximation and no parallax error, and it is all this needs --
because a thing standing on the floor touches the floor, and the bottom edge
of its detection box is that contact point.

So the mapping is not "where is this object in 3D" but "where does it touch
the ground", which is a 2D-to-2D problem on a genuinely planar surface.

Two consequences worth being explicit about:

  * A thing NOT touching the floor maps wrong. A cat on the couch has its
    contact point on the COUCH, ~40cm above the floor plane, so projecting
    its bottom edge through the floor homography places it too far from the
    camera. `contact_to_map` takes the sector's floor height and corrects
    for exactly this -- see `_raise_to_height`.
  * Anything above the horizon line maps to infinity or behind the camera.
    `contact_to_map` returns None there rather than a number, because a
    silently absurd coordinate is worse than a missing one.
"""

import numpy as np
import cv2


class FloorPlane:
    """Wraps the image->map homography and the projection helpers."""

    def __init__(self, H=None, quality=None, camera_height_cm=None,
                 w_sign=None, camera_map_xy=None):
        self.H = None if H is None else np.asarray(H, dtype=np.float64)
        self.quality = quality or {}
        # Needed only to correct contact points that are not on the floor.
        # Asserted by the operator, like the floor heights themselves.
        self.camera_height_cm = camera_height_cm
        # Sign of the homogeneous w for points on the visible side of the
        # horizon. NOT assumed: a homography is only defined up to scale, so
        # +w and -w are both valid solutions for the same mapping and the
        # sign depends on what cv2 happened to return. It is measured from
        # the calibration quad instead -- those points are on the floor by
        # definition, so whatever sign they produce is the correct one.
        self.w_sign = w_sign
        # The camera's own ground position, in MAP coordinates. Required for
        # the raised-contact correction and for nothing else.
        #
        # It cannot be recovered from the homography alone: the homography
        # maps the ground plane, and the camera's nadir point is generally
        # outside the image entirely. Deriving it would need intrinsics and
        # tilt, which this system does not have and will not guess. So it is
        # ASSERTED, like the floor dimensions and the floor heights -- and
        # when it is absent the correction is SKIPPED rather than applied
        # against the wrong origin, which is what a map-origin-relative
        # scale silently does.
        self.camera_map_xy = camera_map_xy

    # ---- construction --------------------------------------------------

    @classmethod
    def from_correspondences(cls, image_pts, map_pts, camera_height_cm=None,
                             camera_map_xy=None):
        """Fit from >= 4 point pairs.

        With exactly 4 points this is an exact solve (getPerspectiveTransform)
        and residuals are meaningless -- reported as such rather than as a
        flattering zero. With more, RANSAC is used and the residuals are a
        real quality signal.
        """
        ip = np.asarray(image_pts, dtype=np.float32).reshape(-1, 2)
        mp = np.asarray(map_pts, dtype=np.float32).reshape(-1, 2)
        if len(ip) != len(mp) or len(ip) < 4:
            raise ValueError('need >= 4 matched point pairs')

        if len(ip) == 4:
            H = cv2.getPerspectiveTransform(ip, mp)
            inliers = 4
            exact = True
        else:
            H, mask = cv2.findHomography(ip, mp, cv2.RANSAC, 5.0)
            if H is None:
                raise ValueError('homography fit failed')
            inliers = int(mask.sum()) if mask is not None else len(ip)
            exact = False

        obj = cls(H, camera_height_cm=camera_height_cm,
                  camera_map_xy=camera_map_xy)
        obj.w_sign = obj._measure_w_sign(ip)
        obj.quality = obj._assess(ip, mp, inliers, exact)
        return obj

    def _measure_w_sign(self, image_pts):
        """Sign of w for the calibration points, which are floor by
        definition. Uses the mean rather than any single corner so one
        near-degenerate point cannot flip it."""
        p = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
        w = self.H[2, 0] * p[:, 0] + self.H[2, 1] * p[:, 1] + self.H[2, 2]
        return 1.0 if float(np.mean(w)) >= 0 else -1.0

    def _assess(self, ip, mp, inliers, exact):
        """Reprojection residuals in map units, plus the degeneracy checks
        that actually matter for a floor quad."""
        proj = self.image_to_map(ip)
        err = np.linalg.norm(proj - mp, axis=1)
        q = {
            'n_points': int(len(ip)),
            'n_inliers': int(inliers),
            'inlier_ratio': round(float(inliers) / len(ip), 3),
            'exact_solve': bool(exact),
            'residual_mean_cm': round(float(err.mean()), 2),
            'residual_max_cm': round(float(err.max()), 2),
            'convex': bool(self._quad_convex(ip)) if len(ip) == 4 else None,
        }
        if exact:
            q['note'] = ('exact 4-point solve: residuals are ~0 by '
                         'construction and are NOT evidence of accuracy')
        return q

    @staticmethod
    def _quad_convex(pts):
        """A floor quad that is not convex means the corners were clicked
        out of order, which produces a plausible-looking but wrong map."""
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if len(p) != 4:
            return True
        signs = []
        for i in range(4):
            a = p[(i + 1) % 4] - p[i]
            b = p[(i + 2) % 4] - p[(i + 1) % 4]
            signs.append(np.sign(a[0] * b[1] - a[1] * b[0]))
        return len(set(s for s in signs if s != 0)) == 1

    # ---- projection ----------------------------------------------------

    def image_to_map(self, pts):
        """Project image points onto the floor plane. Returns (N,2)."""
        if self.H is None:
            raise ValueError('no homography')
        p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(p, self.H)
        return out.reshape(-1, 2)

    def map_to_image(self, pts):
        if self.H is None:
            raise ValueError('no homography')
        Hi = np.linalg.inv(self.H)
        p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, Hi).reshape(-1, 2)

    def is_below_horizon(self, x, y):
        """False if the point projects to infinity or behind the camera --
        i.e. it is at or above the vanishing line and has no floor position."""
        if self.H is None:
            return False
        H = self.H
        w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        s = self.w_sign if self.w_sign is not None else 1.0
        return bool(w * s > 1e-9)

    def contact_to_map(self, box, floor_height_cm=0.0):
        """Map a detection box to a map position via its ground contact.

        `box` is (x, y, w, h) in image pixels; the contact point is the
        bottom-edge centre. Returns (map_x, map_y) or None.

        When the thing is standing on something raised, its contact point
        is not on the floor plane and projecting it directly puts it too
        far away. `floor_height_cm` corrects that -- see `_raise_to_height`.
        """
        bx, by, bw, bh = box
        cx = bx + bw / 2.0
        cy = by + bh
        if not self.is_below_horizon(cx, cy):
            return None
        mx, my = self.image_to_map([(cx, cy)])[0]
        if (floor_height_cm and self.camera_height_cm
                and self.camera_map_xy is not None):
            mx, my = self._raise_to_height(mx, my, floor_height_cm)
        return (float(mx), float(my))

    def can_correct_height(self):
        """Whether raised-contact correction is available. False means a
        cat on the couch is placed as if its feet were on the floor, i.e.
        too far from the camera -- a known, bounded error, not a silent one."""
        return bool(self.camera_height_cm and self.camera_map_xy is not None)

    def _raise_to_height(self, mx, my, h):
        """Correct a floor-plane projection for a contact point at height h.

        The camera sits at height C above its ground position K. A contact
        point that really sits at height h, at ground distance d from K,
        projects onto the FLOOR plane at distance d' = d * C / (C - h) --
        further away than it really is. Inverting gives d = d' * (C - h) / C,
        so the correction is a scale toward K, and it must be measured FROM
        K rather than from the map origin. Scaling toward the origin instead
        is wrong by however far the origin is from the camera, which for a
        floor quad staked out in front of the camera is most of the room.

        Exact for a pinhole camera; needs only C and K, no calibration.
        Degrades safely: if h >= C the geometry is impossible (the surface
        is above the lens) and the point is returned unchanged rather than
        flipped through infinity.
        """
        C = float(self.camera_height_cm)
        if not C or h >= C or self.camera_map_xy is None:
            return (mx, my)
        kx, ky = self.camera_map_xy
        k = (C - h) / C
        return (kx + (mx - kx) * k, ky + (my - ky) * k)
