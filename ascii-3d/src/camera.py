"""The pinhole camera: turning 3D points into pixels and pixels back into 3D.

Coordinate conventions (the same as OpenCV):

    camera at the origin, looking down +z
    +x points right, +y points DOWN (so it matches image rows)
    pixel (u, v): u = column, v = row; pixel centres sit on whole numbers

Every renderer uses this one class, so they all agree exactly on where a 3D
point lands on the screen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Points closer than this count as "behind the camera". Dividing by a z that
# is almost zero would throw them off to infinity.
MIN_DEPTH = 1e-6


@dataclass(frozen=True)
class Camera:
    """Camera intrinsics plus the image size they belong to.

    fx, fy: focal length in pixels (bigger = more zoomed in).
    cx, cy: principal point, the pixel the optical axis goes through.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_image_size(cls, width: int, height: int, focal_scale: float = 1.2) -> Camera:
        """Guess intrinsics for a photo with no calibration data.

        Focal length = 1.2 x the longer side, which is about a 45 degree field
        of view, close to a typical phone camera. A wrong guess only makes the
        result look a little too deep or too flat.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Image size must be positive, got {width}x{height}.")

        focal = focal_scale * max(width, height)

        # (size - 1) / 2 is the exact middle when pixel centres are whole
        # numbers: a 4-pixel row has centres 0, 1, 2, 3, so the middle is 1.5.
        return cls(width, height, focal, focal, (width - 1) / 2, (height - 1) / 2)

    def resized(self, width: int) -> Camera:
        """The same view at a different resolution.

        Scaling the focal length and principal point together with the image
        keeps the field of view, so a small render shows exactly the same
        thing as a big one. (+0.5 / -0.5 because pixel centres are integers.)
        """
        scale = width / self.width
        height = max(1, round(self.height * scale))

        return Camera(
            width=width,
            height=height,
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=(self.cx + 0.5) * scale - 0.5,
            cy=(self.cy + 0.5) * scale - 0.5,
        )

    def cropped(self, left: int, top: int, right: int, bottom: int) -> Camera:
        """A camera that only sees the pixel box [left, right) x [top, bottom).

        Cropping doesn't change the lens, so fx and fy stay the same. Only the
        principal point moves, because it's measured from the new top-left
        corner. This is how renders zoom in on the subject without distortion.
        """
        left, top = max(0, left), max(0, top)
        right, bottom = min(self.width, right), min(self.height, bottom)

        return Camera(right - left, bottom - top, self.fx, self.fy, self.cx - left, self.cy - top)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perspective projection: 3D points (N, 3) -> pixel coordinates.

            u = fx * x / z + cx
            v = fy * y / z + cy

        Dividing by z is what makes distant things look small: the further
        away a point is, the closer to the image centre it lands.

        Returns (u, v, z, valid). `valid` marks points in front of the camera;
        the u, v of invalid points are set to 0 and must be ignored.
        """
        points = np.asarray(points, dtype=np.float64)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        valid = np.isfinite(points).all(axis=1) & (z > MIN_DEPTH)

        u = np.zeros(len(points))
        v = np.zeros(len(points))
        u[valid] = self.fx * x[valid] / z[valid] + self.cx
        v[valid] = self.fy * y[valid] / z[valid] + self.cy

        return u, v, z, valid

    def backproject(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        """The inverse of project(): pixel (u, v) at depth z -> 3D point.

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

        A pixel on its own only defines a ray out of the camera; the depth z
        picks the point on that ray. That's why a depth model is needed.
        Works on arrays of any matching shape and returns shape (..., 3).
        """
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.stack((x, y, z), axis=-1)
