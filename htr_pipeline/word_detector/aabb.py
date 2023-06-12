import numpy as np


class AABB:
    """axis aligned bounding box"""

    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def width(self):
        return self.xmax - self.xmin

    def scale(self, fx, fy):
        new = AABB(self.xmin, self.xmax, self.ymin, self.ymax)
        new.xmin = fx * new.xmin
        new.xmax = fx * new.xmax
        new.ymin = fy * new.ymin
        new.ymax = fy * new.ymax
        return new

    def scale_around_center(self, fx, fy):
        cx = (self.xmin + self.xmax) / 2
        cy = (self.ymin + self.ymax) / 2

        new = AABB(self.xmin, self.xmax, self.ymin, self.ymax)
        new.xmin = cx - fx * (cx - self.xmin)
        new.xmax = cx + fx * (self.xmax - cx)
        new.ymin = cy - fy * (cy - self.ymin)
        new.ymax = cy + fy * (self.ymax - cy)
        return new

    def translate(self, tx, ty):
        new = AABB(self.xmin, self.xmax, self.ymin, self.ymax)
        new.xmin = new.xmin + tx
        new.xmax = new.xmax + tx
        new.ymin = new.ymin + ty
        new.ymax = new.ymax + ty
        return new

    def as_type(self, t):
        new = AABB(self.xmin, self.xmax, self.ymin, self.ymax)
        new.xmin = t(new.xmin)
        new.xmax = t(new.xmax)
        new.ymin = t(new.ymin)
        new.ymax = t(new.ymax)
        return new

    def enlarge_to_int_grid(self):
        new = AABB(self.xmin, self.xmax, self.ymin, self.ymax)
        new.xmin = np.floor(new.xmin)
        new.xmax = np.ceil(new.xmax)
        new.ymin = np.floor(new.ymin)
        new.ymax = np.ceil(new.ymax)
        return new

    def enlarge(self, v):
        new = AABB(self.xmin - v, self.xmax + v, self.ymin - v, self.ymax + v)
        return new

    def clip(self, clip_aabb):
        new = AABB(self.xmin, self.xmax, self.ymin, self.ymax)
        new.xmin = min(max(new.xmin, clip_aabb.xmin), clip_aabb.xmax)
        new.xmax = max(min(new.xmax, clip_aabb.xmax), clip_aabb.xmin)
        new.ymin = min(max(new.ymin, clip_aabb.ymin), clip_aabb.ymax)
        new.ymax = max(min(new.ymax, clip_aabb.ymax), clip_aabb.ymin)
        return new

    def area(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)
