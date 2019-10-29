# -*- coding: utf-8 -*-

"""
@author: samuel
"""

class Shape:
    def __init__(self, center):
        self.center = center

    def __repr__(self):
        return (f"Shape, center={self.center}")


class Plane(Shape):
    def __init__(self, center, normal):
        self.center = center
        self.normal = unit_vector(normal)

    def __repr__(self):
        return (f"Circle, center={self.center}, R={self.R}")


class Circle(Shape):
    def __init__(self, center, R=None, D=None):
        self.center = center
        (R, D) = check_R_and_D(R, D)
        self.R = R
        self.D = D

    def __repr__(self):
        return (f"Circle, center={self.center}, R={self.R}")


class Disc(Circle):
    def __init__(self, center, normal, R=None, D=None):
        self.center = center
        self.normal = unit_vector(normal)
        (R, D) = check_R_and_D(R, D)
        self.R = R
        self.D = D
        self.circle = Circle(center, R=R)
        self.plane = Plane(center, normal)

    def test_intersect(self, ray):
        return intersectDisc(ray, self)

    def __repr__(self):
        return (f"Disc, center={self.center}, R={self.R}, normal={self.normal}")


class Sphere(Shape):
    def __init__(self, center, R=None, D=None):
        self.center = center
        (R, D) = check_R_and_D(R, D)
        self.R = R
        self.D = D

    def test_intersect(self, ray):
        return intersectSphere(ray, self)

    def normal(self, p):
        """The surface normal at the given point on the sphere, pointing toward the center"""
        return unit_vector(self.center - p)

    def __repr__(self):
        return (f"Sphere, center={self.center}, R={self.R}")


class Rectangle(Shape):
    def __init__(self, center, normal, tangent, h=None, w=None):
        self.center = center
        self.normal = unit_vector(normal)
        self.tangent = unit_vector(tangent)
        if not np.isclose(np.dot(self.normal, self.tangent), 0.0):
            raise ValueError("normal and tangent need to be perpendicular")
        self.tangent2 = np.cross(self.normal, self.tangent)
        (h, w) = check_h_and_w(h, w)
        self.h = h
        self.w = w
        self.bounds = np.zeros((4, 3))
        self.bounds[0, :] = self.center + h / 2 * self.tangent - w / 2 * self.tangent2
        self.bounds[1, :] = self.center + h / 2 * self.tangent + w / 2 * self.tangent2
        self.bounds[2, :] = self.center - h / 2 * self.tangent + w / 2 * self.tangent2
        self.bounds[3, :] = self.center - h / 2 * self.tangent - w / 2 * self.tangent2
        self.plane = Plane(center, normal)