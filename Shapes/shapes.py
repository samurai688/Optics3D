# -*- coding: utf-8 -*-

"""
@author: samuel
"""

import numpy as np
from general_optics import unit_vector, distance_between


def check_R_and_D(R, D):
    if R is None:
        if D is None:
            raise ValueError("Need to specify either R or D")
        D = D
        R = D / 2
    elif D is None:
        D = R * 2
        R = R
    else:
        if np.isclose(D, R * 2, rtol=1e-8, atol=1e-8):
            D = D
            R = R
        else:
            raise ValueError("Conflicting R and D specified")
    return R, D


def check_h_and_w(h, w):
    if h is None:
        raise ValueError("Need to specify h")
    if w is None:
        raise ValueError("Need to specify w")
    if h < 0:
        raise ValueError("h must be positive")
    if w < 0:
        raise ValueError("h must be positive")
    return h, w


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
        return (f"Plane, center={self.center}, normal={self.normal}")

    def test_intersect(self, ray):
        denominator = np.dot(unit_vector(self.normal), unit_vector(ray.direction))
        #  ARB_EPSILON_VALUE is an arbitrary epsilon value. We just want
        #  to avoid working with intersections that are almost orthogonal.
        ARB_EPSILON_VALUE = 1e-8
        if np.abs(denominator) > ARB_EPSILON_VALUE:
            difference = self.center - ray.position
            t = np.dot(difference, self.normal) / denominator
            if t > ARB_EPSILON_VALUE:
                intersection_pt = ray.position + t * ray.direction
                return True, intersection_pt, self.normal
        return False, None, None


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
        intersected, intersection_pt, normal = self.plane.test_intersect(ray)
        if intersected:
            if distance_between(intersection_pt, self.center) < self.R:
                return True, intersection_pt, normal
        return False, None, None

    def __repr__(self):
        return (f"Disc, center={self.center}, R={self.R}, normal={self.normal}")


class Sphere(Shape):
    def __init__(self, center, R=None, D=None):
        self.center = center
        (R, D) = check_R_and_D(R, D)
        self.R = R
        self.D = D

    def normal(self, p):
        """The surface normal at the given point on the sphere, pointing toward the center"""
        return unit_vector(self.center - p)

    def __repr__(self):
        return (f"<Sphere, center={self.center}, R={self.R}>")

    def test_intersect(self, ray):
        # adapted from https://github.com/phire/Python-Ray-tracer/blob/master/sphere.py
        # Dec 9 2018
        q = self.center - ray.position
        vDotQ = np.dot(ray.direction, q)
        squareDiffs = np.dot(q, q) - self.R * self.R
        discrim = vDotQ * vDotQ - squareDiffs
        if discrim > 0:  # line intersects in two points
            root = np.sqrt(discrim)
            t0 = (vDotQ - root)
            t1 = (vDotQ + root)
            pt0 = ray.position + t0 * ray.direction
            pt1 = ray.position + t1 * ray.direction
            norm0 = self.normal(pt0)
            norm1 = self.normal(pt1)
            # If both t are positive, ray is facing the sphere and intersecting
            # If one t is positive one t is negative, ray is shooting from inside
            # If both t are negative, ray is shooting away from the sphere, and intersection is impossible.
            # So we have to return the smaller and positive t as the intersecting distance for the ray
            if t0 > 0 and t1 > 0:
                if t0 < t1:
                    return True, pt1, pt0, norm1, norm0
                else:
                    return True, pt0, pt1, norm0, norm1
            elif t0 < 0 and t1 > 0:
                return True, pt1, None, norm1, None
            elif t0 > 0 and t1 < 0:
                return True, pt0, None, norm0, None
            else:
                return False, None, None, None, None
        elif discrim == 0:  # line intersects in one point, tangent
            t0 = vDotQ
            pt0 = ray.position + t0 * ray.direction
            norm0 = self.normal(pt0)
            return True, pt0, None, norm0, None

        else:  # discrim < 0   # line does not intersect
            return False, None, None, None, None


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

    def test_intersect(self, ray):
        # intersected, int_pt = intersectRectangle(ray, self)
        # https://stackoverflow.com/questions/2752725/finding-whether-a-point-lies-inside-a-rectangle-or-not
        # Dec 9 2018
        # Eric Baineville's answer
        intersect_the_plane, intersection_pt, normal = self.plane.test_intersect(ray)
        if intersect_the_plane:
            M = intersection_pt
            A = self.bounds[0, :]
            B = self.bounds[1, :]
            C = self.bounds[2, :]
            AB = B - A
            BC = C - B
            AM = M - A
            BM = M - B
            if 0 <= np.dot(AB, AM) <= np.dot(AB, AB):
                if 0 <= np.dot(BC, BM) <= np.dot(BC, BC):
                    return True, intersection_pt, normal
        return False, None, None
        return intersected, int_pt, normal

    def __repr__(self):
        return (
            f"Rectangle, center={self.center}, normal={self.normal}, tangent={self.tangent}, h={self.h}, w={self.w}")

