# -*- coding: utf-8 -*-

"""
@author: samuel
"""

import numpy as np
from general_optics import unit_vector, distance_between, project_onto_plane, \
    rotation_matrix4, scaling_matrix4, translation_matrix4


INTERSECT_CLIPPING_FLOOR = 1e-12


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
        # print("*** test intersect ***")
        # print(f"self.normal = {self.normal}")
        # print(f"self.center = {self.center}")
        # print(f"ray_position = {ray.position}")
        # print(f"ray.direction = {ray.direction}")
        # print(f"ray z angle = {180 / np.pi * np.arctan(ray.direction[2] / ray.direction[1])} deg")
        denominator = np.dot(unit_vector(self.normal), unit_vector(ray.direction))
        #  ARB_EPSILON_VALUE is an arbitrary epsilon value. We just want
        #  to avoid working with intersections that are almost orthogonal.
        ARB_EPSILON_VALUE = 1e-9
        if np.abs(denominator) > ARB_EPSILON_VALUE:
            difference = self.center - ray.position
            # print(f"difference = {difference}")
            # print(f"denominator = {denominator}")
            t = np.dot(difference, self.normal) / denominator
            # print(f"t = {t}")
            if t > ARB_EPSILON_VALUE:
                intersection_pt = ray.position + t * ray.direction
                # print(f"intersection_pt = {intersection_pt}")
                # print("*** end test intersect -- hit***")
                return True, intersection_pt, self.normal
        # print("*** end test intersect -- no hit ***")
        return False, None, None


class Circle(Shape):
    def __init__(self, center, normal=None, R=None, D=None):
        self.center = center
        (R, D) = check_R_and_D(R, D)
        self.R = R
        self.D = D
        if normal is not None:
            self.normal = unit_vector(normal)
            self.plane = Plane(center, normal)

    def test_intersect(self, ray):
        intersected, intersection_pt, normal = self.plane.test_intersect(ray)
        if intersected:
            if distance_between(intersection_pt, self.center) < self.R:
                return True, intersection_pt, normal
        return False, None, None

    def __repr__(self):
        return (f"Circle, center={self.center}, R={self.R}, normal={self.normal}")



class InfiniteCylinder(Shape):
    def __init__(self, center, direction, R=None, D=None):
        self.center = center
        (R, D) = check_R_and_D(R, D)
        self.R = R
        self.D = D
        self.direction = unit_vector(direction)
        self.plane = Plane(center, normal=direction)

    def test_intersect(self, ray):
        pass

        # transform ray to intersect a unit cylinder
        # step 1 -- translate to origin
        translation_vector = -self.center
        trans_mat = translation_matrix4(translation_vector)
        ray_pos4 = np.array([0.0, 0.0, 0.0, 1.0])
        ray_pos4[0:3] = ray.position
        ray_dir4 = np.array([0.0, 0.0, 0.0, 0.0])
        ray_dir4[0:3] = ray.direction
        ray_pos_translated = np.dot(trans_mat, ray_pos4)
        ray_dir_translated = ray_dir4
        # step 2 -- rotate to align with +z
        z_axis = np.array([0.0, 0.0, 1.0])
        rotation_mat = rotation_matrix4(self.direction, z_axis)
        ray_pos_rotated = np.dot(rotation_mat, ray_pos_translated)
        ray_dir_rotated = np.dot(rotation_mat, ray_dir_translated)
        # step 3 -- scale to unit cylinder
        scale_factor = 1 / self.R
        scaling_vector = np.array([scale_factor, scale_factor, scale_factor])
        scaling_mat = scaling_matrix4(scaling_vector)
        ray_pos_scaled = np.dot(scaling_mat, ray_pos_rotated)
        ray_dir_scaled = ray_dir_rotated
        ray_pos_objectSpace = ray_pos_scaled
        ray_dir_objectSpace = ray_dir_scaled

        # if ray.type == "normal":
        #     print(f"ray_pos4 = {ray_pos4}")
        #     print(f"ray_dir4 = {ray_dir4}")
        #     print(f"ray_pos_objectSpace = {ray_pos_objectSpace}")

        # calculate intersection points on unit cylinder
        # attempting to follow https://www.cl.cam.ac.uk/teaching/1999/AGraphHCI/SMAG/node2.html
        # x^2 + y^2 = 1
        # Ray = E + t * D
        # (xE + txD)^2 + (yE + tyD)^2 = 1
        # t^2(xD^2 + yD^2) + t(2 xE xD + 2 yE yD) +(xE^2 + yE^2 - 1) = 0
        # this is a quadratic equation with
        # A = xD^2 + yD^2
        # B = 2 xE xD + 2 yE yD
        # C = xE^2 + yE^2 - 1
        # solve

        xE = ray_pos_objectSpace[0]
        yE = ray_pos_objectSpace[1]
        # zE = ray_pos_objectSpace[2]
        xD = ray_dir_objectSpace[0]
        yD = ray_dir_objectSpace[1]
        # zD = ray_dir_objectSpace[2]

        A = xD ** 2 + yD ** 2
        B = 2 * xE * xD + 2 * yE * yD
        C = xE ** 2 + yE ** 2 - 1
        discriminant = B * B - 4 * A * C
        if discriminant > 0: # line (not necessarily the ray) intersects in two points
            root = np.sqrt(discriminant)
            # some numerical bit to avoid loss of precision in the quadratic,
            # avoid subtracting two things of potentially similar magnitude
            # https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
            if B >= 0:
                t0_objectSpace = (-B - root) / (2 * A)
                t1_objectSpace = (2 * C) / (-B - root)
            else: # (B < 0)
                t0_objectSpace = (2 * C) / (-B + root)
                t1_objectSpace = (-B + root) / (2 * A)
        elif discriminant == 0:  # line intersects in one point, tangent
            t0_objectSpace = (-B) / (2 * A)
            t1_objectSpace = None
        else: # (discriminant < 0) line intersects in no points
            t0_objectSpace = None
            t1_objectSpace = None


        # transform points back
        if (t0_objectSpace is not None) and (t1_objectSpace is not None):
            # translation doesn't affect distances?
            # rotation shouldn't affect distances?
            # scaling should affect
            t0 = t0_objectSpace / scale_factor
            t1 = t1_objectSpace / scale_factor
            pt0 = ray.position + t0 * ray.direction
            pt1 = ray.position + t1 * ray.direction
            normal0 = self.normal(pt0)
            normal1 = self.normal(pt1)
        elif t0_objectSpace is not None:  # line intersects in one point, tangent
            t0 = t0_objectSpace / scale_factor
            pt0 = ray.position + t0 * ray.direction
            normal0 = self.normal(pt0)
        else:  # (discriminant < 0) line intersects in no points
            pass


        # return
        if (t0_objectSpace is not None) and (t1_objectSpace is not None):
            # If both t are positive, ray is facing the sphere and intersecting
            # If one t is positive one t is negative, ray is shooting from inside
            # If both t are negative, ray is shooting away from the sphere, and intersection is impossible.
            # So we have to return the smaller and positive t as the intersecting distance for the ray
            if t0 > INTERSECT_CLIPPING_FLOOR and t1 > INTERSECT_CLIPPING_FLOOR:
                intersection_count = 2
                if t0 < t1:
                    return True, pt0, pt1, normal0, normal1, intersection_count
                else:
                    return True, pt1, pt0, normal1, normal0, intersection_count
            elif t1 > INTERSECT_CLIPPING_FLOOR:
                intersection_count = 1
                return True, pt1, None, normal1, None, intersection_count
            elif t0 > INTERSECT_CLIPPING_FLOOR:
                intersection_count = 1
                return True, pt0, None, normal0, None, intersection_count
            else:
                intersection_count = 0
                return False, None, None, None, None, intersection_count
        elif t0_objectSpace is not None:  # line intersects in one point, tangent
            # TODO handle negative t0
            intersection_count = 2 # ugh, I'm counting this case as two cause you're probably outside, give me a break
            return True, pt0, None, normal0, None, intersection_count
        else:  # (discriminant < 0) line intersects in no points
            intersection_count = 0
            return False, None, None, None, None, intersection_count


    def test_point(self, point):
        # transform point to a unit cylinder
        # step 1 -- translate cylinder center to origin
        translation_vector = -self.center
        trans_mat = translation_matrix4(translation_vector)
        point4 = np.array([0.0, 0.0, 0.0, 1.0])
        point4[0:3] = point
        point_translated = np.dot(trans_mat, point4)
        # step 2 -- rotate to align with +z
        z_axis = np.array([0.0, 0.0, 1.0])
        rotation_mat = rotation_matrix4(self.direction, z_axis)
        point_rotated = np.dot(rotation_mat, point_translated)
        # step 3 -- scale to unit cylinder
        scale_factor = 1 / self.R
        scaling_vector = np.array([scale_factor, scale_factor, scale_factor])
        scaling_mat = scaling_matrix4(scaling_vector)
        point_scaled = np.dot(scaling_mat, point_rotated)
        point_objectSpace = point_scaled


        # find the answer
        r_point = np.sqrt( point_objectSpace[0] * point_objectSpace[0] +
                           point_objectSpace[1] * point_objectSpace[1] )
        if r_point < 1:
            return True
        else:
            return False


    def normal(self, point):
        """The surface normal at the given point on the infinite cylinder, pointing toward the interior"""
        r = self.center - point;
        n = project_onto_plane(r, self.direction)
        return unit_vector(n)



class Sphere(Shape):
    def __init__(self, center, R=None, D=None):
        self.center = center
        (R, D) = check_R_and_D(R, D)
        self.R = R
        self.D = D

    def normal(self, point):
        """The surface normal at the given point on the sphere, pointing toward the center"""
        return unit_vector(self.center - point)

    def __repr__(self):
        return (f"<Sphere, center={self.center}, R={self.R}>")


    # test if point is in sphere
    def test_point(self, point):
        distance = distance_between(point, self.center)
        # distance = np.sqrt((point[0] - self.center[0]) ** 2 +
        #                    (point[1] - self.center[1]) ** 2 +
        #                    (point[2] - self.center[2]) ** 2 )
        if distance < self.R:
            return True
        else:
            return False

    # test if ray intersects with sphere and get the points
    def test_intersect(self, ray):
        # adapted from https://github.com/phire/Python-Ray-tracer/blob/master/sphere.py
        # Dec 9 2018
        q = self.center - ray.position
        vDotQ = np.dot(ray.direction, q)
        squareDiffs = np.dot(q, q) - self.R * self.R
        discrim = vDotQ * vDotQ - squareDiffs
        intersection_count = 0
        if discrim > 0:  # line (not necessarily the ray) intersects in two points
            root = np.sqrt(discrim)
            t0 = (vDotQ - root)
            t1 = (vDotQ + root)
            pt0 = ray.position + t0 * ray.direction
            pt1 = ray.position + t1 * ray.direction
            norm0 = self.normal(pt0)
            norm1 = self.normal(pt1)

            if ray.type == "faerie_fire":
                pass
            else:
                pass
                # print(f"        t0 = {t0}")
                # print(f"        t1 = {t1}")
            # If both t are positive, ray is facing the sphere and intersecting
            # If one t is positive one t is negative, ray is shooting from inside
            # If both t are negative, ray is shooting away from the sphere, and intersection is impossible.
            # So we have to return the smaller and positive t as the intersecting distance for the ray
            if t0 > INTERSECT_CLIPPING_FLOOR and t1 > INTERSECT_CLIPPING_FLOOR:
                intersection_count = 2
                if t0 < t1:
                    return True, pt1, pt0, norm1, norm0, intersection_count
                else:
                    return True, pt0, pt1, norm0, norm1, intersection_count
            elif t1 > INTERSECT_CLIPPING_FLOOR:
                intersection_count = 1
                return True, pt1, None, norm1, None, intersection_count
            elif t0 > INTERSECT_CLIPPING_FLOOR:
                intersection_count = 1
                return True, pt0, None, norm0, None, intersection_count
            else:
                intersection_count = 0
                return False, None, None, None, None, intersection_count
        elif discrim == 0:  # line intersects in one point, tangent
            # TODO handle negative t0
            t0 = vDotQ
            pt0 = ray.position + t0 * ray.direction
            norm0 = self.normal(pt0)
            intersection_count = 2 # ugh, I'm counting this case as two cause you're probably outside, give me a break
            return True, pt0, None, norm0, None, intersection_count
        else:  # discrim < 0   # line does not intersect
            intersection_count = 0
            return False, None, None, None, None, intersection_count


class Cylinder(Shape):

    def __init__(self, center, R=None, D=None, h=None):
        self.center = center
        (R, D) = check_R_and_D(R, D)
        self.R = R
        self.D = D
        self.h = h

    def normal(self, point):
        """The surface normal at the given point on the cylinder, pointing toward the interior"""
        pass

    def test_intersect(self, ray):
        pass

    def test_point(self, point):
        pass


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

    def __repr__(self):
        return (
            f"Rectangle, center={self.center}, normal={self.normal}, tangent={self.tangent}, h={self.h}, w={self.w}")

