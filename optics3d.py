#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 23:10:35 2018

@author: samuel
"""

import numpy as np
from general import to_precision
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import art3d
from matplotlib.patches import Circle as mplCircle
from matplotlib.patches import Rectangle as mplRectangle

# note: we are representing vectors as 1d numpy arrays, i.e., R = np.array([x, y, z])


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2, v_normal=None):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    if v_normal is not None: # signed
        angle = np.arctan2(np.dot((np.cross(v2_u, v1_u)), v_normal), np.dot(v1_u, v2_u))
    else: # unsigned
        angle = np.arccos(np.dot(v1_u, v2_u))
    return angle


def distance_between(p1, p2):
    """ Returns distance between two points """
    return np.linalg.norm(p1 - p2)


def rotation_matrix_axis_angle(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians, using the Euler-Rodrigues formula

    Note that in numpy you have to use the result (call it RM) like new_r = RM.dot(r),
    not new_r = RM * r like you might expect. 
    """
    # https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    # Dec 9 2018
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


######
###### StackOverflow
######
# Dec 8 2018
# overload matplotlib to allow arbitrary normals of patches
# courtesy of https://stackoverflow.com/questions/18228966/how-can-matplotlib-2d-patches-be-transformed-to-3d-with-arbitrary-normals    
# and the lovely person with the reply for negative angles
######
def rotation_matrix(v1, v2):
    """
    Calculates the rotation matrix that changes v1 into v2.
    """
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    cos_angle = np.dot(v1, v2)
    d = np.cross(v1, v2)
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        M = np.identity(3) if cos_angle > 0. else -np.identity(3)
    else:
        d /= sin_angle

        eye = np.eye(3)
        ddt = np.outer(d, d)
        skew = np.array([[    0,  d[2],  -d[1]],
                      [-d[2],     0,  d[0]],
                      [d[1], -d[0],    0]], dtype=np.float64)

        M = ddt + cos_angle * (eye - ddt) + sin_angle * skew

    return M

def pathpatch_2d_to_3d(pathpatch, z = 0, normal = 'z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str:  # Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1, 0, 0), index)

    path = pathpatch.get_path()  # Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path)  # Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D  # Change the class
    pathpatch._code3d = path.codes  # Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor  # Get the face color

    verts = path.vertices  # Get the vertices in 2D

    M = rotation_matrix(normal, (0, 0, 1))  # Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])

def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta
######
###### End stackoverflow
######





def intersectPlane(ray, plane):
    denominator = np.dot(unit_vector(plane.normal), unit_vector(ray.direction))
    #  0.000001 is an arbitrary epsilon value. We just want
    #  to avoid working with intersections that are almost
    #  orthogonal.
    if np.abs(denominator) > 0.000001:
        difference = plane.center - ray.position
        t = np.dot(difference, plane.normal) / denominator
        if t > 0.000001:
            intersection_pt = ray.position + t * ray.direction
            return True, intersection_pt
    return False, None


def intersectDisc(ray, disc):
    intersect, intersection_pt = intersectPlane(ray, disc.plane)
    if intersect:
        if distance_between(intersection_pt, disc.center) < disc.R:
            return True, intersection_pt
    return False, None


def intersectSphere(ray, sphere):
    # adapted from https://github.com/phire/Python-Ray-tracer/blob/master/sphere.py
    # Dec 9 2018
    q = sphere.center - ray.position
    vDotQ = np.dot(ray.direction, q)
    squareDiffs = np.dot(q, q) - sphere.R * sphere.R
    discrim = vDotQ * vDotQ - squareDiffs
    if discrim > 0: # line intersects in two points
        root = np.sqrt(discrim)
        t0 = (vDotQ - root)
        t1 = (vDotQ + root)
        pt0 = ray.position + t0 * ray.direction
        pt1 = ray.position + t1 * ray.direction
        # If both t are positive, ray is facing the sphere and intersecting
        # If one t is positive one t is negative, ray is shooting from inside
        # If both t are negative, ray is shooting away from the sphere, and intersection is impossible.
        # So we have to return the smaller and positive t as the intersecting distance for the ray
        if t0 > 0 and t1 > 0:
            if t0 < t1:
                return True, pt1, pt0
            else:
                return True, pt0, pt1
        elif t0 < 0 and t1 > 0:
            return True, pt1, None
        elif t0 > 0 and t1 < 0:
            return True, pt0, None
        else:
            return False, None, None
    elif discrim == 0:  # line intersects in one point, tangent
        t0 = vDotQ
        pt0 = ray.position + t0 * ray.direction
        return True, pt0, None

    else:  # discrim < 0   # line does not intersect
        return False, None, None


def intersectRectangle(ray, rectangle):
    # https://stackoverflow.com/questions/2752725/finding-whether-a-point-lies-inside-a-rectangle-or-not
    # Dec 9 2018
    # Eric Baineville's answer
    intersect_the_plane, intersection_pt = intersectPlane(ray, rectangle.plane)
    if intersect_the_plane:
        M = intersection_pt
        A = rectangle.bounds[0, :]
        B = rectangle.bounds[1, :]
        C = rectangle.bounds[2, :]
        AB = B - A
        BC = C - B
        AM = M - A
        BM = M - B
        if 0 <= np.dot(AB, AM) <= np.dot(AB, AB):
            if 0 <= np.dot(BC, BM) <= np.dot(BC, BC):
                return True, intersection_pt
    return False, None


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

def project_onto_plane(x, n):
    d = np.dot(x, n) / np.linalg.norm(n)
    p = [d * unit_vector(n)[i] for i in range(len(n))]
    return np.array([x[i] - p[i] for i in range(len(x))])




    def test_intersect(self, ray):
        return intersectRectangle(ray, self)

    def __repr__(self):
        return(f"Rectangle, center={self.center}, normal={self.normal}, tangent={self.tangent}, h={self.h}, w={self.w}")
    


class Optic:
    def __init__(self, position, normal, shape):
        self.position = position
        self.normal = unit_vector(normal)
        self.shape = shape


    def __repr__(self):
        return("Optic ({}, {}, {}),  ({}, {}, {})".format(self.optic_type,
                                                       to_precision(self.position[0], 5),
                                                       to_precision(self.position[1], 5),
                                                       to_precision(self.position[2], 5),
                                                       to_precision(self.normal[0], 5),
                                                       to_precision(self.normal[1], 5),
                                                       to_precision(self.normal[2], 5)))


class Mirror(Optic):
    def __init__(self, position, normal, shape="circular_flat", D=None, w=None, h=None,
                 thickness=10, f=None, tangent=None):
        self.position = position
        self.normal = unit_vector(normal)
        self.shape = shape
        self.D = D
        self.w = w
        self.h = h
        self.thickness = thickness
        self.f = f
        self.surfaces = []
        if shape == "circular_flat":
            self.surfaces.append(Disc(position, normal, D=D))
        elif shape == "rectangular_flat":
            self.surfaces.append(Rectangle(position, normal, tangent, h=h, w=w))
        elif shape == "circular_concave_spherical":
            self.rc = self.f * 2
            self.p_rc = self.position + self.normal * self.rc
            self.surfaces.append(Disc(position, normal, D=D))
            self.surfaces.append(Sphere(center=self.p_rc, R=self.rc))
        else:
            raise ValueError("Unknown shape")


    def __repr__(self):
        return("Mirror ({}, {}, {}),  ({}, {}, {})".format(to_precision(self.position[0], 5),
                                                           to_precision(self.position[1], 5),
                                                           to_precision(self.position[2], 5),
                                                           to_precision(self.normal[0], 5),
                                                           to_precision(self.normal[1], 5),
                                                           to_precision(self.normal[2], 5)))

    def test_intersect(self, ray):  # Mirror
        intersected = False
        intersection_pt = None
        min_distance = np.inf
        if self.shape == "circular_flat" or self.shape == "rectangular_flat":
            for surface in self.surfaces:  # for now this is just the one disc or rectangle
                intersected_here, int_pt = surface.test_intersect(ray)
                if intersected_here:
                    intersected = True
                    distance_to_surface = distance_between(ray.position, int_pt)
                    if distance_to_surface < min_distance:
                        min_distance = distance_to_surface
                        intersection_pt = int_pt
        elif self.shape == "circular_concave_spherical":
            for surface in self.surfaces:
                if isinstance(surface, Sphere):  # first check for intersection with the sphere
                    intersected_sphere, int_pt_sphere1, int_pt_sphere2 = surface.test_intersect(ray)
                    break
            if intersected_sphere:
                for surface in self.surfaces:
                    if isinstance(surface, Disc):  # then check for intersection with the disc
                        intersected_disc, int_pt_disc = surface.test_intersect(ray)
                        break
                if intersected_sphere and intersected_disc:
                    intersected = True  # pick the sphere intersection that is closest to the disc
                                        # TODO: edge cases, the above isn't exhaustive
                    distance1 = distance_between(int_pt_disc, int_pt_sphere1)
                    if int_pt_sphere2 is not None:
                        distance2 = distance_between(int_pt_disc, int_pt_sphere2)
                        if distance1 < distance2:
                            intersection_pt = int_pt_sphere1
                        else:
                            intersection_pt = int_pt_sphere2
                    else:
                        intersection_pt = int_pt_sphere1
        return intersected, intersection_pt


    def draw(self, ax, view="3d"):  # Mirror
        """Um, somehow draw the optic"""
        if view == "xy" or view == "tangential":
            if self.shape == "circular_flat":
                x = self.position[0]
                y = self.position[1]
                ln, = ax.plot(x, y, 'ok')
            else:
                raise ValueError("unknown shape")
            # return the Artist created
            return ln
        elif view == "xz" or view == "sagittal":
            if self.shape == "circular_flat":
                x = self.position[0]
                z = self.position[2]
                ln, = ax.plot(x, z, 'ok')
            else:
                raise ValueError("unknown shape")
            # return the Artist created
            return ln
        # 3D
        # https://stackoverflow.com/questions/18228966/how-can-matplotlib-2d-patches-be-transformed-to-3d-with-arbitrary-normals
        elif view == "3d" or view == "xyz":
            if self.shape == "circular_flat":
                x = self.position[0]
                y = self.position[1]
                z = self.position[2]
                p = mplCircle((0, 0), self.D / 2, facecolor="silver", alpha=0.5, edgecolor="black")
                ax.add_patch(p)
                pathpatch_2d_to_3d(p, z=0, normal=self.normal)
                pathpatch_translate(p, (x, y, z))
                # return the Artist created
                return None
            elif self.shape == "rectangular_flat":
                x = self.position[0]
                y = self.position[1]
                z = self.position[2]
                p = mplRectangle((-self.w/2, -self.h/2), width=self.w, height=self.h, facecolor="silver", alpha=0.5,
                                  edgecolor="black")
                ax.add_patch(p)
                pathpatch_2d_to_3d(p, z=0, normal=self.normal)
                pathpatch_translate(p, (x, y, z))
                return None
            elif self.shape == "circular_concave_spherical":  # for now just draw as disc
                x = self.position[0]
                y = self.position[1]
                z = self.position[2]
                p = mplCircle((0, 0), self.D / 2, facecolor="silver", alpha=0.5, edgecolor="black")
                ax.add_patch(p)
                pathpatch_2d_to_3d(p, z=0, normal=self.normal)
                pathpatch_translate(p, (x, y, z))
#                ln, = ax.plot(self.p_rc[0], self.p_rc[1], self.p_rc[2], 'ob')
                # return the Artist created
                return None
            else:
                raise ValueError("unknown shape")


class Lens(Optic):
    def __init__(self, position, normal, shape="spherical_biconvex", D=None, w=None, h=None,
                 f=None, tangent=None, thinlens=False, index=1.5):
        self.position = position
        self.normal = unit_vector(normal)
        self.shape = shape
        self.D = D
        self.w = w
        self.h = h
        self.f = f
        self.surfaces = []
        self.thinlens = thinlens
        self.index = index
        if shape == "spherical_biconvex":
            self.surfaces.append(Disc(position, normal, D=D))
        else:
            raise ValueError("Unknown shape")


    def __repr__(self):
        return("Lens ({}, {}, {}),  ({}, {}, {})".format(to_precision(self.position[0], 5),
                                                           to_precision(self.position[1], 5),
                                                           to_precision(self.position[2], 5),
                                                           to_precision(self.normal[0], 5),
                                                           to_precision(self.normal[1], 5),
                                                           to_precision(self.normal[2], 5)))

    def draw(self, ax, view="3d"):  # Lens
        """Um, somehow draw the optic"""
        # 3D
        # https://stackoverflow.com/questions/18228966/how-can-matplotlib-2d-patches-be-transformed-to-3d-with-arbitrary-normals
        if view == "3d" or view == "xyz":
            if self.shape == "spherical_biconvex":
                x = self.position[0]
                y = self.position[1]
                z = self.position[2]
                p = mplCircle((0, 0), self.D / 2, facecolor="silver", alpha=0.5, edgecolor="black")
                ax.add_patch(p)
                pathpatch_2d_to_3d(p, z=0, normal=self.normal)
                pathpatch_translate(p, (x, y, z))
                # return the Artist created
                return None
            else:
                raise ValueError("unknown shape")

    def test_intersect(self, ray):  # Lens
        intersected = False
        intersection_pt = None
        min_distance = np.inf
        if self.shape == "spherical_biconvex":
            if self.thinlens:
                for surface in self.surfaces: # if thinlens, intersect with the disc
                    intersected_here, int_pt = surface.test_intersect(ray)
                    if intersected_here:
                        intersected = True
                        distance_to_surface = distance_between(ray.position, int_pt)
                        if distance_to_surface < min_distance:
                            min_distance = distance_to_surface
                            intersection_pt = int_pt
            else:
                for surface in self.surfaces:  # TODO for now this is just the one disc
                    intersected_here, int_pt = surface.test_intersect(ray)
                    if intersected_here:
                        intersected = True
                        distance_to_surface = distance_between(ray.position, int_pt)
                        if distance_to_surface < min_distance:
                            min_distance = distance_to_surface
                            intersection_pt = int_pt
        elif self.shape == "circular_concave_spherical":
            for surface in self.surfaces:
                if isinstance(surface, Sphere):  # first check for intersection with the sphere
                    intersected_sphere, int_pt_sphere1, int_pt_sphere2 = surface.test_intersect(ray)
                    break
            if intersected_sphere:
                for surface in self.surfaces:
                    if isinstance(surface, Disc):  # then check for intersection with the disc
                        intersected_disc, int_pt_disc = surface.test_intersect(ray)
                        break
                if intersected_sphere and intersected_disc:
                    intersected = True  # pick the sphere intersection that is closest to the disc
                                        # TODO: edge cases, the above isn't exhaustive
                    distance1 = distance_between(int_pt_disc, int_pt_sphere1)
                    distance2 = distance_between(int_pt_disc, int_pt_sphere2)
                    if distance1 < distance2:
                        intersection_pt = int_pt_sphere1
                    else:
                        intersection_pt = int_pt_sphere2
        return intersected, intersection_pt




class Grating(Optic):
    def __init__(self, position, normal, tangent, shape="rectangular_flat", w=None, h=None, G=None,
                 order=None):
        self.position = position
        self.normal = unit_vector(normal)
        self.tangent = unit_vector(tangent)
        self.shape = shape
        self.w = w
        self.h = h
        self.G = G
        self.order = order
        self.surfaces = []
        if shape == "rectangular_flat":
            self.surfaces.append(Rectangle(position, normal, tangent, h=h, w=w))
        else:
            raise ValueError("Unknown shape")

    def __repr__(self):
        return("Grating ({}, {}, {}),  ({}, {}, {})".format(to_precision(self.position[0], 5),
                                                            to_precision(self.position[1], 5),
                                                            to_precision(self.position[2], 5),
                                                            to_precision(self.normal[0], 5),
                                                            to_precision(self.normal[1], 5),
                                                            to_precision(self.normal[2], 5)))

    def test_intersect(self, ray):  # Grating
        intersected = False
        intersection_pt = None
        min_distance = np.inf
        if self.shape == "circular_flat" or self.shape == "rectangular_flat":
            for surface in self.surfaces:  # for now this is just the one disc or rectangle
                intersected_here, int_pt = surface.test_intersect(ray)
                if intersected_here:
                    intersected = True
                    distance_to_surface = distance_between(ray.position, int_pt)
                    if distance_to_surface < min_distance:
                        min_distance = distance_to_surface
                        intersection_pt = int_pt
        return intersected, intersection_pt

    def draw(self, ax, view="3d"):  # Grating
        """Um, somehow draw the optic"""
        if view == "3d" or view == "xyz":
            if self.shape == "rectangular_flat":
                x = self.position[0]
                y = self.position[1]
                z = self.position[2]
                p = mplRectangle((-self.w / 2, -self.h / 2), width=self.w, height=self.h, facecolor="silver", alpha=0.5,
                                 edgecolor="black")
                ax.add_patch(p)
                pathpatch_2d_to_3d(p, z=0, normal=self.normal)
                pathpatch_translate(p, (x, y, z))
                # return the Artist created
                return None


class Detector(Optic):
    def __init__(self, position, normal, tangent, shape="rectangular_flat", w=None, h=None):
        self.position = position
        self.normal = unit_vector(normal)
        self.tangent = unit_vector(tangent)
        self.shape = shape
        self.w = w
        self.h = h
        self.surfaces = []
        self.hit_data = []
        if shape == "rectangular_flat":
            self.surfaces.append(Rectangle(position, normal, tangent, h=h, w=w))
        else:
            raise ValueError("Unknown shape")

    def __repr__(self):
        return("Detector ({}, {}, {}),  ({}, {}, {})".format(to_precision(self.position[0], 5),
                                                             to_precision(self.position[1], 5),
                                                             to_precision(self.position[2], 5),
                                                             to_precision(self.normal[0], 5),
                                                             to_precision(self.normal[1], 5),
                                                             to_precision(self.normal[2], 5)))

    def test_intersect(self, ray):  # Detector
        intersected = False
        intersection_pt = None
        min_distance = np.inf
        if self.shape == "circular_flat" or self.shape == "rectangular_flat":
            for surface in self.surfaces:  # for now this is just the one disc or rectangle
                intersected_here, int_pt = surface.test_intersect(ray)
                if intersected_here:
                    intersected = True
                    distance_to_surface = distance_between(ray.position, int_pt)
                    if distance_to_surface < min_distance:
                        min_distance = distance_to_surface
                        intersection_pt = int_pt
            if intersected:
                self.hit_data.append(np.append(intersection_pt, ray.wavelength))
        return intersected, intersection_pt

    def draw(self, ax, view="3d"):  # Detector
        """Um, somehow draw the optic"""
        if view == "3d" or view == "xyz":
            if self.shape == "rectangular_flat":
                # try instead:
                for surface in self.surfaces:
                    if isinstance(surface, Rectangle):
                        x = [surface.bounds[0, 0], surface.bounds[1, 0], surface.bounds[2, 0], surface.bounds[3, 0]]
                        y = [surface.bounds[0, 1], surface.bounds[1, 1], surface.bounds[2, 1], surface.bounds[3, 1]]
                        z = [surface.bounds[0, 2], surface.bounds[1, 2], surface.bounds[2, 2], surface.bounds[3, 2]]
                        verts = [list(zip(x, y, z))]
                        ax.add_collection3d(Poly3DCollection(verts, facecolor='gray', edgecolor='black'), zs=z)
                return None

        

class Annotation(Optic):
    pass


class Filter(Optic):
    pass


class Window(Optic):
    pass


class Block(Optic):
    pass




class Ray:
    def __init__(self, position, direction, wavelength=532, order=0, print_trajectory=False):
        self.position = position
        self.direction = unit_vector(direction)
        self.wavelength = wavelength # nm
        self.order = order # nm
        self.point_history = []
        self.point_history.append(position)
        self.print_trajectory = print_trajectory
        if self.print_trajectory:
            print(self)
        
    def __repr__(self):
        return("Ray ({}, {}, {}),  ({}, {}, {})".format(to_precision(self.position[0], 5),
                                                     to_precision(self.position[1], 5),
                                                     to_precision(self.position[2], 5),
                                                     to_precision(self.direction[0], 5),
                                                     to_precision(self.direction[1], 5),
                                                     to_precision(self.direction[2], 5)))

    def get_plot_repr(self):
        """Get the trajectory data of the ray for plotting"""
        return np.array(self.point_history)


    def fly(self, distance=None):
        """Fly a ray through space in a straight line."""
        if distance is not None:
            self.position = self.position + distance * self.direction
        else:
            raise ValueError("Insufficient correctness")
        self.point_history.append(self.position)
        if self.print_trajectory:
            print(self)

    def refract(self, refract_type="thin_lens", normal=None, optic=None, intersection_pt=None):
        """Refract a ray."""
        if refract_type == "thin_lens":
            # from "Thin Lens Ray Tracing", Gatland, 2002
            # equation 10, nbold_doubleprime = nbold - rbold / f
            r = intersection_pt - optic.position
            self.direction = self.direction - r / optic.f
        else:
            raise ValueError("Unrecognized refract_type input ")
        if self.print_trajectory:
            print(self)

    def reflect(self, reflect_type="specular_flat", normal=None, optic=None, intersection_pt=None):
        """
        Reflect the ray
        """
        if reflect_type == "specular_flat":
            i = unit_vector(self.direction)  # incident unit vector
            n = unit_vector(normal)  # normal unit vector
            r = i - 2 * np.dot(i, n) * n  # reflected
            self.direction = unit_vector(r)
        elif reflect_type == "grating":
            normal = unit_vector(normal)
            incident = unit_vector(self.direction)
            # get only angles in the grating active plane
            dispersion_plane_normal = optic.tangent
            n = unit_vector(project_onto_plane(normal, dispersion_plane_normal))
            i = unit_vector(project_onto_plane(incident, dispersion_plane_normal))
            alpha = angle_between(-i, n, dispersion_plane_normal)  # incident light angle with respect to grating normal (defined as plus)
#            print(f"dispersion_plane_normal = {dispersion_plane_normal}")
#            print(f"n = {n}")
#            print(f"i = {i}")
#            print(f"alpha = {alpha}")
            if alpha < 0:
                grating_sign_convention = -1
            else:
                grating_sign_convention = +1
            # the grating equation: G m lambda = sin(alpha) + sin(beta),
            # G = 1/d = groove density = "grooves per millimeter"
            # m = order (-1, 0, +1, etc)
            # d = groove spacing (mm)
            # sign convention: negative orders are towards the side of zero order
            #                  positive orders are backtowards the incident light
            beta = np.arcsin(optic.G / 1e6 * self.order * self.wavelength - np.sin(grating_sign_convention * alpha))
            # theta = n + grating_sign_convention * beta
            # TODO convert back to ray direction
            r = incident - 2 * np.dot(incident, normal) * normal  # reflected
            RM1 = rotation_matrix_axis_angle(dispersion_plane_normal, -grating_sign_convention * alpha)
#            self.direction = unit_vector(RM1.dot(r))
            RM2 = rotation_matrix_axis_angle(dispersion_plane_normal, -beta)
            self.direction = unit_vector(RM2.dot(unit_vector(RM1.dot(r))))
            if self.print_trajectory:
                print("light at wavelength {} turned {} by grating".format(self.wavelength, alpha + beta))
        else:
            raise ValueError(f"Unknown reflect type {reflect_type}")

    def test_intersect(self, optic, max_distance):
        """
        Test if the ray intersects a particular optic
        """
        intersected, intersection_pt = optic.test_intersect(self)  # self is the ray
        return intersected, intersection_pt




    def run(self, max_distance, optic_list=[], max_interactions=np.inf):
        """
        Run the ray through an optical system
        """
        interaction_count = 0
        distance_remaining = max_distance
        blocked = False
        if self.print_trajectory:
            print("--- run begin ---")
        while(distance_remaining > 0 and interaction_count < max_interactions and not blocked):
            intersected = False
            min_distance = np.inf
            for optic in optic_list:
                if isinstance(optic, Annotation):
                    continue
                elif isinstance(optic, Filter):
                    continue
                elif isinstance(optic, Window):
                    continue
                intersected_here, pt = self.test_intersect(optic, distance_remaining)
                if intersected_here:
                    intersected = True
                    distance_to_optic = distance_between(self.position, pt)
                    if distance_to_optic < min_distance:
                        min_distance = distance_to_optic
                        intersected_optic = optic
                        intersection_pt = pt
            if not intersected:
                if self.print_trajectory:
                    print("not intersected (messages TODO)")
                self.fly(distance=distance_remaining)
                distance_remaining -= max_distance
            else:
                if self.print_trajectory:
                    print("intersected (messages TODO)")
                distance_to_int_optic = distance_between(self.position, intersection_pt)
                self.fly(distance=distance_to_int_optic)
                distance_remaining -= distance_to_int_optic
                interaction_count += 1
                if isinstance(intersected_optic, Mirror):
                    if intersected_optic.shape == "circular_flat" or intersected_optic.shape == "rectangular_flat":
                        self.reflect(reflect_type="specular_flat",
                                     normal=intersected_optic.normal,
                                     intersection_pt=intersection_pt)
                    elif intersected_optic.shape == "circular_concave_spherical":
                        for surface in intersected_optic.surfaces:
                            if isinstance(surface, Sphere):
                                normal = surface.normal(intersection_pt)
                                break
                        self.reflect(reflect_type="specular_flat",
                                     normal=normal,
                                     intersection_pt=intersection_pt)
                    else:
                        raise ValueError("Unrecognized shape")
                elif isinstance(intersected_optic, Grating):
                    if self.order == 0:
                        self.reflect(reflect_type="specular_flat", normal=intersected_optic.normal,
                                     intersection_pt=intersection_pt)
                    else:
                        self.reflect(reflect_type="grating", normal=intersected_optic.normal,
                                     optic=intersected_optic, intersection_pt=intersection_pt)
                elif isinstance(intersected_optic, Lens):
                    self.refract(refract_type="thin_lens", normal=intersected_optic.normal,
                                 optic=intersected_optic, intersection_pt=intersection_pt)
                elif isinstance(intersected_optic, Detector):
                    blocked = True
                elif isinstance(intersected_optic, Block):
                    blocked = True
                elif isinstance(intersected_optic, Annotation):
                    pass
                elif isinstance(intersected_optic, Filter):
                    pass
                elif isinstance(intersected_optic, Window):
                    pass
                else:
                    raise NotImplementedError("Dude")
        if self.print_trajectory:
            print("--- run complete ---")






















