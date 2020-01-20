#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 23:10:35 2018

@author: samuel
"""

import numpy as np
from general import to_precision
from general_optics import unit_vector, angle_between, distance_between, pathpatch_2d_to_3d, pathpatch_translate, \
    rotation_matrix_axis_angle, project_onto_plane, postOrderEval, test_tree_intersect, test_tree_point
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Circle as mplCircle
from matplotlib.patches import Rectangle as mplRectangle
from Shapes.shapes import Rectangle, Disc, Sphere

# note: we are representing vectors as 1d numpy arrays, i.e., R = np.array([x, y, z])



INTERSECT_CLIPPING_FLOOR = 1e-12
INDEX_OF_THE_WORLD = 1.0000000


# faerie fire function aka the world's worst 3d graphics
def add_faerie_fire_rays(Ray_list, FF_radius, FF_center):
    N = 1000
    for i in range(N):
        v = np.array([0, 0, 0])  # initialize so we go into the while loop
        while np.linalg.norm(v) < .000001:
            x = np.random.normal()  # random standard normal
            y = np.random.normal()
            z = np.random.normal()
            v = np.array([x, y, z])
        v = v / np.linalg.norm(v)  # normalize to unit norm
        v_dir = -v
        v_ff = FF_center + v * FF_radius # scale and shift to problem
        Ray_list.append(Ray(v_ff, v_dir, wavelength=532, print_trajectory=False, type="faerie_fire"))
    return Ray_list


# test the whole world
def get_index_at_point(Optic_list, point):
    for optic in Optic_list:
        index = optic.get_index_at_point_if_inside(point)
        if index is not None:
            return index
    return INDEX_OF_THE_WORLD







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

    def do_intersect(self, ray, int_pt, int_normal, shooting_from_outside):
        # do nothing
        return ray

    def draw(self, ax, view="3d"):
        pass

    def get_index_at_point_if_inside(self, point):
        pass




class Compound(Optic):
    def __init__(self, tree, surface_behavior="reflect", index=1.5):
        self.tree = tree;
        self.surface_behavior = surface_behavior
        self.index = index

    def __repr__(self):
        return ("Compound")

    def test_intersect(self, ray):
        intersected, pt1, pt2, norm1, norm2, intersection_count = test_tree_intersect(self.tree, ray)
        # if intersection_count is an even number, we're shooting from outside
        if np.isclose(np.mod(intersection_count, 2), 0): # if even number:
            shooting_from_outside = True
        else:
            shooting_from_outside = False
        return intersected, pt1, norm1, shooting_from_outside

    def do_intersect(self, ray, intersection_pt, intersection_normal, shooting_from_outside):
        # (f"intersection_pt = {intersection_pt}")
        # print(f"intersection_normal = {intersection_normal}")
        if self.surface_behavior == "reflect":
            ray.reflect(reflect_type="specular_flat",
                        normal=intersection_normal,
                        intersection_pt=intersection_pt)
        elif self.surface_behavior == "refract":
            # subscript 1 is the material you are coming from
            # subscript 2 is the material you are going into
            # intersection_normal direction should get fixed in ray.refract
            optic_index = self.get_index_at_point_if_inside(intersection_pt + 1e-8 * ray.direction)
            if optic_index is not None:
                eta1 = INDEX_OF_THE_WORLD
                eta2 = self.index
            else:
                eta1 = self.index
                eta2 = INDEX_OF_THE_WORLD
            int_normal = intersection_normal
            # print(f"shooting_from_outside = {shooting_from_outside}")
            # print(f"int_normal = {int_normal}")
            ray.refract(refract_type="snells_law",
                        normal=int_normal,
                        optic=self,
                        intersection_pt=intersection_pt,
                        eta1=eta1,
                        eta2=eta2)
        else:
            raise ValueError("Unknown value for surface_behavior")

    def get_index_at_point_if_inside(self, point):
        inside_final_optic = test_tree_point(self.tree, point)
        if inside_final_optic:
            return self.index
        else:
            return None

    def draw(self, ax, view="3d"):
        pass




class Mirror(Optic):
    def __init__(self, position, shape="circular_flat", normal=None, D=None, w=None, h=None,
                 thickness=10, f=None, tangent=None):
        self.position = position
        self.shape = shape
        self.D = D
        self.w = w
        self.h = h
        self.thickness = thickness
        self.f = f
        self.surfaces = []
        if shape == "circular_flat":
            self.normal = unit_vector(normal)
            self.surfaces.append(Disc(position, normal, D=D))
        elif shape == "rectangular_flat":
            self.normal = unit_vector(normal)
            self.surfaces.append(Rectangle(position, normal, tangent, h=h, w=w))
        elif shape == "circular_concave_spherical":
            self.normal = unit_vector(normal)
            self.rc = self.f * 2
            self.p_rc = self.position + self.normal * self.rc
            self.surfaces.append(Disc(position, normal, D=D))
            self.surfaces.append(Sphere(center=self.p_rc, R=self.rc))
        elif shape == "circular_convex_spherical":
            self.surfaces.append(Sphere(center=self.position, D=self.D))
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
        shooting_from_outside = None ## TODO
        if self.shape == "circular_flat" or self.shape == "rectangular_flat":
            for surface in self.surfaces:  # for now this is just the one disc or rectangle
                intersected_here, int_pt, normal = surface.test_intersect(ray)
                if intersected_here:
                    intersected = True
                    distance_to_surface = distance_between(ray.position, int_pt)
                    if distance_to_surface < min_distance:
                        min_distance = distance_to_surface
                        intersection_pt = int_pt
                        normal = surface.normal
        elif self.shape == "circular_concave_spherical":
            normal = None
            for surface in self.surfaces:
                if isinstance(surface, Sphere):  # first check for intersection with the sphere
                    intersected_sphere, int_pt_sphere1, int_pt_sphere2, norm1, norm2, shooting_from_outside = surface.test_intersect(ray)
                    break
            if intersected_sphere:
                for surface in self.surfaces:
                    if isinstance(surface, Disc):  # then check for intersection with the disc
                        intersected_disc, int_pt_disc, normal_disc = surface.test_intersect(ray)
                        break
                if intersected_sphere and intersected_disc:
                    intersected = True  # pick the sphere intersection that is closest to the disc
                                        # TODO: edge cases, the above isn't exhaustive
                    distance1 = distance_between(int_pt_disc, int_pt_sphere1)
                    if int_pt_sphere2 is not None:
                        distance2 = distance_between(int_pt_disc, int_pt_sphere2)
                        if distance1 < distance2:
                            intersection_pt = int_pt_sphere1
                            normal = norm1
                        else:
                            intersection_pt = int_pt_sphere2
                            normal = norm2
                    else:
                        intersection_pt = int_pt_sphere1
                        normal = norm1
        elif self.shape == "circular_convex_spherical":
            normal = None
            for surface in self.surfaces:
                if isinstance(surface, Sphere):  # first check for intersection with the sphere
                    intersected_sphere, int_pt_sphere1, int_pt_sphere2, norm1, norm2, shooting_from_outside = surface.test_intersect(ray)
                if intersected_sphere:
                    intersected = True
                    distance_to_surface = distance_between(ray.position, int_pt_sphere1)
                    if distance_to_surface < min_distance:
                        min_distance = distance_to_surface
                        intersection_pt = int_pt_sphere1
                        normal = -norm1
                    if int_pt_sphere2 is not None:
                        distance_to_surface = distance_between(ray.position, int_pt_sphere2)
                        if distance_to_surface < min_distance:
                            min_distance = distance_to_surface
                            intersection_pt = int_pt_sphere2
                            normal = -norm2
        else:
            raise ValueError("unknown shape")
        return intersected, intersection_pt, normal, shooting_from_outside

    def do_intersect(self, ray, intersection_pt, intersection_normal, shooting_from_outside):
        ray.reflect(reflect_type="specular_flat",
                    normal=intersection_normal,
                    intersection_pt=intersection_pt)

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
            elif self.shape == "circular_convex_spherical":
                N = 50
                stride = 2
                u = np.linspace(0, 2 * np.pi, N)
                v = np.linspace(0, np.pi, N)
                x = self.position[0] + self.D / 2 * np.outer(np.cos(u), np.sin(v))
                y = self.position[1] + self.D / 2 * np.outer(np.sin(u), np.sin(v))
                z = self.position[2] + self.D / 2 * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, linewidth=1.0, cstride=stride, rstride=stride, alpha=0.5)
            else:
                raise ValueError("unknown shape")


class Lens(Optic):
    def __init__(self, position, normal=None, shape="spherical_biconvex", D=None, w=None, h=None,
                 f=None, tangent=None, thinlens=False, index=1.5):
        self.position = position
        if normal is not None:
            self.normal = unit_vector(normal)
        else:
            self.normal = normal
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
        elif shape == "spherical":
            self.surfaces.append(Sphere(position, D=D))
        else:
            raise ValueError("Unknown shape")


    def __repr__(self):
        return("Lens ({}, {}, {}),  ({}, {}, {})".format(to_precision(self.position[0], 5),
                                                           to_precision(self.position[1], 5),
                                                           to_precision(self.position[2], 5),
                                                           to_precision(self.normal[0], 5),
                                                           to_precision(self.normal[1], 5),
                                                           to_precision(self.normal[2], 5)))

    def test_intersect(self, ray):  # Lens
        intersected = False
        intersection_pt = None
        normal = None
        min_distance = np.inf
        shooting_from_outside = True
        if self.shape == "spherical_biconvex":
            normal = None
            if self.thinlens:
                for surface in self.surfaces: # if thinlens, intersect with the disc
                    intersected_here, int_pt, normal = surface.test_intersect(ray)
                    if intersected_here:
                        intersected = True
                        distance_to_surface = distance_between(ray.position, int_pt)
                        if distance_to_surface < min_distance:
                            min_distance = distance_to_surface
                            intersection_pt = int_pt
            else:
                for surface in self.surfaces:  # TODO implement as csg compound
                    intersected_here, int_pt, normal = surface.test_intersect(ray)
                    if intersected_here:
                        intersected = True
                        distance_to_surface = distance_between(ray.position, int_pt)
                        if distance_to_surface < min_distance:
                            min_distance = distance_to_surface
                            intersection_pt = int_pt
        elif self.shape == "spherical":
            for surface in self.surfaces:
                intersected_sphere = False
                if isinstance(surface, Sphere):  # check for intersection with the sphere
                    intersected_sphere, int_pt_sphere1, int_pt_sphere2, norm1, norm2, intersection_count = surface.test_intersect(ray)
                    # if intersection_count is an even number, we're shooting from outside
                    if np.isclose(np.mod(intersection_count, 2), 0):  # if even number:
                        shooting_from_outside = True
                    else:
                        shooting_from_outside = False
                if intersected_sphere:
                    intersected = True
                    distance_to_surface = distance_between(ray.position, int_pt_sphere1)
                    if distance_to_surface < min_distance:
                        min_distance = distance_to_surface
                        intersection_pt = int_pt_sphere1
                        normal = -norm1
                    if int_pt_sphere2 is not None:
                        distance_to_surface = distance_between(ray.position, int_pt_sphere2)
                        if distance_to_surface < min_distance:
                            min_distance = distance_to_surface
                            intersection_pt = int_pt_sphere2
                            normal = -norm2
        else:
            raise ValueError("Unhandled shape")
        return intersected, intersection_pt, normal, shooting_from_outside

    def do_intersect(self, ray, intersection_pt, intersection_normal, shooting_from_outside):
        if self.thinlens:
            ray.refract(refract_type="thin_lens",
                        normal=intersection_normal,
                        optic=self,
                        intersection_pt=intersection_pt)
        else:
            # subscript 1 is the material you are coming from
            # subscript 2 is the material you are going into
            if shooting_from_outside:
                eta1 = 1
                eta2 = self.index
                int_normal = intersection_normal
            else:
                eta1 = self.index
                eta2 = 1
                int_normal = -intersection_normal
            ray.refract(refract_type="snells_law",
                        normal=int_normal,
                        optic=self,
                        intersection_pt=intersection_pt,
                        eta1=eta1,
                        eta2=eta2)

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
            elif self.shape == "spherical":
                N = 50
                stride = 2
                u = np.linspace(0, 2 * np.pi, N)
                v = np.linspace(0, np.pi, N)
                x = self.position[0] + self.D / 2 * np.outer(np.cos(u), np.sin(v))
                y = self.position[1] + self.D / 2 * np.outer(np.sin(u), np.sin(v))
                z = self.position[2] + self.D / 2 * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, linewidth=1.0, cstride=stride, rstride=stride, alpha=0.5)
            else:
                raise ValueError("unknown shape")


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
        shooting_from_outside = None
        if self.shape == "circular_flat" or self.shape == "rectangular_flat":
            for surface in self.surfaces:  # for now this is just the one disc or rectangle
                intersected_here, int_pt, normal = surface.test_intersect(ray)
                if intersected_here:
                    intersected = True
                    distance_to_surface = distance_between(ray.position, int_pt)
                    if distance_to_surface < min_distance:
                        min_distance = distance_to_surface
                        intersection_pt = int_pt
        return intersected, intersection_pt, normal, shooting_from_outside

    def do_intersect(self, ray, intersection_pt, intersection_normal, shooting_from_outside):
        if ray.order == 0:
            ray.reflect(reflect_type="specular_flat",
                        normal=intersection_normal,
                        intersection_pt=intersection_pt)
        else:
            ray.reflect(reflect_type="grating",
                        normal=intersection_normal,
                        optic=self,
                        intersection_pt=intersection_pt)

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
        shooting_from_outside = None
        if self.shape == "circular_flat" or self.shape == "rectangular_flat":
            for surface in self.surfaces:  # for now this is just the one disc or rectangle
                intersected_here, int_pt, normal = surface.test_intersect(ray)
                if intersected_here:
                    intersected = True
                    distance_to_surface = distance_between(ray.position, int_pt)
                    if distance_to_surface < min_distance:
                        min_distance = distance_to_surface
                        intersection_pt = int_pt
            if intersected:
                self.hit_data.append(np.append(intersection_pt, ray.wavelength))
        return intersected, intersection_pt, normal, shooting_from_outside

    def do_intersect(self, ray, int_pt, int_normal, shooting_from_outside):
        ray.blocked = True

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

    def do_intersect(self, ray, int_pt, int_normal, shooting_from_outside):
        ray.blocked = True


class Ray:
    def __init__(self, position, direction, wavelength=532, order=0, print_trajectory=False, type="normal"):
        self.position = position
        self.direction = unit_vector(direction)
        self.wavelength = wavelength # nm
        self.order = order # nm
        self.point_history = []
        self.point_history.append(position)
        self.print_trajectory = print_trajectory
        self.blocked = False
        self.type = type
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

    def refract(self, refract_type="thin_lens", normal=None, optic=None, intersection_pt=None, eta1=None, eta2=None):
        """Refract a ray."""
        if refract_type == "thin_lens":
            # from "Thin Lens Ray Tracing", Gatland, 2002
            # equation 10, nbold_doubleprime = nbold - rbold / f
            r = intersection_pt - optic.position
            self.direction = self.direction - r / optic.f
        elif refract_type == "snells_law":
            # from 2006 internet pdf:
            # "Reflections and Refractions in Ray Tracing"
            # by Bram de Greve (bram.degreve@gmail.com)
            #           November 13, 2006
            #
            # subscript 1 is the material you are coming from
            # subscript 2 is the material you are going into
            # eta is index of refraction
            #
            # surface normal should face toward material you are coming from
            #

            do_total_internal_reflection = False
            i = self.direction
            n = normal

            # okay
            # let's just try to flip the normal in here so that it's right to prevent all these issues
            # I see no way this could go poorly
            # so, the angle between the ray direction and the normal should be more than 90 right? that's what it is?
            # If the angle between A and B are greater than 90 degrees, the dot product will be negative, if not flip n
            if np.dot(i, n) < 0:
                pass
            else:
                n = -n

            cos_theta_incident = np.dot(-i, n)
            sin_squared_theta_transmitted = (eta1 / eta2) ** 2 * (1 - cos_theta_incident ** 2)

            if eta1 > eta2:
                if sin_squared_theta_transmitted > 1:  # eq. 24
                    do_total_internal_reflection = True
                    self.reflect(reflect_type="specular_flat",
                                 normal=normal,
                                 intersection_pt=intersection_pt)

            print("TIR: " + str(do_total_internal_reflection))
            if not do_total_internal_reflection:
                t = (eta1 / eta2) * i + ((eta1 / eta2) * cos_theta_incident - np.sqrt(1 - sin_squared_theta_transmitted)) * n
                self.direction = t

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

    def run(self, max_distance, optic_list=[], max_interactions=np.inf):
        """
        Run the ray through an optical system
        """
        interaction_count = 0
        distance_remaining = max_distance
        if self.print_trajectory:
            print("--- run begin ---")
        while(distance_remaining > 0 and interaction_count < max_interactions and not self.blocked):
            intersected = False
            min_distance = np.inf
            for optic in optic_list:
                if isinstance(optic, Annotation):
                    continue
                elif isinstance(optic, Filter):
                    continue
                elif isinstance(optic, Window):
                    continue
                # do it:
                intersected_here, int_pt, normal, shooting_from_outside = optic.test_intersect(self)
                # we did it
                if intersected_here:
                    distance_to_optic = distance_between(self.position, int_pt)
                    if distance_to_optic > distance_remaining:
                        pass  # allow intersected to remain False, pass through to fly distance_remaining
                    elif distance_to_optic > INTERSECT_CLIPPING_FLOOR:
                        intersected = True  # hit
                        if distance_to_optic < min_distance:  # set intersect the nearest optic
                            min_distance = distance_to_optic
                            intersected_optic = optic
                            intersection_pt = int_pt
                            intersection_normal = normal
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
                if self.type == "faerie_fire":
                    self.blocked = True
                else:
                    # do it:
                    intersected_optic.do_intersect(self, intersection_pt, intersection_normal, shooting_from_outside)
                    # we did it
        if self.print_trajectory:
            print("--- run complete ---")






















