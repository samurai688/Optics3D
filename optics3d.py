#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 23:10:35 2018

@author: samuel
"""

import numpy as np
from general import to_precision
from general_optics import unit_vector, angle_between, distance_between, pathpatch_2d_to_3d, pathpatch_translate, \
    rotation_matrix_axis_angle, project_onto_plane
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Circle as mplCircle
from matplotlib.patches import Rectangle as mplRectangle
from Shapes.shapes import Rectangle, Disc, Sphere

# note: we are representing vectors as 1d numpy arrays, i.e., R = np.array([x, y, z])







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






















