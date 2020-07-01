#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 21:47:48 2019

@author: samuel
"""


import numpy as np
from optics3d import Ray, Lens, Detector
import matplotlib.pyplot as plt
from general import set_axes_equal, wavelength_to_rgb
from general_optics import image_rays
plt.close("all")


image_f_number = 8
object_size = 10  # mm
image_half_angle = np.arctan(1 / 2 / image_f_number)  # radians


lens_f1 = 50
lens_f2 = 25
object_y = 0
lens_y1 = 50 # 50
lens_y2 = 75
object_dist = lens_y1 - object_y
image_dist = 2 * lens_f2
image_y = lens_y2 + image_dist
mag = image_dist / object_dist
image_size = object_size * mag
print(f"object_size = {object_size}")
print(f"object_dist = {object_dist}")
print(f"image_dist = {image_dist}")
print(f"image_y = {image_y}")
print(f"mag = {mag}")
print(f"image_size = {image_size}")



lens_center = np.array([0, lens_y1, 0])
lens_normal = np.array([0, -1, 0])
lens_tangent = np.array([0, 0, 1])
lens1 = Lens(lens_center, normal=lens_normal, shape="spherical_biconvex",
             tangent=lens_tangent, D=50, type="ideal_collimate", f=lens_f1)

lens_center = np.array([0, lens_y2, 0])
lens_normal = np.array([0, -1, 0])
lens_tangent = np.array([0, 0, 1])
lens2 = Lens(lens_center, normal=lens_normal, shape="spherical_biconvex",
             tangent=lens_tangent, D=50, type="ideal_focus", f=lens_f2)

detector_center = np.array([0, image_y, 0])
detector_normal = np.array([0, -1, 0])
detector_tangent = np.array([0, 0, 1])
detector_h = 30
detector_w = 30
detector1 = Detector(detector_center, normal=detector_normal, shape="rectangular_flat",
                     tangent=detector_tangent, h=detector_h, w=detector_w)


Optic_list = []
Optic_list.append(lens1)
Optic_list.append(lens2)
Optic_list.append(detector1)

max_ray_run_distance = 150

Ray_list = []
origins, dirs, waves = image_rays(0, size=object_size, angle=image_half_angle, x_res=1, z_res=1, angle_res=5)
for ix, origin in enumerate(origins):
    Ray_list.append(Ray(origins[ix], dirs[ix], wavelength=waves[ix], print_trajectory=False))



for ray in Ray_list:
    ray.run(max_distance=max_ray_run_distance, optic_list=Optic_list)



# 2d plots

fig = plt.figure()
ax = plt.axes()
for ray in Ray_list:
    ray_history = ray.get_plot_repr()
    ax.plot(ray_history[:, 1], ray_history[:, 2], "-b")
plt.grid(True)
plt.axis("equal")



# detector plot
fig = plt.figure()
intensifier_diam = 25
theta_intensifier = np.linspace(0, 2 * np.pi, 361)
x_intensifier = intensifier_diam/2 * np.cos(theta_intensifier)
z_intensifier = intensifier_diam/2 * np.sin(theta_intensifier)
plt.plot(detector_center[0] + x_intensifier, detector_center[2] + z_intensifier, "-k")
for datarow in detector1.hit_data:
    plt.plot(datarow[0], datarow[2], 'o', markersize=2, color=wavelength_to_rgb(datarow[3]))
plt.xlim(detector_center[0] - detector_w/2, detector_center[0] + detector_w/2)
plt.ylim(detector_center[2] - detector_h/2, detector_center[2] + detector_h/2)
plt.xlabel("x (mm)")
plt.ylabel("z (mm)")





# 3d plots
fig = plt.figure()
ax = plt.axes(projection='3d')

for optic in Optic_list:
    optic.draw(ax, view="3d")
for ray in Ray_list:
    ray_history = ray.get_plot_repr()
    ax.plot(ray_history[:, 0], ray_history[:, 1], ray_history[:, 2], "-b")

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

set_axes_equal(ax)
plt.show()