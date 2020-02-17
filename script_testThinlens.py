#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 21:47:48 2019

@author: samuel
"""


import numpy as np
from optics3d import Ray, Lens
import matplotlib.pyplot as plt
from general import set_axes_equal
plt.close("all")

lens_center = np.array([0, 50, 0])
lens_normal = np.array([0, -1, 0])
lens_tangent = np.array([0, 0, 1])

lens1 = Lens(lens_center, normal=lens_normal, shape="spherical_biconvex",
             tangent=lens_tangent, D=50, type="ideal", f=100)
Optic_list = []
Optic_list.append(lens1)

max_ray_run_distance = 150

x_start = [-15, 0, 15]
y_start = [0]
z_start = [-15, 0, 15]

Ray_list = []
for x_val in x_start:
    for y_val in y_start:
        for z_val in z_start:
            Ray_list.append(Ray(np.array([x_val, y_val, z_val]), np.array([0, 1, 0]), wavelength=532, print_trajectory=False))



for ray in Ray_list:
    ray.run(max_distance=max_ray_run_distance, optic_list=Optic_list)


# 3d plots
fig = plt.figure()
ax = plt.axes(projection='3d')

for optic in Optic_list:
    optic.draw(ax, view="3d")
for ray in Ray_list:
    ray_history = ray.get_plot_repr()
    ax.plot(ray_history[:, 0], ray_history[:, 1], ray_history[:, 2], "-r")

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

set_axes_equal(ax)
plt.show()