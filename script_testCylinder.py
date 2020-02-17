#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 21:47:48 2019

@author: samuel
"""


import numpy as np
from optics3d import Ray, Compound, add_faerie_fire_rays
import matplotlib.pyplot as plt
from Shapes.shapes import InfiniteCylinder
from general import set_axes_equal
from general_optics import BinaryTree
plt.close("all")

y_center = 0
center_pos = np.array([0, 2, 0])
cyl1_pos = np.array([0, 2, 0])
shape1 = InfiniteCylinder(cyl1_pos, R=2, direction=[0, 1, 1])

shapeCOMPOUNDtree = BinaryTree(shape1)
shapeCOMPOUND = Compound(shapeCOMPOUNDtree, surface_behavior="reflect", index=1.5167)

Optic_list = []
Optic_list.append(shapeCOMPOUND)


max_ray_run_distance = 10

x_start = [-6]
y_start = [2]
z_start = np.linspace(-2.0, 2.0, 21)

Ray_list = []
for x_val in x_start:
    for y_val in y_start:
        for z_val in z_start:
            Ray_list.append(Ray(np.array([x_val, y_val, z_val]), np.array([1, 0, 0]), wavelength=532, print_trajectory=False))



FF_radius = max_ray_run_distance * 0.85
FF_center = center_pos
Ray_list = add_faerie_fire_rays(Ray_list, FF_radius, FF_center)



for ray in Ray_list:
    ray.run(max_distance=max_ray_run_distance, optic_list=Optic_list)



# 2d plots

fig = plt.figure()
ax = plt.axes()
for ray in Ray_list:
    ray_history = ray.get_plot_repr()
    if ray.type == "normal":
        ax.plot(ray_history[:, 0], ray_history[:, 2], "-r")
    elif ray.type == "faerie_fire":
        ax.plot(ray_history[1:, 0], ray_history[1:, 2], "ob", MarkerSize=1)
plt.axis("equal")




# 3d plots
fig = plt.figure()
ax = plt.axes(projection='3d')

for optic in Optic_list:
    optic.draw(ax, view="3d")
for ray in Ray_list:
    ray_history = ray.get_plot_repr()
    if ray.type == "normal":
        ax.plot(ray_history[:, 0], ray_history[:, 1], ray_history[:, 2], "-r")
    elif ray.type == "faerie_fire":
        ax.plot(ray_history[1:, 0], ray_history[1:, 1], ray_history[1:, 2], "ob", MarkerSize=0.5)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

set_axes_equal(ax)
plt.show()