#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 21:47:48 2019

@author: samuel
"""


import numpy as np
from optics3d import Ray, Lens, Compound
import matplotlib.pyplot as plt
from Shapes.shapes import Sphere
from general import set_axes_equal
from general_optics import BinaryTree
plt.close("all")



ball1_pos = np.array([0, 145, 0])
ball2_pos = np.array([0, 55, 0])
lens1 = Sphere(ball1_pos, D=100)
lens2 = Sphere(ball2_pos, D=100)

lensCOMPOUNDtree = BinaryTree("intersect")
lensCOMPOUNDtree.insertLeft(lens1)
lensCOMPOUNDtree.insertRight(lens2)
lensCOMPOUND = Compound(lensCOMPOUNDtree, surface_behavior="refract", index=1.5)

Optic_list = []
Optic_list.append(lensCOMPOUND)

max_ray_run_distance = 175

x_start = [0]
y_start = [0]
z_start = np.linspace(-10, 10, 21)

Ray_list = []
for x_val in x_start:
    for y_val in y_start:
        for z_val in z_start:
            Ray_list.append(Ray(np.array([x_val, y_val, z_val]), np.array([0, 1, 0]), wavelength=532, print_trajectory=False))



for ray in Ray_list:
    ray.run(max_distance=max_ray_run_distance, optic_list=Optic_list)



# 2d plots

fig = plt.figure()
ax = plt.axes()
for ray in Ray_list:
    ray_history = ray.get_plot_repr()
    ax.plot(ray_history[:, 1], ray_history[:, 2], "-r")




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