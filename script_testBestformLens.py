#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 21:47:48 2019

@author: samuel
"""


import numpy as np
from optics3d import Ray, Lens, Compound, add_faerie_fire_rays, get_index_at_point
import matplotlib.pyplot as plt
from Shapes.shapes import Sphere
from general import set_axes_equal
from general_optics import BinaryTree
plt.close("all")

# this dimensions might not be precisely right but it does seem slightly better than the singlet already
y_backprinciple = 400
center_thick = 4.0
edge_thick = 2.4
f = 100
f_back = 97.3
r1 = 60.02
r2 = 353.30
lens_dia = 25.4
h_ = lens_dia / 2
x1_ = r1 - h_ * np.tan(np.arccos(h_ / r1))
x2_ = r2 - h_ * np.tan(np.arccos(h_ / r2))


zero_pos = np.array([0, 0, 0])
center_pos = np.array([0, y_backprinciple, 0])
ball1_pos = np.array([0, y_backprinciple + r1 - (center_thick - (f - f_back)), 0])
ball2_pos = np.array([0, y_backprinciple - r2 + (f - f_back), 0])
lens1 = Sphere(ball1_pos, D=r1*2)
lens2 = Sphere(ball2_pos, D=r2*2)

lensCOMPOUNDtree = BinaryTree("intersect")
lensCOMPOUNDtree.insertLeft(lens1)
lensCOMPOUNDtree.insertRight(lens2)
lensCOMPOUND = Compound(lensCOMPOUNDtree, surface_behavior="refract", index=1.5167)

Optic_list = []
Optic_list.append(lensCOMPOUND)

test_index_of_world = get_index_at_point(Optic_list, zero_pos)
test_index_of_optic = get_index_at_point(Optic_list, center_pos)
print(f"test_index_of_world = {test_index_of_world}")
print(f"test_index_of_optic = {test_index_of_optic}")

max_ray_run_distance = 550

x_start = [0]
y_start = [0]
z_start = np.linspace(-25.4/4, 25.4/4, 21)

Ray_list = []
for x_val in x_start:
    for y_val in y_start:
        for z_val in z_start:
            Ray_list.append(Ray(np.array([x_val, y_val, z_val]), np.array([0, 1, 0]), wavelength=532, print_trajectory=False))



FF_radius = max_ray_run_distance - 100
FF_center = np.array([0, y_backprinciple, 0])
Ray_list = add_faerie_fire_rays(Ray_list, FF_radius, FF_center)


for ray in Ray_list:
    ray.run(max_distance=max_ray_run_distance, optic_list=Optic_list)



# 2d plots

fig = plt.figure()
ax = plt.axes()
for ray in Ray_list:
    ray_history = ray.get_plot_repr()
    if ray.type == "normal":
        ax.plot(ray_history[:, 1], ray_history[:, 2], "-r")
    elif ray.type == "faerie_fire":
        ax.plot(ray_history[1:, 1], ray_history[1:, 2], "ob", MarkerSize=1)




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