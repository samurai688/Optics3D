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

y_center = 400
r1 = 102.4
center_thick = 3.6
ball1_pos = np.array([0, y_center + r1 - center_thick/2, 0])
ball2_pos = np.array([0, y_center - r1 + center_thick/2, 0])
lens1 = Sphere(ball1_pos, D=r1*2)
lens2 = Sphere(ball2_pos, D=r1*2)

lensCOMPOUNDtree = BinaryTree("intersect")
lensCOMPOUNDtree.insertLeft(lens1)
lensCOMPOUNDtree.insertRight(lens2)
lensCOMPOUND = Compound(lensCOMPOUNDtree, surface_behavior="refract", index=1.5167)

Optic_list = []
Optic_list.append(lensCOMPOUND)

max_ray_run_distance = 550

x_start = [0]
y_start = [0]
z_start = np.linspace(-10, 10, 21)

Ray_list = []
for x_val in x_start:
    for y_val in y_start:
        for z_val in z_start:
            Ray_list.append(Ray(np.array([x_val, y_val, z_val]), np.array([0, 1, 0]), wavelength=532, print_trajectory=False))



FF_radius = max_ray_run_distance - 100
FF_center = np.array([0, y_center, 0])
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
    Ray_list.append(Ray(v_ff, v_dir, wavelength=532, print_trajectory=False, type="fairie_fire"))



for ray in Ray_list:
    ray.run(max_distance=max_ray_run_distance, optic_list=Optic_list)



# 2d plots

fig = plt.figure()
ax = plt.axes()
for ray in Ray_list:
    ray_history = ray.get_plot_repr()
    if ray.type == "normal":
        ax.plot(ray_history[:, 1], ray_history[:, 2], "-r")
    elif ray.type == "fairie_fire":
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
    elif ray.type == "fairie_fire":
        ax.plot(ray_history[1:, 0], ray_history[1:, 1], ray_history[1:, 2], "ob", MarkerSize=0.5)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

set_axes_equal(ax)
plt.show()