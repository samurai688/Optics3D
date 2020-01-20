# -*- coding: utf-8 -*-

"""
Created on Sat Dec  29

@author: samuel
"""

import numpy as np
from optics3d import Ray, Compound, add_faerie_fire_rays
import matplotlib.pyplot as plt
from Shapes.shapes import Sphere
from general import set_axes_equal
from general_optics import BinaryTree
plt.close("all")

max_ray_run_distance = 150 # mm



center_pos = np.array([0, 50, 0])
mirror1_pos = np.array([0, 80, 0])
mirror2_pos = np.array([0, 20, 0])
mirror1 = Sphere(mirror1_pos, D=65)
mirror2 = Sphere(mirror2_pos, D=65)


mirrorCOMPOUNDtree = BinaryTree("intersect")
mirrorCOMPOUNDtree.insertLeft(mirror1)
mirrorCOMPOUNDtree.insertRight(mirror2)
mirrorCOMPOUND = Compound(mirrorCOMPOUNDtree, surface_behavior="reflect", index=1.5)


Optic_list = []
Optic_list.append(mirrorCOMPOUND)

Ray_list = []

FF_radius = max_ray_run_distance - 50
FF_center = center_pos
Ray_list = add_faerie_fire_rays(Ray_list, FF_radius, FF_center)


for ray in Ray_list:
    ray.run(max_distance=max_ray_run_distance, optic_list=Optic_list)



# 3d plots
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.set_aspect('equal')

for ray in Ray_list:
    ray_history = ray.get_plot_repr()
    if ray.type == "normal":
        ax.plot(ray_history[:, 0], ray_history[:, 1], ray_history[:, 2], "-r")
    elif ray.type == "faerie_fire":
        ax.plot(ray_history[1:, 0], ray_history[1:, 1], ray_history[1:, 2], "ob", MarkerSize=0.5)
for optic in Optic_list:
    optic.draw(ax, view="3d")

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

set_axes_equal(ax)
plt.show()