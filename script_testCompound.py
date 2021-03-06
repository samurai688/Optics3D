# -*- coding: utf-8 -*-

"""
Created on Sat Dec  29

@author: samuel
"""

import numpy as np
from optics3d import Ray, Compound
import matplotlib.pyplot as plt
from Shapes.shapes import Sphere
from general import set_axes_equal
from general_optics import BinaryTree
plt.close("all")

max_ray_run_distance = 150 # mm



mirror1_pos = np.array([0, 50, 0])
mirror2_pos = np.array([15, 50, 0])
mirror3_pos = np.array([30, 50, 0])

ray1_pos = np.array([0, 0, 0])
ray1_direction = np.array([0, 1, 0])

mirror1 = Sphere(mirror1_pos, D=25)
mirror2 = Sphere(mirror2_pos, D=25)
mirror3 = Sphere(mirror3_pos, D=25)
mirrorCOMPOUNDtree = BinaryTree("union")
mirrorCOMPOUNDtree.insertLeft(mirror1)
mirrorCOMPOUNDtree.insertRight("union")
mirrorCOMPOUNDtree.getRightChild().insertLeft(mirror2)
mirrorCOMPOUNDtree.getRightChild().insertRight(mirror3)
mirrorCOMPOUND = Compound(mirrorCOMPOUNDtree)
Optic_list = []
Optic_list.append(mirrorCOMPOUND)



Ray_list = []
ray_z = np.linspace(0, 0, 1)
ray_x = np.linspace(-20, 55, 76)
for x_ix, x_val in enumerate(ray_x):
    for z_ix, z_val in enumerate(ray_z):
        ray_pos = np.array([x_val, 0, z_val])
        ray_dir = np.array([0, 1, 0])
        ray = Ray(ray_pos, ray_dir, wavelength=532, print_trajectory=False)
        Ray_list.append(ray)



for ray in Ray_list:
    ray.run(max_distance=max_ray_run_distance, optic_list=Optic_list)


# 3d plots
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.set_aspect('equal')

for ray in Ray_list:
    ray_history = ray.get_plot_repr()
    ax.plot(ray_history[:, 0], ray_history[:, 1], ray_history[:, 2], "-r")
for optic in Optic_list:
    optic.draw(ax, view="3d")

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

set_axes_equal(ax)
plt.show()