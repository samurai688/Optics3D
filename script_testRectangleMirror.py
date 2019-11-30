# -*- coding: utf-8 -*-

"""
Created on Sat Dec  8 23:47:35 2018

@author: samuel
"""

import numpy as np
from optics3d import Ray, Mirror
import matplotlib.pyplot as plt
from general import set_axes_equal
plt.close("all")

mir_center = np.array([0, 50, 0])
mir_normal = np.array([0.1, -1, 0])
mir_tangent = np.array([0, 0, 1])

mirror1 = Mirror(mir_center, normal=mir_normal, shape="rectangular_flat", tangent=mir_tangent, h=50, w=50)
Optic_list = []
Optic_list.append(mirror1)

max_ray_run_distance = 150

ray = Ray(np.array([0, 0, 0]), np.array([0, 1, 0]), wavelength=532, print_trajectory=True)
Ray_list = []
Ray_list.append(ray)


for ray in Ray_list:
    ray.run(max_distance=max_ray_run_distance, optic_list=Optic_list)


# 3d plots
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.set_aspect("equal")
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


