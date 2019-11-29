# -*- coding: utf-8 -*-

"""
Created on Sat Dec  29

@author: samuel
"""

import numpy as np
from optics3d import Ray, Mirror
import matplotlib.pyplot as plt
from general import set_axes_equal
plt.close("all")

max_ray_run_distance = 150 # mm



mirror1_pos = np.array([0, 50, 0])
mirror1_normal = np.array([0, -1, 0])

ray1_pos = np.array([0, 0, 0])
ray1_direction = np.array([0, 1, 0])

mirror1 = Mirror(mirror1_pos, normal=mirror1_normal, shape="circular_convex_spherical", D=50, f=50)
Optic_list = []
Optic_list.append(mirror1)


Ray_list = []
ray_z = np.linspace(-24, 24, 25)
ray_x = np.linspace(-24, 24, 3)
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
