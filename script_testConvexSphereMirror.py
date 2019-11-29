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

max_ray_run_distance = 400 # mm



mirror1_pos = np.array([0, 50, 0])
mirror2_pos = np.array([0, -50, 50])
mirror3_pos = np.array([0, 50, 50])
mirror4_pos = np.array([0, -50, 100])

ray1_pos = np.array([0, 0, 0])
ray1_direction = np.array([0, 1, 0])

mirror1 = Mirror(mirror1_pos, shape="circular_convex_spherical", D=25)
mirror2 = Mirror(mirror2_pos, shape="circular_convex_spherical", D=25)
mirror3 = Mirror(mirror3_pos, shape="circular_convex_spherical", D=25)
mirror4 = Mirror(mirror4_pos, shape="circular_convex_spherical", D=25)
Optic_list = []
Optic_list.append(mirror1)
Optic_list.append(mirror2)
Optic_list.append(mirror3)
Optic_list.append(mirror4)


Ray_list = []
ray_z = np.linspace(3.2485, 3.2485, 1)
ray_x = np.linspace(0, 0, 1)
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
