# -*- coding: utf-8 -*-

"""
Created on Sat Dec  8 23:47:35 2018

@author: samuel
"""

import numpy as np
from optics3d import Ray, Mirror, Annotation
import matplotlib.pyplot as plt
from general import set_axes_equal
plt.close("all")


max_ray_run_distance = 200 # mm

mirror1_pos = np.array([0, 100, 0])
mirror1_normal = np.array([0, -1, 0])
annotation1_pos = np.array([0, 120, 0])

ray1_pos = np.array([-100, 0, 0])
ray1_direction = np.array([1, 1, 0])
ray2_pos = np.array([-100, 0, 0])
ray2_direction = np.array([1, 0.8, 0])

Optic_list = []
Optic_list.append(Mirror(mirror1_pos, normal=mirror1_normal, shape="circular_flat", D=50))
Optic_list.append(Annotation(annotation1_pos, mirror1_normal, shape="circular_flat"))

Ray_list = []
Ray_list.append(Ray(ray1_pos, ray1_direction, wavelength=532, print_trajectory=False))
Ray_list.append(Ray(ray2_pos, ray2_direction, wavelength=532, print_trajectory=True))

for ray in Ray_list:
    ray.run(max_distance=max_ray_run_distance, optic_list=Optic_list)


# 2d plots
fig, axs = plt.subplots(2, 1)
# tangential, x-y plane:
for optic in Optic_list:
    optic.draw(axs[0], view="xy")
for ray in Ray_list:
    ray_history = ray.get_plot_repr()
    axs[0].plot(ray_history[:, 0], ray_history[:, 1], "-r")
# sagittal, x-z plane:
for optic in Optic_list:
    optic.draw(axs[1], view="xz")
for ray in Ray_list:
    ray_history = ray.get_plot_repr()
    axs[1].plot(ray_history[:, 0], ray_history[:, 2], "-r")
    
    
# 3d plots
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.set_aspect('equal')
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








