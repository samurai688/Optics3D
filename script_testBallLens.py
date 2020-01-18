#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 21:47:48 2019

@author: samuel
"""


import numpy as np
from optics3d import Ray, Lens
import matplotlib.pyplot as plt
plt.close("all")

lens_center = np.array([0, 100, 0])

lens1 = Lens(lens_center, shape="spherical", D=50, thinlens=False, index=1.5)
Optic_list = []
Optic_list.append(lens1)

max_ray_run_distance = 150

x_start = [0]
y_start = [0]
z_start = np.linspace(-15, 15, 7)

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
xlim = [0, 0]
ylim = [0, 0]
zlim = [0, 0]
for optic in Optic_list:
    optic.draw(ax, view="3d")
for ray in Ray_list:
    ray_history = ray.get_plot_repr()
    ax.plot(ray_history[:, 0], ray_history[:, 1], ray_history[:, 2], "-r")
    if np.min(ray_history[:, 0]) < xlim[0]:
        xlim[0] = np.min(ray_history[:, 0])
    if np.max(ray_history[:, 0]) > xlim[1]:
        xlim[1] = np.max(ray_history[:, 0])
    if np.min(ray_history[:, 1]) < ylim[0]:
        ylim[0] = np.min(ray_history[:, 1])
    if np.max(ray_history[:, 1]) > ylim[1]:
        ylim[1] = np.max(ray_history[:, 1])
    if np.min(ray_history[:, 2]) < zlim[0]:
        zlim[0] = np.min(ray_history[:, 2])
    if np.max(ray_history[:, 2]) > zlim[1]:
        zlim[1] = np.max(ray_history[:, 2])
# Create cubic bounding box to simulate equal aspect ratio
max_range = np.array([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (xlim[1] + xlim[0])
Yb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (ylim[1] + ylim[0])
Zb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (zlim[1] + zlim[0])
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
    ax.plot([xb], [yb], [zb], 'w')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()