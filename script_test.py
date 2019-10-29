# -*- coding: utf-8 -*-

"""
Created on Sat Dec  8 23:47:35 2018

@author: samuel
"""

import numpy as np
from optics3d import Ray, Mirror, Disc, intersectDisc
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.close("all")


max_ray_run_distance = 200 # mm

mirror1_pos = np.array([0, 100, 0])
mirror1_normal = np.array([0, -1, 0])

ray1_pos = np.array([-100, 0, 0])
ray1_direction = np.array([1, 1, 0])
ray2_pos = np.array([-100, 0, 0])
ray2_direction = np.array([1, 0.8, 0])

Optic_list = []
Optic_list.append(Mirror(mirror1_pos, mirror1_normal, shape="circular_flat", D=50))

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
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xlim[1] + xlim[0])
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ylim[1] + ylim[0])
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zlim[1] + zlim[0])
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
   ax.plot([xb], [yb], [zb], 'w')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()








