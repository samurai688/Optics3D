# -*- coding: utf-8 -*-

"""
Created on Sat Dec  8 23:47:35 2018

@author: samuel
"""

import numpy as np
from general import wavelength_to_rgb
from optics3d import Ray, Grating, rotation_matrix_axis_angle
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.close("all")


grating_center = np.array([0, 100, 0])
grating_normal = np.array([0.0, -1, 0])
grating_tangent = np.array([0, 0, 1])


G = 600
littrow_angle_700nm = np.arcsin(G / 1e6 * 1 * 700 / 2)
# rotate to Littrow angle for 700 nm, m=1, G=600: 0.2115749597580956 radians
RM = rotation_matrix_axis_angle(grating_tangent, littrow_angle_700nm)
grating_normal = RM.dot(grating_normal)

grating1 = Grating(grating_center, grating_normal, shape="rectangular_flat",
                  tangent=grating_tangent, h=50, w=50, G=G)


Optic_list = []
Optic_list.append(grating1)

max_ray_run_distance = 300

ray1 = Ray(np.array([0, 0, 0]), np.array([0, 1, 0]), wavelength=400, order=1, print_trajectory=False)
ray2 = Ray(np.array([0, 0, 0]), np.array([0, 1, 0]), wavelength=450, order=1, print_trajectory=False)
ray3 = Ray(np.array([0, 0, 0]), np.array([0, 1, 0]), wavelength=500, order=1, print_trajectory=False)
ray4 = Ray(np.array([0, 0, 0]), np.array([0, 1, 0]), wavelength=550, order=1, print_trajectory=False)
ray5 = Ray(np.array([0, 0, 0]), np.array([0, 1, 0]), wavelength=600, order=1, print_trajectory=False)
ray6 = Ray(np.array([0, 0, 0]), np.array([0, 1, 0]), wavelength=650, order=1, print_trajectory=False)
ray7 = Ray(np.array([0, 0, 0]), np.array([0, 1, 0]), wavelength=700, order=1, print_trajectory=False)
ray8 = Ray(np.array([0, 0, 0]), np.array([0, 1, 0]), wavelength=400, order=-1, print_trajectory=False)
ray9 = Ray(np.array([0, 0, 0]), np.array([0, 1, 0]), wavelength=450, order=-1, print_trajectory=False)
ray10 = Ray(np.array([0, 0, 0]), np.array([0, 1, 0]), wavelength=500, order=-1, print_trajectory=False)
ray11 = Ray(np.array([0, 0, 0]), np.array([0, 1, 0]), wavelength=550, order=-1, print_trajectory=False)
ray12 = Ray(np.array([0, 0, 0]), np.array([0, 1, 0]), wavelength=600, order=-1, print_trajectory=False)
ray13 = Ray(np.array([0, 0, 0]), np.array([0, 1, 0]), wavelength=650, order=-1, print_trajectory=False)
ray14 = Ray(np.array([0, 0, 0]), np.array([0, 1, 0]), wavelength=700, order=-1, print_trajectory=False)
ray15 = Ray(np.array([0, 0, 0]), np.array([0, 1, 0]), wavelength=700, order=0, print_trajectory=False)
Ray_list = []
Ray_list.append(ray1)
Ray_list.append(ray2)
Ray_list.append(ray3)
Ray_list.append(ray4)
Ray_list.append(ray5)
Ray_list.append(ray6)
Ray_list.append(ray7)
Ray_list.append(ray8)
Ray_list.append(ray9)
Ray_list.append(ray10)
Ray_list.append(ray11)
Ray_list.append(ray12)
Ray_list.append(ray13)
Ray_list.append(ray14)
Ray_list.append(ray15)


for ray in Ray_list:
    ray.run(max_distance=max_ray_run_distance, optic_list=Optic_list)


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
    ax.plot(ray_history[:, 0], ray_history[:, 1], ray_history[:, 2], "-", color=wavelength_to_rgb(ray.wavelength, gamma=1.0))
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
