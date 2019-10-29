# -*- coding: utf-8 -*-

"""
Created on Sat Dec  8 23:47:35 2018

@author: samuel
"""

import numpy as np
import matplotlib.pyplot as plt
from general import wavelength_to_rgb, set_axes_equal
from general_optics import rotation_matrix_axis_angle
from optics3d import Ray, Grating
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
ax.set_aspect('equal')
for optic in Optic_list:
    optic.draw(ax, view="3d")
for ray in Ray_list:
    ray_history = ray.get_plot_repr()
    ax.plot(ray_history[:, 0], ray_history[:, 1], ray_history[:, 2], "-", color=wavelength_to_rgb(ray.wavelength, gamma=1.0))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
set_axes_equal(ax)
plt.show()
