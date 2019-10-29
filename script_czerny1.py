# -*- coding: utf-8 -*-

"""
Created on Mon Dec  10 01:32:18 2018

@author: samuel
"""

import numpy as np
from general import wavelength_to_rgb
from optics3d import Ray, Grating, Mirror, Detector
import matplotlib.pyplot as plt
from general import set_axes_equal
plt.close("all")


mirror1_center = np.array([0, 150, 0])
mirror1_normal = np.array([0.25, -1, 0])
mirror1_f = 150
mirror1_D = 50.8
mirror1 = Mirror(mirror1_center, mirror1_normal, shape="circular_concave_spherical",
                  f=mirror1_f, D=mirror1_D)

grating_center = np.array([55, 50, 0])
grating_normal = np.array([0, 1, 0])
grating_tangent = np.array([0, 0, 1])
grating_G = 300
grating1 = Grating(grating_center, grating_normal, shape="rectangular_flat",
                  tangent=grating_tangent, h=50, w=50, G=grating_G)

mirror2_center = np.array([120, 137.5, 0])
mirror2_normal = np.array([-0.35, -1, 0])
mirror2_f = 150
mirror2_D = 50.8
mirror2 = Mirror(mirror2_center, mirror2_normal, shape="circular_concave_spherical",
                  f=mirror2_f, D=mirror2_D)

detector_center = np.array([119, 0, 0])
detector_normal = np.array([0, 1, 0])
detector_tangent = np.array([0, 0, 1])
detector_h = 25
detector_w = 25
detector1 = Detector(detector_center, detector_normal, shape="rectangular_flat",
                     tangent=detector_tangent, h=detector_h, w=detector_w)




Optic_list = []
Optic_list.append(mirror1)
Optic_list.append(grating1)
Optic_list.append(mirror2)
Optic_list.append(detector1)

max_ray_run_distance = 550

ray_sweep_xnorm = np.linspace(-0.08, 0.08, 3)
ray_sweep_znorm = np.linspace(-0.09, 0.09, 3)
ray_sweep_z = np.linspace(-5, 5, 11)
ray_sweep_x = np.linspace(-0.03, 0.03, 2)
ray_sweep_wavelength = [350, 530, 600, 610, 700]

Ray_list = []
for xnorm in ray_sweep_xnorm:
    for znorm in ray_sweep_znorm:
        for z in ray_sweep_z:
            for x in ray_sweep_x:
                for wave in ray_sweep_wavelength:
                    Ray_list.append(Ray(np.array([x, 0, z]), np.array([xnorm, 1, znorm]), wavelength=wave, order=1, print_trajectory=False))


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



# detector plot
fig = plt.figure()
intensifier_diam = 25
theta_intensifier = np.linspace(0, 2 * np.pi, 361)
x_intensifier = intensifier_diam/2 * np.cos(theta_intensifier)
z_intensifier = intensifier_diam/2 * np.sin(theta_intensifier)
plt.plot(detector_center[0] + x_intensifier, detector_center[2] + z_intensifier, "-k")
for datarow in detector1.hit_data:
    plt.plot(datarow[0], datarow[2], 'o', markersize=2, color=wavelength_to_rgb(datarow[3]))
plt.xlim(detector_center[0] - detector_w/2, detector_center[0] + detector_w/2)
plt.ylim(detector_center[2] - detector_h/2, detector_center[2] + detector_h/2)
plt.xlabel("x (mm)")
plt.ylabel("z (mm)")

plt.show()
