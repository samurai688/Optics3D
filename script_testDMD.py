# -*- coding: utf-8 -*-

"""
Created on Sat Dec  8 23:47:35 2018

@author: samuel
"""

import numpy as np
from optics3d import Ray, Mirror, Lens, Detector
import matplotlib.pyplot as plt
from general import set_axes_equal, wavelength_to_rgb
from general_optics import image_rays
plt.close("all")


image_f_number = 8
object_size = 5  # mm
image_half_angle = np.arctan(1 / 2 / image_f_number)  # radians
lens1_f = 50
object_y = 0
lens1_y = 100  # 50
object_dist = lens1_y - object_y
image_dist = 1 / (1 / lens1_f - 1 / (object_dist))
image_y = lens1_y + image_dist
mag = image_dist / object_dist
image_size = object_size * mag
print(f"object_size = {object_size}")
print(f"object_dist = {object_dist}")
print(f"image_dist = {image_dist}")
print(f"image_y = {image_y}")
print(f"mag = {mag}")
print(f"image_size = {image_size}")


lens_center = np.array([0, lens1_y, 0])
lens_normal = np.array([0, -1, 0])
lens_tangent = np.array([0, 0, 1])
lens1 = Lens(lens_center, normal=lens_normal, shape="spherical_biconvex",
             tangent=lens_tangent, D=25, type="ideal", f=lens1_f)

mir_center = np.array([0, image_y, 0])
mir_normal = np.array([0, -1, 0])
dmd_angle_tan = np.tan(10.9 * np.pi / 180)
dmd_normal = np.array([dmd_angle_tan, -1, 0])
print(f"lens angle: {np.arctan(dmd_angle_tan) * 180 / np.pi}")
mir_tangent = np.array([0, 0, 1])
mirror1 = Mirror(mir_center, normal=mir_normal, shape="rectangular_flat", tangent=mir_tangent, h=50, w=50,
                 type="dmd", dmd_normal=dmd_normal)


lens2_f = 25
lens2_dist = 2 * lens2_f
lens2_angle_tan = np.tan(-21.8 * np.pi / 180)
lens2_angle_rad = np.arctan(lens2_angle_tan)
lens2_y = image_y - lens2_dist * np.cos(lens2_angle_rad)
lens2_x = lens2_dist * -np.sin(lens2_angle_rad)
lens_center = np.array([lens2_x, lens2_y, 0])
lens_normal = np.array([lens2_angle_tan, 1, 0])
lens_tangent = np.array([0, 0, 1])
print(f"lens angle: {np.arctan(lens2_angle_tan) * 180 / np.pi}")
lens2 = Lens(lens_center, normal=lens_normal, shape="spherical_biconvex",
             tangent=lens_tangent, D=25, type="ideal", f=lens2_f)

det_dist = 4 * lens2_f
det_angle_tan = np.tan(-43.6 * np.pi / 180)
det_angle_rad = np.arctan(det_angle_tan)
det_y = image_y - det_dist * np.cos(lens2_angle_rad)
det_x = det_dist * -np.sin(lens2_angle_rad)
detector_center = np.array([det_x, det_y, 0])
detector_normal = np.array([det_angle_tan, 1, 0])
print(f"detector angle: {np.arctan(det_angle_tan) * 180 / np.pi}")
detector_tangent = np.array([0, 0, 1])
detector_h = 30
detector_w = 30
detector1 = Detector(detector_center, normal=detector_normal, shape="rectangular_flat",
                     tangent=detector_tangent, h=detector_h, w=detector_w)

Optic_list = []
Optic_list.append(mirror1)
Optic_list.append(lens1)
Optic_list.append(lens2)
Optic_list.append(detector1)

max_ray_run_distance = 400



Ray_list = []
origins, dirs, waves = image_rays(0, size=object_size, angle=image_half_angle, x_res=5, z_res=5, angle_res=3)
for ix, origin in enumerate(origins):
    Ray_list.append(Ray(origins[ix], dirs[ix], wavelength=waves[ix], print_trajectory=False))

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
