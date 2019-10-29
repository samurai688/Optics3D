# -*- coding: utf-8 -*-

"""
Created on Sat Dec  8 23:47:35 2018

@author: samuel
"""

import pytest
import numpy as np
from optics3d import Ray, Sphere, intersectSphere

two_hit_rays = [
    Ray(np.array([0, 0, 0]), np.array([0, 1, 0]), wavelength=532, print_trajectory=False),
]
one_hit_rays = [
    Ray(np.array([0, 10, 0]), np.array([0, 1, 0]), wavelength=532, print_trajectory=False),
    Ray(np.array([0, 10, 0]), np.array([0, -1, 0]), wavelength=532, print_trajectory=False),
    Ray(np.array([0, 10, 0]), np.array([0, 0, 1]), wavelength=532, print_trajectory=False),
    Ray(np.array([0, 10, 0]), np.array([0, 0, -1]), wavelength=532, print_trajectory=False),
    Ray(np.array([1, 0, 0]), np.array([0, 1, 0]), wavelength=532, print_trajectory=False),
]
miss_rays = [
    Ray(np.array([0, 0, 0]), np.array([0, -1, 0]), wavelength=532, print_trajectory=False),
    Ray(np.array([2, 0, 0]), np.array([0, 1, 0]), wavelength=532, print_trajectory=False),
]


@pytest.fixture
def my_sphere():
    return Sphere(np.array([0, 10, 0]), R=1)


@pytest.mark.parametrize("ray", two_hit_rays)
def test_two_hit_rays(my_sphere, ray):
    assert intersectSphere(ray, my_sphere)[0] is True
    assert intersectSphere(ray, my_sphere)[2] is not None


@pytest.mark.parametrize("ray", one_hit_rays)
def test_one_hit_rays(my_sphere, ray):
    assert intersectSphere(ray, my_sphere)[0] is True
    assert intersectSphere(ray, my_sphere)[2] is None


@pytest.mark.parametrize("ray", miss_rays)
def test_miss_rays(my_sphere, ray):
    assert intersectSphere(ray, my_sphere)[0] is False






