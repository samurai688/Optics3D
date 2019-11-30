# -*- coding: utf-8 -*-

"""
Created on Sat Dec  8 23:47:35 2018

@author: samuel
"""

import pytest
import numpy as np
from optics3d import Ray
from Shapes.shapes import Rectangle



hit_rays = [
    Ray(np.array([0, 0, 0]), np.array([0, 1, 0]), wavelength=532, print_trajectory=False),
    Ray(np.array([0, 0, 0.499]), np.array([0, 1, 0]), wavelength=532, print_trajectory=False),
]

miss_rays = [
    Ray(np.array([0, 0, 0.501]), np.array([0, 1, 0]), wavelength=532, print_trajectory=False),
]


@pytest.fixture
def my_rectangle():
    rect_center = np.array([0, 5, 0])
    rect_normal = np.array([0, -1, 0])
    rect_tangent = np.array([0, 0, 1])
    return Rectangle(rect_center, rect_normal, rect_tangent, h=1, w=1)


@pytest.mark.parametrize("ray", hit_rays)
def test_hit_rays(my_rectangle, ray):
    assert my_rectangle.test_intersect(ray)[0] is True


@pytest.mark.parametrize("ray", miss_rays)
def test_miss_rays(my_rectangle, ray):
    assert my_rectangle.test_intersect(ray)[0] is False


