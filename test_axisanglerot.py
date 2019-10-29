# -*- coding: utf-8 -*-

"""
Created on Sat Dec  8 23:47:35 2018

@author: samuel
"""

import pytest
import numpy as np
from optics3d import rotation_matrix_axis_angle

r_in_list = [
    np.array([0, 1, 0]),
    np.array([0, 1, 0]),
    np.array([0, 1, 0]),
    np.array([0, 1, 0]),
]
theta_list = [
    np.pi/2,
    -np.pi/2,
    np.pi/4,
    -np.pi/4,
]
expected_list = [
    np.array([0, 0, 1]),
    -np.array([0, 0, 1]),
    np.array([0,  np.sin(np.pi/4),  np.sin(np.pi/4)]),
    np.array([0,  np.sin(np.pi/4), -np.sin(np.pi/4)]),
]


@pytest.mark.parametrize("r_in,theta,expected", tuple(zip(r_in_list, theta_list, expected_list)))
def test_axis_angle_rot(r_in, theta, expected):
    r = r_in
    axis = np.array([1, 0, 0])
    RM = rotation_matrix_axis_angle(axis, theta)
    r_out = RM.dot(r)
    # print(r)
    # print(RM)
    # print(r_out)
    assert all(np.isclose(r_out,  expected))