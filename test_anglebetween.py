# -*- coding: utf-8 -*-

"""
Created on Sat Dec  8 23:47:35 2018

@author: samuel
"""

import pytest
import numpy as np
from optics3d import angle_between


v1_list = [
    np.array([1, 1, 0]),
    np.array([1, 1, 0]),
]
v2_list = [
    np.array([1, 0, 0]),
    np.array([1, 0, 0]),
]
v_normal_list = [
    np.array([0, 0, 1]),
    np.array([0, 0, -1]),
]
expected_list = [
    45 * np.pi / 180,
    -45 * np.pi / 180,
]


@pytest.mark.parametrize("v1,v2,v_normal,expected", tuple(zip(v1_list, v2_list, v_normal_list, expected_list)))
def test_angle_between(v1, v2, v_normal, expected):
    assert np.isclose(angle_between(v1, v2, v_normal=v_normal), expected)