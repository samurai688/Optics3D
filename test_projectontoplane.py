#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 22:49:09 2018

@author: samuel
"""

import pytest
import numpy as np
from general_optics import project_onto_plane

vectors_to_project = [
    np.array([1, 1, 1]),
    np.array([1, 0, 1]),
    np.array([1, -1, 1]),
]

answers = [
    np.array([1/3, -1/3, 1/3]),
    np.array([2/3, -2/3, 2/3]),
    np.array([1, -1, 1]),
]


@pytest.mark.parametrize("x", vectors_to_project)
def test_project_onto_xz_plane(x):
    normal_xz = np.array([0, 1, 0])
    assert all(np.isclose(project_onto_plane(x, normal_xz), np.array([1, 0, 1])))


@pytest.mark.parametrize("x,ans", tuple(zip(vectors_to_project, answers)))
def test_project_onto_arb_plane(x, ans):
    normal_arb = np.array([0.5, 1, 0.5])
    assert all(np.isclose(project_onto_plane(x, normal_arb), ans))


def test_project_the_norm():
    normal_arb = np.array([-1, 1, -1])
    assert all(np.isclose(project_onto_plane(normal_arb, normal_arb), np.array([0, 0, 0])))



