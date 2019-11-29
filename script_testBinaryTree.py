# -*- coding: utf-8 -*-

"""
Created on 11/19/2019

@author: samuel
"""

from general_optics import BinaryTree, postOrderEval
from Shapes.shapes import Sphere
import numpy as np

cen1 = np.array([1, 1, 1])
cen2 = np.array([-3, -3, -3])
cen3 = np.array([3, -1, 3])
cen4 = np.array([1, 5, -1])

# result should be 15

r = BinaryTree('union')
r.insertLeft('difference')
r.insertRight(Sphere(center=cen1, D=1))
r.getLeftChild().insertLeft('intersect')
r.getLeftChild().insertRight(Sphere(center=cen2, D=1))
r.getLeftChild().getLeftChild().insertLeft(Sphere(center=cen3, D=1))
r.getLeftChild().getLeftChild().insertRight(Sphere(center=cen4, D=1))

a = postOrderEval(r)
print(a)

