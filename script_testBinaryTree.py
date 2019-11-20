# -*- coding: utf-8 -*-

"""
Created on 11/19/2019

@author: samuel
"""

from general_optics import BinaryTree, postOrderEval

# result should be 15

r = BinaryTree('+')
r.insertLeft('-')
r.insertRight('+')
r.getLeftChild().insertLeft(5)
r.getLeftChild().insertRight(1)
r.getRightChild().insertLeft(9)
r.getRightChild().insertRight(2)

a = postOrderEval(r)
print(a)

