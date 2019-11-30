# -*- coding: utf-8 -*-
"""
@author: samuel
"""



import numpy as np
import operator
from mpl_toolkits.mplot3d import art3d


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2, v_normal=None):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    if v_normal is not None: # signed
        angle = np.arctan2(np.dot((np.cross(v2_u, v1_u)), v_normal), np.dot(v1_u, v2_u))
    else: # unsigned
        angle = np.arccos(np.dot(v1_u, v2_u))
    return angle


def distance_between(p1, p2):
    """ Returns distance between two points """
    return np.linalg.norm(p1 - p2)


def rotation_matrix_axis_angle(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians, using the Euler-Rodrigues formula

    Note that in numpy you have to use the result (call it RM) like new_r = RM.dot(r),
    not new_r = RM * r like you might expect.
    """
    # https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    # Dec 9 2018
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


######
###### StackOverflow
######
# Dec 8 2018
# overload matplotlib to allow arbitrary normals of patches
# courtesy of https://stackoverflow.com/questions/18228966/how-can-matplotlib-2d-patches-be-transformed-to-3d-with-arbitrary-normals
# and the lovely person with the reply for negative angles
######
def rotation_matrix(v1, v2):
    """
    Calculates the rotation matrix that changes v1 into v2.
    """
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    cos_angle = np.dot(v1, v2)
    d = np.cross(v1, v2)
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        M = np.identity(3) if cos_angle > 0. else -np.identity(3)
    else:
        d /= sin_angle

        eye = np.eye(3)
        ddt = np.outer(d, d)
        skew = np.array([[    0,  d[2],  -d[1]],
                      [-d[2],     0,  d[0]],
                      [d[1], -d[0],    0]], dtype=np.float64)

        M = ddt + cos_angle * (eye - ddt) + sin_angle * skew

    return M

def pathpatch_2d_to_3d(pathpatch, z = 0, normal = 'z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str:  # Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1, 0, 0), index)

    path = pathpatch.get_path()  # Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path)  # Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D  # Change the class
    pathpatch._code3d = path.codes  # Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor  # Get the face color

    verts = path.vertices  # Get the vertices in 2D

    M = rotation_matrix(normal, (0, 0, 1))  # Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])

def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta
######
###### End stackoverflow
######


def project_onto_plane(x, n):
    d = np.dot(x, n) / np.linalg.norm(n)
    p = [d * unit_vector(n)[i] for i in range(len(n))]
    return np.array([x[i] - p[i] for i in range(len(x))])


class BinaryTree:
    def __init__(self,rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self,newNode):
        if self.leftChild == None:
            self.leftChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t

    def insertRight(self,newNode):
        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t

    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def setRootVal(self,obj):
        self.key = obj

    def getRootVal(self):
        return self.key


# only works if the tree is constructed how we expect
def postOrderEval(tree):
    opers = {'union': printCsgUnion, 'difference': printCsgDifference, 'intersect': printCsgIntersect}
    res1 = None
    res2 = None
    if tree:
        res1 = postOrderEval(tree.getLeftChild())
        res2 = postOrderEval(tree.getRightChild())
        if res1 and res2:
            return opers[tree.getRootVal()](res1, res2)
        else:
            return tree.getRootVal()

def printCsgUnion(res1, res2):
    return "( " + str(res1) + " u " + str(res2) + " )"

def printCsgIntersect(res1, res2):
    return "( " + str(res1) + " n " + str(res2) + " )"

def printCsgDifference(res1, res2):
    return "( " + str(res1)  + " \ " + str(res2) + " )"


def test_tree_intersect(tree, ray):
    opers = {'union': operCsgUnion, 'difference': operCsgDifference, 'intersect': operCsgIntersect}
    res1 = None
    res2 = None
    if tree:
        res1 = test_tree_intersect(tree.getLeftChild(), ray)
        res2 = test_tree_intersect(tree.getRightChild(), ray)
        if res1 and res2:
            return opers[tree.getRootVal()](res1, res2, ray)
        else:
            return tree.getRootVal().test_intersect(ray)


def operCsgUnion(resA, resB, ray):
    # have two inputs of syntax e.g.  "False, None, None, None, None"  or  "True, int-pt1, int-pt2, norm1, norm2"
    # UNION: min(tA_min, tB_min)
    intersected = False
    if resA[0] and resB[0]:
        # print("hi 1")
        intersected = True
        normB = None
        normA = None
        if (resA[1] is not None) and (resA[2] is not None):
            distanceA1 = distance_between(ray.position, resA[1])
            distanceA2 = distance_between(ray.position, resA[2])
            if distanceA1 < distanceA2:
                pointA = resA[1]
                normA = resA[3]
            else:
                pointA = resA[2]
                normA = resA[4]
        elif (resA[1] is not None):
            pointA = resA[1]
            normA = resA[3]
        else:
            pointA = None

        if (resB[1] is not None) and (resB[2] is not None):
            distanceB1 = distance_between(ray.position, resB[1])
            distanceB2 = distance_between(ray.position, resB[2])
            if distanceB1 < distanceB2:
                pointB = resB[1]
                normB = resB[3]
            else:
                pointB = resB[2]
                normB = resB[4]
        elif (resB[1] is not None):
            pointB = resB[1]
            normB = resB[3]
        else:
            pointB = None

        if (pointA is not None) and (pointB is not None):
            distanceA = distance_between(ray.position, pointA)
            distanceB = distance_between(ray.position, pointB)
            if distanceA < distanceB:
                intersection_point = pointA
                normal = normA
            else:
                intersection_point = pointB
                normal = normB
        elif (pointA is not None):
            intersection_point = pointA
            normal = normA
        else:
            intersection_point = None
            normal = None

        return intersected, intersection_point, None, normal, None

    elif resA[0]:
        # print("hi 2")
        if (resA[1] is not None) and (resA[2] is not None):
            distanceA1 = distance_between(ray.position, resA[1])
            distanceA2 = distance_between(ray.position, resA[2])
            if distanceA1 < distanceA2:
                pointA = resA[1]
                normA = resA[3]
            else:
                pointA = resA[2]
                normA = resA[4]
        elif (resA[1] is not None):
            pointA = resA[1]
            normA = resA[3]
        else:
            pointA = None
        # print(pointA)
        return True, pointA, None, normA, None

    elif resB[0]:
        if (resB[1] is not None) and (resB[2] is not None):
            distanceB1 = distance_between(ray.position, resB[1])
            distanceB2 = distance_between(ray.position, resB[2])
            if distanceB1 < distanceB2:
                pointB = resB[1]
                normB = resB[3]
            else:
                pointB = resB[2]
                normB = resB[4]
        elif (resB[1] is not None):
            pointB = resB[1]
            normB = resB[3]
        else:
            pointB = None
        # print(pointB)
        return True, pointB, None, normB, None

    else:
        # print("hi 4")
        return False, None, None, None, None



def operCsgIntersect(res1, res2):
    return "( " + str(tree1) + " n " + str(tree2) + " )"

def operCsgDifference(res1, res2):
    return "( " + str(tree1)  + " \ " + str(tree2) + " )"