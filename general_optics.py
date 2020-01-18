# -*- coding: utf-8 -*-
"""
@author: samuel
"""



import numpy as np
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
            # if we are at an interior node, do the specified boolean on the child nodes
            return opers[tree.getRootVal()](res1, res2, ray)
        else:
            # if we are at a leaf node, test intersect with that primitive:
            return tree.getRootVal().test_intersect(ray)


def helper_findIntersects(res, ray, mode='nearest'):
    if (res[1] is not None) and (res[2] is not None):
        distance1 = distance_between(ray.position, res[1])
        distance2 = distance_between(ray.position, res[2])
        if mode == 'nearest':
            if distance1 < distance2:
                point = res[1]
                normal = res[3]
                distance = distance1
            else:
                point = res[2]
                normal = res[4]
                distance = distance2
        elif mode == 'farthest':
            if distance1 > distance2:
                point = res[1]
                normal = res[3]
                distance = distance1
            else:
                point = res[2]
                normal = res[4]
                distance = distance2
        number_of_intersections = 2
    elif (res[1] is not None):
        distance1 = distance_between(ray.position, res[1])
        point = res[1]
        normal = res[3]
        distance = distance1
        number_of_intersections = 1
    elif (res[2] is not None):
        distance2 = distance_between(ray.position, res[1])
        point = res[2]
        normal = res[4]
        distance = distance2
        number_of_intersections = 1
    else:
        point = None
        normal = None
        distance = None
        number_of_intersections = 0
    return point, normal, distance, number_of_intersections


def operCsgUnion(resA, resB, ray):
    # res's are results of the primitive shapes test_intersect methods
    # have two inputs of syntax: intersected, int-pt1, int-pt2, norm1, norm2, intersection_count    (None if none)
    # e.g.  "False, None, None, None, None, 0"    if there's no intersection with that primitive
    # e.g. "True, int-pt1, int-pt2, norm1, norm2, 2"   if the ray intersects twice, goes in and out
    # UNION: min(tA_min, tB_min)
    # always add up the intersect counts
    intersected = False
    normalB = None
    normalA = None
    intersection_count_total = resA[5] + resB[5]

    # if the ray is hitting both primitives:
    if resA[0] and resB[0]:
        intersected = True

        # since resA is a hit, get tA_min
        pointA, normalA, distanceA, number_intersectionsA = helper_findIntersects(resA, ray, mode='nearest')

        # since resB is a hit, get tB_min
        pointB, normalB, distanceB, number_intersectionsB = helper_findIntersects(resB, ray, mode='nearest')

        # get int pt for UNION: min(tA_min, tB_min)
        if (pointA is not None) and (pointB is not None):
            if distanceA < distanceB:
                intersection_point = pointA
                normal = normalA
            else:
                intersection_point = pointB
                normal = normalB
        elif (pointA is not None):
            intersection_point = pointA
            normal = normalA
        else:
            intersection_point = None
            normal = None

        # we did it! return
        return intersected, intersection_point, None, normal, None, intersection_count_total

    # else, if the ray is only hitting primitive A:
    elif resA[0]:
        intersected = True
        pointA, normalA, distanceA, number_intersectionsA = helper_findIntersects(resA, ray, mode='nearest')
        return intersected, pointA, None, normalA, None, intersection_count_total

    # else, if the ray is only hitting primitive B:
    elif resB[0]:
        intersected = True
        pointB, normalB, distanceB, number_intersectionsB = helper_findIntersects(resB, ray, mode='nearest')
        return intersected, pointB, None, normalB, None, intersection_count_total

    # else, we missed everything
    else:
        intersected = False
        return intersected, None, None, None, None, intersection_count_total



def operCsgIntersect(resA, resB, ray):
    # magic internet pseudocode:
    #   "First time in A and B"
    #

    # um, not sure about the intersection count logic here, might be right might not be ## TODO
    intersection_count_total = resA[5] + resB[5]


    # if the ray is hitting both primitives:
    if resA[0] and resB[0]:
        intersected = True

        pointA_near, normalA_near, distanceA_near, number_intersectionsA = helper_findIntersects(resA, ray, mode='nearest')
        pointB_near, normalB_near, distanceB_near, number_intersectionsB = helper_findIntersects(resB, ray, mode='nearest')
        pointA_far, normalA_far, distanceA_far, number_intersectionsA = helper_findIntersects(resA, ray, mode='farthest')
        pointB_far, normalB_far, distanceB_far, number_intersectionsB = helper_findIntersects(resB, ray, mode='farthest')
        # take THAT for data

        # print(f"distanceA_near = {distanceA_near}")
        # print(f"distanceB_near = {distanceB_near}")
        # print(f"distanceA_far = {distanceA_far}")
        # print(f"distanceB_far = {distanceB_far}")

        # if both numbers are even, we are hitting both things from the outside
        if np.isclose(np.mod(number_intersectionsA, 2), 0) and np.isclose(np.mod(number_intersectionsB, 2), 0):
            # from the magic internet pseudocode:
            if ((distanceA_near < distanceB_near) and (distanceA_far > distanceB_near)):
                return intersected, pointB_near, None, normalB_near, None, intersection_count_total
            elif ((distanceB_near < distanceA_near) and (distanceB_far > distanceA_near)):
                return intersected, pointA_near, None, normalA_near, None, intersection_count_total
            else:
                # I don't think it's supposed to reach here, but whatever, lets just put something
                print("alert: something weird happening in operCsgIntersect")
                intersected = False
                return intersected, None, None, None, None, intersection_count_total
        # if A is even and B is odd, we are outside A and we are inside B
        elif np.isclose(np.mod(number_intersectionsA, 2), 0):
            # both the B distances (_near, _far) returned from our helper should be the same
            distanceB = distanceB_near
            # if the B intersection is nearer than both the A intersections, we exit B before we hit A, so no intersect
            if ((distanceB < distanceA_near) and (distanceB < distanceA_far)):
                intersected = False
                return intersected, None, None, None, None, intersection_count_total
            else:
                print("unhandled in operCsgIntersect")
                pass # TODO
        # if B is even and A is odd, we are outside B and we are inside A
        elif np.isclose(np.mod(number_intersectionsB, 2), 0):
            # both the A distances (_near, _far) returned from our helper should be the same
            distanceA = distanceA_near
            # if the A intersection is nearer than both the B intersections, we exit A before we hit B, so no intersect
            if ((distanceA < distanceB_near) and (distanceA < distanceB_far)):
                intersected = False
                return intersected, None, None, None, None, intersection_count_total
            else:
                print("unhandled in operCsgIntersect")
                pass # TODO
        # if A and B are both odd, we are inside both A and B
        else:
            # in this case, each one gives odd intersection counts, so it'll say we're inside, but we're not...
            # this is probably a sign of big trouble, but for now let me just put a plus one in here ## TODO
            intersection_count_total += 1
            # so we just need to take the nearest intersection
            # both the A and B distances (_near, _far) returned from our helper should be the same
            distanceA = distanceA_near
            distanceB = distanceB_near
            if distanceA < distanceB:
                intersected = True
                return intersected, pointA_near, None, normalA_near, None, intersection_count_total
            else:
                intersected = True
                return intersected, pointB_near, None, normalB_near, None, intersection_count_total


    # else, if the ray is only hitting primitive A, for intersect this is a miss:
    elif resA[0]:
        intersected = False
        return intersected, None, None, None, None, intersection_count_total

    # else, if the ray is only hitting primitive B, for intersect this is a miss:
    elif resB[0]:
        intersected = False
        return intersected, None, None, None, None, intersection_count_total

    # else, we missed everything
    else:
        intersected = False
        return intersected, None, None, None, None, intersection_count_total



def operCsgDifference(resA, resB, ray):
    # magic internet pseudocode:
    #   "First time in A not in B"
    #
    # the order is important for this one:  we are doing A - B
    #

    # um, not sure about the intersection count logic here, might be right might not be ## TODO
    intersection_count_total = resA[5] + resB[5]

    # if the ray is hitting both primitives:
    if resA[0] and resB[0]:
        intersected = True

        pointA_near, normalA_near, distanceA_near, number_intersectionsA = helper_findIntersects(resA, ray, mode='nearest')
        pointB_near, normalB_near, distanceB_near, number_intersectionsB = helper_findIntersects(resB, ray, mode='nearest')
        pointA_far, normalA_far, distanceA_far, number_intersectionsA = helper_findIntersects(resA, ray, mode='farthest')
        pointB_far, normalB_far, distanceB_far, number_intersectionsB = helper_findIntersects(resB, ray, mode='farthest')
        # take THAT for data

        # from the magic internet pseudocode:
        if distanceA_near < distanceB_near:
            return intersected, pointA_near, None, normalA_near, None, intersection_count_total
        elif distanceB_far < distanceA_far:
            return intersected, pointB_far, None, normalB_far, None, intersection_count_total
        else:
            # I don't know if it can even reach here, but whatever, lets just put something
            print("alert: something weird happening in operCsgDifference")
            intersected = False
            return intersected, None, None, None, None, intersection_count_total

    # else, if the ray is only hitting primitive A, for difference A - B this is a hit:
    elif resA[0]:
        intersected = True
        pointA, normalA = helper_findIntersects(resA, ray, mode='nearest')
        return intersected, pointA, None, normalA, None, intersection_count_total

    # else, if the ray is only hitting primitive B, for difference A - B this is a miss:
    elif resB[0]:
        intersected = False
        return intersected, None, None, None, None, intersection_count_total

    # else, we missed everything
    else:
        intersected = False
        return intersected, None, None, None, None, intersection_count_total
