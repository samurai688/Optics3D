


import numpy as np
import matplotlib.pyplot as plt
from general_optics import rotation_matrix, rotation_matrix2


# https://www.cl.cam.ac.uk/teaching/1999/AGraphHCI/SMAG/node2.html
# to transform a point, scale, then rotate, then translate
# to transform a "displacement", scale, then rotate    (translation doesn't matter for "direction" vectors)
# P = T R S P^
# D = R S D^


# https://cg.informatik.uni-freiburg.de/course_notes/graphics2_01_transformations.pdf
# homogeneous coordinates
# scale matrices
# rotation matrices
# translation matrices



# initial vars

point1 = np.array([100, 0, 30])
point1_4d = np.array([100, 0, 30, 1])

world_direction = np.array([0.0, 0.0, 1.0])
world_direction_4d = np.array([0.0, 0.0, 1.0, 0.0])

cyl1_radius = 30
cyl1_direction = np.array([1.0, 1.0, 1.0])
cyl1_direction /= np.linalg.norm(cyl1_direction)

cyl1_direction_4d = np.array([1.0, 1.0, 1.0, 0.0])
cyl1_direction_4d /= np.linalg.norm(cyl1_direction_4d)



# TRANSLATION

# this should be relatively easy

translation_vector = np.array([-10, 10, -10])
translation_matrix = np.eye(4)
translation_matrix[0:3,3] = translation_vector

result = np.dot(translation_matrix, point1_4d)

print(point1_4d)
print(translation_matrix)
print(result)

# sure, rockin

print("****")
print("****")
print("****")



# ROTATION
# this is the tricky one right
# we have a few methods for matrix rotation from the interwebs, let's try 'em out


M = rotation_matrix(cyl1_direction, world_direction)
M2 = rotation_matrix2(cyl1_direction, world_direction)

result = np.dot(cyl1_direction, M)
result2 = np.dot(M2, cyl1_direction)

print(cyl1_direction)
print(M)
print(result)
print(result2)

# seems like both methods are roughly equivalent and work for a variety of inputs
# first might be a tad more accurate numerically
# but the second has the minus signs where I like them so you can left-multiply with the matrix
# let's try 4d "homogeneous coordinates"

M2_4d = np.eye(4)
M2_4d[:3,:3] = M2
result2_4d = np.dot(M2_4d, cyl1_direction_4d)

print('4d:')
print(cyl1_direction_4d)
print(M2_4d)
print(result2_4d)

# sure, rotation seems to be doing it's thing

print("****")
print("****")
print("****")





# SCALING

# this should just be an identity matrix with whatever components, right

scaling_vector = np.array([0.1, 1, 10])
scaling_matrix = np.eye(4)
scaling_matrix[0, 0] = scaling_vector[0]
scaling_matrix[1, 1] = scaling_vector[1]
scaling_matrix[2, 2] = scaling_vector[2]

result = np.dot(scaling_matrix, point1_4d)

print(point1_4d)
print(scaling_matrix)
print(result)

# boom, okay we should be good to go


