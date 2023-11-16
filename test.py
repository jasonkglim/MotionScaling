import numpy as np
import math

def transform_2d_coordinate(point, new_x_axis, new_origin, old_y_axis=[0, -1]):
    # Convert inputs to NumPy arrays for easy computation
    point = np.array(point)
    new_x_axis = np.array(new_x_axis)
    new_origin = np.array(new_origin)
    old_y_axis = np.array(old_y_axis)

    # Compute the transformation matrix
    old_origin = point - new_origin
    old_x_axis = np.array([1, 0])  # Assuming the old x-axis is the standard unit vector
    rotation_angle = np.arctan2(new_x_axis[1], new_x_axis[0])
    rotation_matrix = np.array([[np.cos(rotation_angle), np.sin(rotation_angle)],
                                [-np.sin(rotation_angle), np.cos(rotation_angle)]])

    # Apply the transformation to the point
    new_point = np.dot(rotation_matrix, old_origin)

    return new_point

# point_to_transform = [0, 0]  # Example point
# new_x_axis_vector = [0, 1]  # Example new x-axis vector
# new_origin_point = [0, 1]  # Example new origin point

# transformed_point = transform_2d_coordinate(point_to_transform, new_x_axis_vector, new_origin_point)
# print("Transformed point:", transformed_point)

# v = [0,-1]
# print(np.degrees(np.arctan2(v[1], v[0])))
end_points = np.array([[0, 1], [3, 0], [-1, 0], [0, -1]])
mean_end_point = np.mean(end_points, axis=0)
deviations = np.linalg.norm(end_points - mean_end_point, axis=1)
SD = np.sqrt(np.sum(deviations**2)/4)
print(SD)
print(np.linalg.norm(np.std(end_points, axis=0)))



