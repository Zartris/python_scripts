import time
from typing import Tuple

import numpy as np
from numba import njit, prange
import cv2


# remember https://www.cs.columbia.edu/~allen/F17/NOTES/icckinematics.pdf
# https://medium.com/@nahmed3536/wheel-odometry-model-for-differential-drive-robotics-91b85a012299
@njit
def my_clip(a: np.ndarray, a_min: float, a_max: float) -> np.ndarray:
    # Clip an array to a specified min and max same as np.clip but numba compatible
    return np.maximum(np.minimum(a, a_max), a_min)


@njit
def my_clip_axis1(a: np.ndarray, a_min: float, a_max: float) -> np.ndarray:
    new_a = np.zeros_like(a)
    # Clip an array to a specified min and max same as np.clip but numba compatible
    new_a[:, 0] = np.maximum(np.minimum(a[:, 0], a_max), a_min)
    new_a[:, 1] = np.maximum(np.minimum(a[:, 1], a_max), a_min)
    return new_a


@njit(fastmath=True)
def norm_2d(vector):
    """
    np.linalg.norm(axis=1) but numba compatible
    """
    return np.sqrt(vector[:, 0] ** 2 + vector[:, 1] ** 2)


@njit
def distance(p1, p2):
    return np.linalg.norm(p1 - p2)


@njit
def distance_squared(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


@njit
def _differential_drive_update_state_2(current_state, linear_velocity, angular_velocity, dt):
    """
    Update the state of a differential drive robot using its current state, linear_velocity,
    angular_velocity, and the time step (dt).

    Y is the forward direction of the robot

    :param current_state: A 3x1 NumPy array representing the current state of the robot, with
                          [x_position, y_position, orientation].
    :param linear_velocity: A float representing the linear velocity of the robot (m/s).
    :param angular_velocity: A float representing the angular velocity of the robot (rad/s).
    :param dt: A float representing the time step used for state update.
    :return: A 3x1 NumPy array representing the updated state of the robot after the given time step.
    """
    orientation = current_state[-1, 0]
    A = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    B = np.array([
        [np.sin(orientation + np.pi / 2) * dt, 0],
        [np.cos(orientation + np.pi / 2) * dt, 0],
        [0, dt]
    ], dtype=np.float64)

    vel = np.array([
        [linear_velocity],
        [angular_velocity]
    ], dtype=np.float64)
    return np.dot(A, current_state) + np.dot(B, vel)


# This is a constant matrix that is used in the _differential_drive_update_state function.
A = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], dtype=np.float64)


@njit
def _differential_drive_update_state(current_state, linear_velocity, angular_velocity, dt):
    """
    Update the state of a differential drive robot using its current state, linear_velocity,
    angular_velocity, and the time step (dt).

    X is the forward direction of the robot

    :param current_state: A 3x1 NumPy array representing the current state of the robot, with
                          [x_position, y_position, orientation].
    :param linear_velocity: A float representing the linear velocity of the robot (m/s).
    :param angular_velocity: A float representing the angular velocity of the robot (rad/s).
    :param dt: A float representing the time step used for state update.
    :return: A 3x1 NumPy array representing the updated state of the robot after the given time step.
    """
    orientation = current_state[-1, 0]
    # For performance reasons this is moved out as a constant
    # A = np.array([
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]
    # ], dtype=np.float64)

    B = np.array([
        [np.cos(orientation) * dt, 0],
        [np.sin(orientation) * dt, 0],
        [0, dt]
    ], dtype=np.float64)

    vel = np.array([
        [linear_velocity],
        [angular_velocity]
    ], dtype=np.float64)
    return np.dot(A, current_state) + np.dot(B, vel)


@njit
def _differential_drive_inverse_kinematics(v: float, omega: float, wheel_radius: float, wheel_distance: float):
    """
    This is the inverse_kinematics of a differential drive robot
    Calculate the wheel angular velocities for a differential drive robot.

    Args:
    v: Linear velocity of the robot (m/s)
    omega: Angular velocity of the robot (rad/s)
    wheel_radius: Radius of the wheels (m)
    wheel_distance: Distance between the wheels (m)

    Returns:
    w_left: Left wheel angular velocity (rad/s)
    w_right: Right wheel angular velocity (rad/s)
    """
    # Calculate the linear velocity of each wheel
    v_right: float = v + (omega * wheel_distance) / 2
    v_left: float = v - (omega * wheel_distance) / 2
    # R = wheel_distance / 2.0 * (v_right + v_left) / (v_right - v_left)

    # Convert linear velocities to angular velocities
    w_right: float = v_right / wheel_radius
    w_left: float = v_left / wheel_radius

    return w_left, w_right


@njit
def _differential_drive_forward_kinematics(w_left: float, w_right: float, wheel_radius: float, wheel_distance: float) -> \
        Tuple[float, float]:
    """
    Calculate the linear and angular velocities of a differential drive robot.

    Args:
    w_right: Right wheel angular velocity (rad/s)
    w_left: Left wheel angular velocity (rad/s)
    wheel_radius: Radius of the wheels (m)
    wheel_distance: Distance between the wheels (m)

    Returns:
    v: Linear velocity of the robot (m/s)
    omega: Angular velocity of the robot (rad/s)
    """

    # Calculate the linear velocity of the left and right wheel.
    v_right = w_right * wheel_radius
    v_left = w_left * wheel_radius

    # Calculate the linear and angular velocity of the robot.
    v = (v_right + v_left) / 2.0
    omega = (v_right - v_left) / wheel_distance

    return v, omega


@njit(fastmath=True)
def linear_to_angular_velocity(linear_velocity, wheel_radius):
    """
    Convert linear velocity to angular velocity
    :param linear_velocity: linear velocity in m/s
    :param wheel_radius: radius of the wheel in meters
    :return: angular velocity in rad/s
    """
    # convert linear velocity to angular velocity
    angular_velocity = linear_velocity / wheel_radius
    return angular_velocity


@njit(cache=True, fastmath=True)
def get_orientation_difference(current_orientation, target_orientation):
    """
    Calculate the shortest angular difference between two orientation angles.

    Given the robot's current orientation (a) and a target orientation (b),
    this function returns the amount the robot needs to rotate (positive or negative)
    to align with the target orientation.

    A positive value indicates a counterclockwise (left) rotation,
    while a negative value indicates a clockwise (right) rotation.

    The function is optimized using Numba JIT compilation for improved performance.

    Parameters
    ----------
    current_orientation : float
        The robot's current orientation angle (in radians).
    target_orientation : float
        The target orientation angle (in radians).

    Returns
    -------
    float
        The shortest angular difference (in radians) between the two orientation angles.
        Positive values indicate counterclockwise (left) rotations,
        while negative values indicate clockwise (right) rotations.

    Example
    -------
    >>> current_orientation = np.pi / 2  # 90 degrees
    >>> target_orientation = np.pi       # 180 degrees
    >>> rotation_amount = get_orientation_difference(current_orientation, target_orientation)
    >>> rotation_amount
    -1.5707963267948966  # Approximately -90 degrees (clockwise rotation)
    """
    diff = (current_orientation - target_orientation) % (2 * np.pi)
    if diff > np.pi:
        diff -= 2 * np.pi
    return diff


@njit(cache=True)
def get_orientation_between_points(looking_from_point, looking_at_point):
    """
    Calculate the orientation angle between two points in a 2D Cartesian coordinate system.

    Given a point 'looking_from_point' and a target point 'looking_at_point',
    this function returns the orientation angle (in radians) required to face
    from 'looking_from_point' towards 'looking_at_point'.

    The function is optimized using Numba JIT compilation for improved performance.

    Parameters
    ----------
    looking_from_point : array_like
        The point from which we are looking, specified as a 2-element array-like object (x, y).
    looking_at_point : array_like
        The point we are looking at, specified as a 2-element array-like object (x, y).

    Returns
    -------
    float
        The orientation angle (in radians) between the two points.

    Example
    -------
    >>> looking_from_point = np.array([0, 0])  # Origin
    >>> looking_at_point = np.array([1, 1])   # Point (1, 1)
    >>> orientation_angle = get_orientation_between_points(looking_from_point, looking_at_point)
    >>> orientation_angle
    0.7853981633974483  # Approximately 45 degrees

    Notes
    -----
    The orientation angle is computed using the arctangent function (np.arctan2),
    which returns a value in the range [-pi, pi]. A positive angle represents
    a counterclockwise rotation from the positive x-axis, while a negative angle
    represents a clockwise rotation.
    """
    return np.arctan2(looking_at_point[1] - looking_from_point[1], looking_at_point[0] - looking_from_point[0])


@njit
def check_collision(list_of_agents_pos, list_of_radius):
    collisions = np.zeros((len(list_of_agents_pos),))
    for i, pos in enumerate(list_of_agents_pos[:-1]):
        # Distance to all other agents
        agents_pos = list_of_agents_pos[i + 1:]
        radii = list_of_radius[i + 1:]
        deltas = np.abs(agents_pos - pos)
        dists = deltas[:, 0] ** 2 + deltas[:, 1] ** 2
        (idxs,) = np.where(dists <= (radii + list_of_radius[i]) ** 2)
        if len(idxs) > 0:
            collisions[i] = 1
            collisions[i + 1 + idxs] = 1
    return collisions


@njit
def cast_ray(grid, origin, angle, max_distance):
    x0, y0 = origin
    dx, dy = np.cos(angle), np.sin(angle)
    t = 0
    visible_space = []
    while t < max_distance:
        x, y = x0 + t * dx, y0 + t * dy
        i, j = int(np.rint(x)), int(np.rint(y))

        if grid[j, i] == 1:  # Check if the current cell is a wall
            return np.ceil(t), visible_space
        visible_space.append((i, j))
        t += 0.1

    return max_distance, visible_space


@njit
def compute_visibility(grid, origin, fov, num_rays, max_distance):
    h, w = grid.shape
    visibility = np.ones((h, w), dtype=np.uint8)
    start_angle, end_angle = -fov / 2, fov / 2
    angles = np.linspace(start_angle, end_angle, num_rays)

    for angle in angles:
        distance, visible_space = cast_ray(grid, origin, angle, max_distance)
        for i, j in visible_space:
            visibility[j, i] = 0

    return visibility


def visualize(grid, origin, use_visibility, fov, num_rays, max_distance):
    h, w = grid.shape
    img = np.zeros((h * 50, w * 50, 3), dtype=np.uint8)

    # Draw the grid
    for y in range(h):
        for x in range(w):
            if grid[y, x] == 1:
                img[y * 50:(y + 1) * 50, x * 50:(x + 1) * 50] = (255, 255, 255)

    # Draw the robot
    cv2.circle(img, (origin[0] * 50 + 25, origin[1] * 50 + 25), 10, (0, 255, 0), -1)

    # Compute visibility if necessary
    if use_visibility:
        visibility = compute_visibility(grid, origin, fov, num_rays, max_distance)
    else:
        visibility = None

    # Draw the rays
    start_angle, end_angle = -fov / 2, fov / 2
    angles = np.linspace(start_angle, end_angle, num_rays)

    for angle in angles:
        if visibility is not None:
            distance, visible_space = cast_ray(grid, origin, angle, max_distance)
        else:
            distance = max_distance
        x, y = origin[0] + distance * np.cos(angle), origin[1] + distance * np.sin(angle)
        x1, y1 = origin[0] * 50 + 25, origin[1] * 50 + 25
        x2, y2 = int(x * 50 + 25), int(y * 50 + 25)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Show the visualization
    cv2.imshow('Grid and Rays', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    grid = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=np.uint8)

    origin = (1, 1)
    fov = 2 * np.pi
    num_rays = 360
    max_distance = 10
    _ = compute_visibility(grid, origin, fov, num_rays, max_distance)

    start = time.perf_counter()
    visibility = compute_visibility(grid, origin, fov, num_rays, max_distance)
    print("One run took", time.perf_counter() - start, "seconds")
    print(visibility)
    start = time.perf_counter()
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 0:
                # visualize(grid, (x, y), True, fov, num_rays, max_distance)
                visibility = compute_visibility(grid, (x, y), fov, num_rays, max_distance)
                # print(visibility)
    print("all took", time.perf_counter() - start, "seconds")
