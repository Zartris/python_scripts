import time

import cv2
import numpy as np
from numba import njit
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from agv import AAGV
from utils import _differential_drive_inverse_kinematics, _differential_drive_forward_kinematics, \
    my_clip, get_orientation_difference, get_orientation_between_points, _differential_drive_update_state, \
    norm_2d

debug = False


@njit(nogil=True)
def predict_trajectory(robot_position: np.ndarray, robot_orientation: np.ndarray,
                       controls: np.ndarray, dt: float, wheel_radius: float,
                       wheel_distance: float):
    """
    Predict the trajectory of a differential drive robot given its initial state and a sequence of control inputs.

    Args:
        robot_position (np.ndarray): The initial 2D position of the robot (x, y).
        robot_orientation (np.ndarray): The initial orientation of the robot (theta) as a 1D array.
        current_wheel_vel (np.ndarray): The initial left and right wheel angular velocities.
        controls (np.ndarray): An array of control inputs, each consisting angular velocity of left and right wheel (rad/s).
        dt (float): The time step between control inputs in seconds.
        wheel_radius (float): The radius of the wheels in meters.
        wheel_distance (float): The distance between the wheels in meters.
        max_angluar_velocity (float): The maximum allowable angular velocity for the wheels in rad/s.
        max_acceleration (float): The maximum allowable linear acceleration for the wheels in m/s^2.

    Returns:
        predicted_trajectory (np.ndarray): An array of predicted 2D robot positions for each time step.
    """
    horizon = len(controls)
    robot_pos = robot_position.astype(np.float64)  # Cast robot_position to float64
    robot_orientation = robot_orientation.astype(np.float64)  # Cast robot_orientation to float64

    predicted_trajectory = np.zeros((horizon, 3))
    robot_state = np.array([
        [robot_pos[0]],
        [robot_pos[1]],
        [robot_orientation[0]]])
    for t, (w_left, w_right) in enumerate(controls):
        v, w = _differential_drive_forward_kinematics(w_left, w_right, wheel_radius, wheel_distance)
        robot_state = _differential_drive_update_state(robot_state, v, w, dt)
        predicted_trajectory[t] = np.array([robot_state[0, 0], robot_state[1, 0], robot_state[2, 0]])

    return predicted_trajectory


# @njit
# def find_closest_point(point, points):
#     """Finds the closest point to a given point from a list of points"""
#     min_distance = np.inf
#     closest_point = None
#     for p in points:
#         distance = np.sqrt(np.sum((point - p) ** 2))
#         if distance < min_distance:
#             min_distance = distance
#             closest_point = p
#     return closest_point, min_distance


@njit(nogil=True, fastmath=True)
# @profile
def _cost_function(controls, robot_position, robot_orientation, path, dt, horizon, wheel_radius,
                   wheel_distance, R, Q, Rd):
    """
    Calculate the cost of a sequence of control inputs for a differential drive robot.

    Args:
        controls (np.ndarray): An array of control inputs, each consisting of angular velocity for left and right wheel.
        robot_position (np.ndarray): The initial 2D position of the robot (x, y).
        robot_orientation (np.ndarray): The initial orientation of the robot (theta) as a 1D array.
        current_wheel_vel (np.ndarray): The initial left and right wheel angular velocities.
        path (np.ndarray): An array of the desired 2D robot positions for each time step.
        dt (float): The time step between control inputs in seconds.
        horizon (int): The number of control inputs to consider.
        wheel_radius (float): The radius of the wheels in meters.
        wheel_distance (float): The distance between the wheels in meters.
        max_angluar_velocity (float): The maximum allowable angular velocity for the wheels in rad/s.
        max_acceleration (float): The maximum allowable linear acceleration for the wheels in m/s^2.
        weights (tuple): A tuple containing the weights for different cost components:
                         (R: weight for control inputs, Q: weight for position error, Rd: weight for change in controls)

    Returns:
        cost (float): The total cost of the given sequence of control inputs.
    """
    controls = controls.reshape((horizon, 2))  # The controls in terms of linear velocity and angular velocity
    R = R  # control_cost
    Q = np.ascontiguousarray(Q[:2, :2])  # distance_cost
    Rd = Rd  # control_change_cost
    trajectory = predict_trajectory(robot_position, robot_orientation, controls, dt,
                                    wheel_radius, wheel_distance)

    # Subtract the path array from the reshaped trajectory array
    diffs = trajectory - path
    sqr_diffs = diffs ** 2
    sqr_diffs = np.ascontiguousarray(sqr_diffs)

    distance_cost = np.zeros(horizon, dtype=np.float64)
    # heading_cost = np.zeros(horizon, dtype=np.float64)
    # control_cost = np.zeros(horizon, dtype=np.float64)
    # control_change_cost = np.zeros(horizon - 1, dtype=np.float64)
    for i in range(horizon):
        distance_cost[i] = np.sum(np.dot(Q, sqr_diffs[i, :2]))
        # diff_orientation = get_orientation_difference(trajectory[i, -1],
        #                                               get_orientation_between_points(trajectory[i, :2],
        #                                                                              path[i, :2])) ** 2
        #
        # heading_cost[i] = Q[-1, -1] * diff_orientation * ((horizon - i) / horizon)  # prioritize the first heading error

        # control_cost[i] = np.sum(np.dot(R, controls[i] ** 2))
        # if i < horizon - 1:
        #     control_change_cost[i] = np.sum(np.dot(Rd, (controls[i + 1] - controls[i]) ** 2))
    # for debugging:
    dist_cost = np.sum(distance_cost) * 1
    # head_cost = np.sum(heading_cost) * 1
    # cont_cost = np.sum(control_cost) * 1
    # cont_change_cost = np.sum(control_change_cost) * 1
    # total_cost = dist_cost + head_cost + cont_cost + cont_change_cost
    # if debug:
    #     visualize_robot_and_target(trajectory[0], path[0, :2])
    return dist_cost


@njit(fastmath=True)
def _cost_function_quick(controls, robot_position, robot_orientation, path, dt, horizon, wheel_radius,
                         wheel_distance, weights):
    controls = controls.reshape((horizon, 2))
    R = weights[0]
    Q = weights[1]
    Rd = weights[2]

    trajectory = predict_trajectory(robot_position, robot_orientation, controls, dt,
                                    wheel_radius, wheel_distance)

    diffs = trajectory - path
    sqr_diffs = diffs[:, :2] ** 2
    distance_cost = np.sum(np.dot(sqr_diffs[:, :2], Q[:2, :2].T), axis=1)

    diff_orientations = np.array([
        get_orientation_difference(trajectory[i, -1],
                                   get_orientation_between_points(trajectory[i, :2], path[i, :2])) ** 2
        for i in range(horizon)
    ])

    heading_cost = Q[-1, -1] * diff_orientations * np.arange(horizon, 0, -1) / horizon

    control_cost = np.sum(np.dot(controls, R.T) ** 2, axis=1)

    control_change_cost = np.sum(np.dot((controls[:-1] - controls[1:]), Rd.T) ** 2, axis=1)

    total_cost = (
            np.sum(distance_cost) +
            np.sum(heading_cost) +
            np.sum(control_cost) +
            np.sum(control_change_cost)
    )

    return total_cost


def cost_function(controls, robot_position, robot_orientation, current_wheel_vel, path, dt, horizon, wheel_radius,
                  wheel_distance, max_angluar_velocity, max_acceleration, R, Q, Rd):
    """
    Wrapper for the numba implementation of the cost function for a differential drive robot.

    Args:
        Same as the _cost_function (see its docstring for details).

    Returns:
        cost (float): The total cost of the given sequence of control inputs.
    """
    return float(
        _cost_function(controls, robot_position, robot_orientation, path, dt, horizon, wheel_radius,
                       wheel_distance, R, Q, Rd))


def mpc_controller(path, robot_position, robot_orientation, dt, horizon, wheel_radius, wheel_distance, w_left, w_right,
                   max_angular_velocity, max_accel, R, Q, Rd):
    """
    Model Predictive Control (MPC) for a differential drive robot.

    Args:
        path (np.ndarray): An array of the desired 2D robot positions for each time step.
        robot_position (np.ndarray): The current 2D position of the robot (x, y).
        robot_orientation (np.ndarray): The current orientation of the robot (theta) as a 1D array.
        dt (float): The time step between control inputs in seconds.
        horizon (int): The number of control inputs to consider.
        wheel_radius (float): The radius of the wheels in meters.
        wheel_distance (float): The distance between the wheels in meters.
        w_left (float): The current left wheel angular velocity in rad/s.
        w_right (float): The current right wheel angular velocity in rad/s.
        max_angluar_velocity (float): The maximum allowable angular velocity for the wheels in rad/s.
        max_accel (float): The maximum allowable linear acceleration for the wheels in m/s^2.
        weights (tuple): A tuple containing the weights for different cost components:
                         (R: weight for control inputs, Q: weight for position error, Rd: weight for change in controls)

    Returns:
        optimal_controls (np.ndarray): An array of the optimal control inputs, each consisting of linear velocity (v)
                                       and angular velocity (w).
        cost (float): The total cost of the optimal sequence of control inputs.
    """
    horizon = min(len(path), horizon)

    bounds = [(-max_angular_velocity, max_angular_velocity),  # left wheel angular velocity
              (-max_angular_velocity, max_angular_velocity)  # right wheel angular velocity
              ] * horizon
    # Initialize the controls is w_left, w_right in radians per second
    initial_controls = np.array([(w_left), (w_right)] * horizon)

    result = minimize(cost_function,
                      x0=initial_controls,
                      args=(
                          robot_position, robot_orientation, np.array([w_left, w_right]), path, dt, horizon,
                          wheel_radius,
                          wheel_distance, max_angular_velocity, max_accel, R, Q, Rd),
                      bounds=bounds,
                      method='SLSQP',
                      # options={'maxiter': 1000}
                      )

    optimal_controls = result.x.reshape((horizon, 2))
    # Return the first control input of the optimal sequence
    return optimal_controls, result.fun


@njit(fastmath=True)
def update_path(path, robot_position, threshold_distance=0.25):
    # distances = frobenius_norm_axis1(path[:, :2] - robot_position[:2, 0])
    diff = path[:, :2] - robot_position[:2, 0]
    distances = norm_2d(diff)
    # distances = np.sum(diff ** 2, axis=1)
    indx = np.argwhere(distances <= threshold_distance)
    if len(indx) > 0:
        path = path[indx[-1, 0] + 1:]
    return path


def generate_trajectory(path, distance_between_points=0.1):
    # Calculate the length of the path
    length = len(path)

    # Check if the path has at least 2 points
    if length < 2:
        raise ValueError("Path must have at least 2 points")

    # Calculate the distances between consecutive points in the path
    distances = np.linalg.norm(np.diff(path, axis=0), axis=1)

    # Calculate the cumulative distances along the path
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    # Create a cubic spline interpolator for x and y coordinates
    x_spline = CubicSpline(cumulative_distances, path[:, 0])
    y_spline = CubicSpline(cumulative_distances, path[:, 1])

    # Calculate the total distance of the path
    total_distance = cumulative_distances[-1]

    # Generate the interpolated distances based on the specified distance between points
    interpolated_distances = np.arange(0, total_distance, distance_between_points)

    # Evaluate the spline interpolator at the interpolated distances
    x_trajectory = x_spline(interpolated_distances)
    y_trajectory = y_spline(interpolated_distances)

    # Combine the x and y coordinates into the final trajectory
    trajectory = np.column_stack((x_trajectory, y_trajectory))
    orientations = calculate_orientations(trajectory)
    trajectory = np.column_stack((trajectory, orientations))
    return trajectory


@njit(cache=True)
def calculate_orientations(coords):
    n = coords.shape[0]
    orientations = np.zeros(n, dtype=np.float64)

    for i in range(n - 1):
        dx = coords[i + 1, 0] - coords[i, 0]
        dy = coords[i + 1, 1] - coords[i, 1]
        orientations[i] = np.arctan2(dy, dx)

    orientations[-1] = orientations[-2]

    return orientations


import matplotlib.pyplot as plt


def plot_path_and_trajectory(path, trajectory):
    plt.figure(figsize=(8, 6))
    plt.plot(path[:, 0], path[:, 1], 'o-', label='Original Path', markersize=8)
    plt.plot(trajectory[:, 0], trajectory[:, 1], '.-', label='Smoothed Trajectory', markersize=4)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Path and Smoothed Trajectory')
    plt.grid()
    plt.show()


def get_global_to_local_transform(robot_state):
    """
    Get a 3x3 NumPy array representing the global to local coordinate transform.

    :param robot_state: A 3x1 NumPy array representing the current state of the robot, with
                        [x_position, y_position, orientation].
    :return: A 3x3 NumPy array representing the global to local coordinate transform.
    """
    x, y, theta = robot_state.flatten()
    transform = np.array([
        [np.cos(theta), -np.sin(theta), -x],
        [np.sin(theta), np.cos(theta), -y],
        [0, 0, 1]
    ])
    t = np.array([
        [1, 0, -x],
        [0, 1, -y],
        [0, 0, 1]
    ])
    r = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return transform


def visualize_robot_and_target(robot_state, target_point, scale=100, img_size=(500, 500)):
    """
    Visualize the robot's current position and a target point in local coordinates using OpenCV (cv2).

    :param robot_state: A 3x1 NumPy array representing the current state of the robot, with
                        [x_position, y_position, orientation].
    :param target_point: A 2x1 NumPy array representing the target point in global coordinates (x, y).
    :param scale: A float representing the scale factor for visualization (default: 100).
    :param img_size: A tuple representing the size of the output image (width, height).
    :return: None
    """
    img = np.zeros((*img_size, 3), dtype=np.uint8)

    # Convert global coordinates to local coordinates
    target_local = get_global_to_local_transform(robot_state)
    target_local_coords = np.dot(target_local, np.append(target_point, 1))

    # Calculate positions in the image
    robot_pos_img = np.array(img_size) // 2
    target_pos_img = (target_local_coords[:2] * scale + robot_pos_img).astype(int)
    target_pos_img[1] = img_size[1] - target_pos_img[1]  # Flip the y-coordinate

    # Draw the robot and the target point
    cv2.circle(img, tuple(robot_pos_img), 10, (0, 255, 0), -1)  # Robot (green)
    cv2.circle(img, tuple(target_pos_img), 5, (0, 0, 255), -1)  # Target point (red)

    diff = get_orientation_between_points((0, 0), target_local_coords)
    # display the difference between the robot's orientation and the orientation to the target point
    cv2.putText(img, f"Orientation difference: deg={np.rad2deg(diff):.2f}, rad={diff:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show the image
    cv2.imshow("Robot and Target", img)


def visualize_trajectory(path, trajectory, optimal_controls, robot_position, robot_orientation, w_left, w_right,
                         wheel_radius, wheel_distance, dt, max_angluar_velocity, max_acceleration,
                         scale=10, radius=5,
                         thickness=-1, wait_time=0):
    global debug
    # Set the dimensions for the visualization window
    buffer = 100
    width = int(max(np.max(path[:, 0]), np.max(trajectory[:, 0])) * scale) + 2 * radius + buffer * 2
    height = int(max(np.max(path[:, 1]), np.max(trajectory[:, 1])) * scale) + 2 * radius + buffer * 2
    # Create a black background
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.line(img, (buffer, 0), (buffer, height), (255, 255, 255), 1)
    cv2.line(img, (0, buffer), (width, buffer), (255, 255, 255), 1)

    # Define colors
    path_color = (255, 0, 0)  # Blue
    trajectory_color = (0, 255, 0)  # Green
    robot_color = (0, 0, 255)  # Red
    pred_path_color = (255, 255, 255)

    # Draw path points
    for point in path:
        x, y = point[:2]
        x = scale * x + buffer
        y = scale * y + buffer
        x = int(x)
        y = int(y)
        cv2.circle(img, (x, y), radius, path_color, thickness)

    # Draw trajectory points
    for point in trajectory:
        x, y = point[:2]
        x = scale * x + buffer
        y = scale * y + buffer
        x = int(x)
        y = int(y)
        cv2.circle(img, (x, y), radius, trajectory_color, thickness)

    pred_path = predict_trajectory(robot_position, robot_orientation, optimal_controls, dt,
                                   wheel_radius,
                                   wheel_distance)
    with_start = np.concatenate((robot_position.reshape(1, 2), pred_path[:, :2]), axis=0)
    for i, point in enumerate(with_start):
        x, y = point
        x = scale * x + buffer
        y = scale * y + buffer
        x = int(x)
        y = int(y)
        if i == 0:
            last_point = (x, y)
            continue
        cv2.line(img, last_point, (x, y), pred_path_color, 2)
        # cv2.circle(img, tuple((point * scale).astype(int)), radius, trajectory_color, thickness)
        last_point = (x, y)

    # Draw robot position
    x, y = robot_position
    x = scale * x + buffer
    y = scale * y + buffer
    x = int(x)
    y = int(y)
    cv2.circle(img, (x, y), radius * 2, robot_color, thickness)

    # Draw robot orientation line
    orientation_x = x + int(radius * 2 * np.cos(robot_orientation))
    orientation_y = y + int(radius * 2 * np.sin(robot_orientation))
    cv2.line(img, (x, y), (orientation_x, orientation_y), (0, 0, 0), 2)

    img = np.flipud(img)
    # Display the image
    cv2.imshow("Trajectory Visualization", img)

    visualize_robot_and_target(np.append(robot_position, robot_orientation), trajectory[0, :2])

    # Wait for a key press and close the window
    key = cv2.waitKey(wait_time)
    if key == ord('q'):
        cv2.destroyAllWindows()
        exit(0)
    if key == ord('p'):
        wait_time = 0 if wait_time == 1 else 1
        debug = False if debug else True

    # cv2.destroyAllWindows()
    return wait_time


@njit
def get_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# @njit
def pid_controller(Kp, Ki, Kd, setpoints, current_states, prev_errors, integral_errors, dt):
    errors = get_distance(setpoints[0], setpoints[1], current_states[0], current_states[1])
    integral_errors += errors
    derivative_errors = (errors - prev_errors)
    control_outputs = Kp * errors + Ki * integral_errors + Kd * derivative_errors
    return control_outputs, errors, integral_errors


# @njit
def pid_controller_angle(Kp, Ki, Kd, setpoints, current_states, prev_errors, integral_errors, dt):
    errors = get_orientation_difference(current_states[-1, 0],
                                        get_orientation_between_points(setpoints[:2], current_states[:2, 0]))
    body_to_goal = get_orientation_between_points(current_states[:2, 0], setpoints[:2])
    errors = (-body_to_goal) - current_states[2, 0]
    integral_errors += errors
    derivative_errors = (errors - prev_errors)
    control_outputs = Kp * errors + Ki * integral_errors + Kd * derivative_errors
    return control_outputs, errors, integral_errors


if __name__ == '__main__':
    visualize = True
    # Test the controller with a simple path
    path = np.array([[1., 0.], [0., 1.], [1., 1.], [2., 0.], [3., 1.], [4., 0.]])
    traj = generate_trajectory(path, distance_between_points=0.02)
    agv_config = {
        "wheel_radius": 0.1,
        "wheel_distance": 0.5,
        "max_speed": 1.,
        "max_acceleration": 1,
        "max_deceleration": 1.,
        "max_battery": 1000,
        "move_cost": 0,
        "idle_cost": 0,
        "radius": 1,
    }
    agv = AAGV(0, None, agv_config)
    agv.robot_state = np.array([
        [1.],  # x position
        [-0.2],  # y position
        [np.pi / 2],  # orientation
    ])
    # agv.orientation = agv.orientation

    dt = 1 / 240
    dt = 1 / 10
    wait_time = 5
    simulated_time = 0

    for t in range(1000):
        # Update the path before passing it to the MPC controller
        traj = update_path(traj, agv.robot_state[:2], threshold_distance=0.1)
        start = time.perf_counter()
        optimal_controls, cost = mpc_controller(traj[:agv.horizon],
                                                np.array([agv.x, agv.y]),
                                                np.array([agv.orientation]),
                                                dt, agv.horizon, agv.wheel_radius,
                                                agv.wheel_distance,
                                                agv.lw_speed, agv.rw_speed,
                                                agv.max_wheel_speed, agv.max_acceleration,
                                                agv.R, agv.Q, agv.Rd)

        end = time.perf_counter()
        if visualize:
            wait_time = visualize_trajectory(path, traj, optimal_controls, np.array([agv.x, agv.y]),
                                             np.array([agv.orientation]), agv.lw_speed, agv.rw_speed,
                                             agv.wheel_radius, agv.wheel_distance, dt, agv.max_speed,
                                             agv.max_acceleration,
                                             scale=250,
                                             radius=5,
                                             thickness=-1, wait_time=wait_time)

            pred_path = predict_trajectory(np.array([agv.x, agv.y]),
                                           np.array([agv.orientation]),
                                           optimal_controls, dt,
                                           agv.wheel_radius, agv.wheel_distance)

        # update the robot state
        lw_vel, rw_vel = optimal_controls[0]
        agv.set_wheel_velocity(lw_speed=lw_vel, rw_speed=rw_vel)
        agv.update(dt)
        state = agv.robot_state
        simulated_time += dt

        output = (
            f"Right wheel angular velocity: {agv.rw_speed}, input {rw_vel}\n"
            f"Left wheel angular velocity: {agv.lw_speed}, input {lw_vel}\n"
            f"Robot position: {[agv.x, agv.y]}\n"
            f"Robot orientation: {agv.orientation}\n"
            f"Target position: {traj[0]}\n"
            f"Linear velocity: {agv.linear_velocity}\n"
            f"Angular velocity: {agv.angular_velocity}\n"
            f"Length of path: {len(traj)}\n"
            f"Cost: {cost}\n"
            f"Time: {end - start}, hz: {1 / (end - start)}\n"

            f"Simulated time: {simulated_time}\r\n"
        )
        print(output)
