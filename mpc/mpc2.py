import time

import cv2
import numpy as np
from numba import njit
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from agv import AAGV
from utils import _differential_drive_inverse_kinematics, _differential_drive_forward_kinematics, \
    my_clip, get_orientation_difference, get_orientation_between_points, _differential_drive_update_state

debug = False


# @njit
def predict_trajectory(robot_position: np.ndarray, robot_orientation: np.ndarray, current_wheel_vel: np.ndarray,
                       controls: np.ndarray, dt: float, wheel_radius: float,
                       wheel_distance: float,
                       max_angluar_velocity: float, max_acceleration: float):
    """
    Predict the trajectory of a differential drive robot given its initial state and a sequence of control inputs.

    Args:
        robot_position (np.ndarray): The initial 2D position of the robot (x, y).
        robot_orientation (np.ndarray): The initial orientation of the robot (theta) as a 1D array.
        current_wheel_vel (np.ndarray): The initial left and right wheel angular velocities.
        controls (np.ndarray): An array of control inputs, each consisting of linear velocity (v) and angular velocity (w).
        dt (float): The time step between control inputs in seconds.
        wheel_radius (float): The radius of the wheels in meters.
        wheel_distance (float): The distance between the wheels in meters.
        max_angluar_velocity (float): The maximum allowable angular velocity for the wheels in rad/s.
        max_acceleration (float): The maximum allowable linear acceleration for the wheels in m/s^2.

    Returns:
        predicted_trajectory (np.ndarray): An array of predicted 2D robot positions for each time step.
        actual_controls (np.ndarray): An array of the actual control inputs (v, w) applied at each time step,
                                      considering the constraints on wheel velocities and accelerations.
    """
    horizon = len(controls)
    robot_pos = robot_position.astype(np.float64)  # Cast robot_position to float64
    lw_vel = current_wheel_vel[0]
    rw_vel = current_wheel_vel[1]
    predicted_trajectory = np.zeros((horizon, 3))
    actual_controls = np.zeros((horizon, 2))
    robot_state = np.array([
        [robot_pos[0]],
        [robot_pos[1]],
        [robot_orientation[0]]])
    for t, (v, w) in enumerate(controls):
        w_left, w_right = _differential_drive_inverse_kinematics(v, w, wheel_radius, wheel_distance)

        # check limits
        w_left = my_clip(w_left, -max_angluar_velocity, max_angluar_velocity)
        w_right = my_clip(w_right, -max_angluar_velocity, max_angluar_velocity)
        w_left = my_clip(w_left, lw_vel - max_acceleration * dt, lw_vel + max_acceleration * dt)
        w_right = my_clip(w_right, rw_vel - max_acceleration * dt, rw_vel + max_acceleration * dt)

        lw_vel = w_left
        rw_vel = w_right

        a_v, a_w = _differential_drive_forward_kinematics(w_left, w_right, wheel_radius, wheel_distance)
        actual_controls[t] = a_v, a_w
        robot_state = _differential_drive_update_state(robot_state, a_v, a_w, dt)
        predicted_trajectory[t] = np.array([robot_state[0, 0], robot_state[1, 0], robot_state[2, 0]])

    return predicted_trajectory, actual_controls


@njit
def find_closest_point(point, points):
    """Finds the closest point to a given point from a list of points"""
    min_distance = np.inf
    closest_point = None
    for p in points:
        distance = np.sqrt(np.sum((point - p) ** 2))
        if distance < min_distance:
            min_distance = distance
            closest_point = p
    return closest_point, min_distance


# @njit
def _cost_function(controls, robot_position, robot_orientation, current_wheel_vel, path, dt, horizon, wheel_radius,
                   wheel_distance, max_angluar_velocity, max_acceleration, weights):
    """
    Calculate the cost of a sequence of control inputs for a differential drive robot.

    Args:
        controls (np.ndarray): An array of control inputs, each consisting of linear velocity (v) and angular velocity (w).
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
    R = weights[0]
    Q = weights[1]
    Rd = weights[2]
    trajectory, actual_controls = predict_trajectory(robot_position, robot_orientation, current_wheel_vel, controls, dt,
                                                     wheel_radius, wheel_distance, max_angluar_velocity,
                                                     max_acceleration)
    # Calculate the distance from the predicted trajectory to the path
    # Reshape the trajectory array to have shape (horizon, 1, 2)
    reshaped_trajectory = trajectory.reshape(horizon, 1, 3)

    # Subtract the path array from the reshaped trajectory array
    diffs = trajectory - path
    sqr_diffs = diffs ** 2
    # abs_diffs = np.abs(diffs)
    # abs_sum_dif = sqr_diffs[:, 0] + sqr_diffs[:, 1]
    distance_cost = np.zeros(horizon, dtype=np.float64)
    heading_cost = np.zeros(horizon, dtype=np.float64)
    control_cost = np.zeros(horizon, dtype=np.float64)
    control_change_cost = np.zeros(horizon - 1, dtype=np.float64)
    for i in range(horizon):
        distance_cost[i] = np.sum(Q[:2, :2] @ sqr_diffs[i, :2])  # * ((horizon - i) / horizon)
        # heading_cost[i] = Q[-1, -1] * sqr_diffs[i, -1] * ((horizon - i) / horizon)  # prioritize the first heading error

        diff_orientation = get_orientation_difference(trajectory[i, -1],
                                                      get_orientation_between_points(trajectory[i, :2],
                                                                                     path[i, :2])) ** 2

        heading_cost[i] = Q[-1, -1] * diff_orientation * ((horizon - i) / horizon)  # prioritize the first heading error

        control_cost[i] = np.sum(R @ controls[i] ** 2)
        # cost += np.sum(Q @ ((actual_controls[i] - controls[i]) ** 2)) * 10000
        if i < horizon - 1:
            control_change_cost[i] = np.sum(Rd @ (controls[i + 1] - controls[i]) ** 2)
    # for debugging:
    dist_cost = np.sum(distance_cost)
    head_cost = np.sum(heading_cost) * 0
    cont_cost = np.sum(control_cost) * 0
    cont_change_cost = np.sum(control_change_cost) * 0
    total_cost = dist_cost + head_cost + cont_cost + cont_change_cost
    # if debug:
    #     visualize_robot_and_target(trajectory[0], path[0, :2])
    return total_cost


def cost_function(controls, robot_position, robot_orientation, current_wheel_vel, path, dt, horizon, wheel_radius,
                  wheel_distance, max_angluar_velocity, max_acceleration, weights):
    """
    Wrapper for the numba implementation of the cost function for a differential drive robot.

    Args:
        Same as the _cost_function (see its docstring for details).

    Returns:
        cost (float): The total cost of the given sequence of control inputs.
    """
    return float(
        _cost_function(controls, robot_position, robot_orientation, current_wheel_vel, path, dt, horizon, wheel_radius,
                       wheel_distance, max_angluar_velocity, max_acceleration, weights))


def mpc_controller(path, robot_position, robot_orientation, dt, horizon, wheel_radius, wheel_distance, w_left, w_right,
                   max_angular_velocity, max_accel, weights):
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

    # min and max linear velocity)]
    bounds = [(0, 5),  # min and max linear velocity
              (np.deg2rad(-180), np.deg2rad(180))  # min and max angular velocity
              ] * horizon
    # Initialize the controls is linear velocity and angular velocity
    initial_controls = np.zeros((2 * horizon))

    result = minimize(cost_function,
                      x0=initial_controls,
                      args=(
                          robot_position, robot_orientation, np.array([w_left, w_right]), path, dt, horizon,
                          wheel_radius,
                          wheel_distance, max_angular_velocity, max_accel, weights),
                      bounds=bounds,
                      method='SLSQP',
                      options={'maxiter': 1000}
                      )

    optimal_controls = result.x.reshape((horizon, 2))
    # Return the first control input of the optimal sequence
    return optimal_controls, result.fun


def body_to_world(robot_pos, robot_orientation, v, w, dt):
    updated_orientation = robot_orientation + w * dt
    updated_position = robot_pos + np.array(
        [v * np.cos(updated_orientation), v * np.sin(updated_orientation)]) * dt
    return updated_position, updated_orientation


def update_path(path, robot_position, threshold_distance=0.25):
    distances = np.linalg.norm(path[:, :2] - robot_position, axis=1)
    remaining_points = distances > threshold_distance
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


def visualize_trajectories(path, agv_states, agv_trajectories, optimal_controls_list, wheel_radius, wheel_distance, dt,
                           total_time, scale=250, radius=5, thickness=-1, wait_time=0
                           ):
    global debug, frames
    # print(agv_states[0])
    num_agvs = len(agv_states)

    # Set the dimensions for the visualization window
    buffer = 100
    width = int(max(np.max(path[:, 0]),
                    max([np.max(traj[:, 0]) for traj in agv_trajectories])) * scale) + 2 * radius + buffer * 2
    height = int(max(np.max(path[:, 1]),
                     max([np.max(traj[:, 1]) for traj in agv_trajectories])) * scale) + 2 * radius + buffer * 2

    # width = int(max(np.max(path[:, 0]), np.max(agv_trajectories[0][:, 0])) * scale) + 2 * radius + buffer * 2
    # height = int(max(np.max(path[:, 1]), np.max(agv_trajectories[0][:, 1])) * scale) + 2 * radius + buffer * 2

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
        x = int(scale * x + buffer)
        y = int(scale * y + buffer)
        cv2.circle(img, (x, y), radius, path_color, thickness)

    # Draw trajectory points and robot positions for each AGV
    for i in range(num_agvs):
        trajectory = agv_trajectories[i]
        robot_position, robot_orientation = agv_states[i][:2, 0], agv_states[i][-1]

        for point in trajectory:
            x, y = point[:2]
            x = int(scale * x + buffer)
            y = int(scale * y + buffer)
            cv2.circle(img, (x, y), radius, trajectory_color, thickness)
        if optimal_controls_list is not None:
            pred_path = predict_trajectory(robot_position, robot_orientation, optimal_controls_list[i], dt,
                                           wheel_radius, wheel_distance)
            with_start = np.concatenate((robot_position.reshape(1, 2), pred_path[:, :2]), axis=0)
            for j, point in enumerate(with_start):
                x, y = point
                x = int(scale * x + buffer)
                y = int(scale * y + buffer)
                if j == 0:
                    last_point = (x, y)
                    continue
                cv2.line(img, last_point, (x, y), pred_path_color, 2)
                last_point = (x, y)

        # Draw robot position
        x, y = robot_position
        x = int(scale * x + buffer)
        y = int(scale * y + buffer)
        cv2.circle(img, (x, y), radius * 2, robot_color, thickness)

        # Draw robot orientation line
        orientation_x = x + int(radius * 2 * np.cos(robot_orientation))
        orientation_y = y + int(radius * 2 * np.sin(robot_orientation))
        cv2.line(img, (x, y), (orientation_x, orientation_y), (0, 0, 0), 2)

    img = np.flipud(img)
    img = img.astype(np.uint8)
    cv2.putText(img, total_time, (width - 200, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Save the frame as an image
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB format
    frames.append(frame)  # Add the frame to the list of frames

    # Display the image
    cv2.imshow("Trajectory Visualization", img)

    # Wait for a key press and close the window
    key = cv2.waitKey(wait_time)
    if key == ord('q'):
        cv2.destroyAllWindows()
        exit(0)
    if key == ord('p'):
        wait_time = 0 if wait_time == 1 else 1
        debug = False if debug else True

    return wait_time


@njit
def get_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Function to calculate desired heading and velocity
def calculate_desired_heading_and_velocity(position_error, kp_position, max_velocity):
    distance = np.linalg.norm(position_error)
    desired_heading = np.arctan2(position_error[1], position_error[0])

    desired_velocity = kp_position * distance
    if desired_velocity > max_velocity:
        desired_velocity = max_velocity

    return desired_heading, desired_velocity


# Function to convert desired heading and velocity to wheel velocities
def convert_to_wheel_velocities(desired_heading, desired_velocity, wheel_radius, wheel_distance):
    # Calculate the required angular velocity
    angular_error = desired_heading - robot.get_orientation()
    kp_angular = 0.5  # Tune this parameter
    desired_angular_velocity = kp_angular * angular_error

    desired_lw_vel, desired_rw_vel = _differential_drive_inverse_kinematics(desired_velocity, desired_angular_velocity,
                                                                            wheel_radius, wheel_distance)
    # Calculate the desired linear velocity
    wheel_radius = robot.get_wheel_radius()
    wheel_distance = robot.get_wheel_distance()

    # Calculate desired wheel velocities
    desired_left_wheel_velocity = (2 * desired_velocity - desired_angular_velocity * wheel_distance) / (
            2 * wheel_radius)
    desired_right_wheel_velocity = (2 * desired_velocity + desired_angular_velocity * wheel_distance) / (
            2 * wheel_radius)

    return desired_left_wheel_velocity, desired_right_wheel_velocity


# @njit
def pid_controller(Kp: float, Ki: float, Kd: float, desired_position: np.ndarray,
                   current_states: np.ndarray, prev_errors: np.ndarray, integral_errors: np.ndarray):
    position_error = desired_position[:, :2] - current_states[:, :2, 0]
    distance_error = np.linalg.norm(position_error, axis=1)

    integral_errors += distance_error
    derivative_errors = (distance_error - prev_errors)
    desired_velocity = Kp * distance_error + Ki * integral_errors + Kd * derivative_errors
    return desired_velocity, distance_error, integral_errors


# @njit
def pid_controller_angle(Kp: float, Ki: float, Kd: float, desired_position: np.ndarray,
                         current_states: np.ndarray, prev_errors: np.ndarray, integral_errors: np.ndarray):
    position_error = desired_position[:, :2] - current_states[:, :2, 0]
    desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
    angular_error = desired_heading - current_states[:, -1, 0]
    angular_error = np.arctan2(np.sin(angular_error), np.cos(angular_error))

    integral_errors += angular_error
    derivative_errors = (angular_error - prev_errors)
    desired_angular_velocity = Kp * angular_error + Ki * integral_errors + Kd * derivative_errors
    return desired_angular_velocity, angular_error, integral_errors


if __name__ == '__main__':
    visualize = True

    # Test the controller with a simple path
    path = np.array([[1., 0.], [0., 1.], [1., 1.], [2., 0.], [3., 1.], [4., 0.]])
    traj = generate_trajectory(path, distance_between_points=0.1)
    agv_config = {
        "wheel_radius": 0.1,
        "wheel_distance": 0.5,
        "max_speed": 5.,
        "max_acceleration": 5,
        "max_deceleration": 5.,
        "max_battery": 1000,
        "move_cost": 0,
        "idle_cost": 0,
        "radius": 1,
    }
    number_of_robots = 100

    agvs = [AAGV(0, None, agv_config) for _ in range(number_of_robots)]
    for agv in agvs:
        agv.robot_state = np.array([
            [np.random.uniform(0.5, 1.5)],  # x position
            [np.random.uniform(-0.2, 0.2)],  # y position
            [np.random.uniform(-np.pi, np.pi)],  # orientation
        ])
    trajs = [traj for _ in range(number_of_robots)]
    # agv.orientation = agv.orientation

    dt = 1 / 240
    dt = 1 / 10
    wait_time = 1
    simulated_time = 0
    prev_errors, integral_errors = np.zeros((number_of_robots,)), np.zeros((number_of_robots,))
    a_prev_errors, a_integral_errors = np.zeros((number_of_robots,)), np.zeros((number_of_robots,))
    optimal_controls = None
    dones = np.zeros(number_of_robots, dtype=np.bool)
    for t in range(10000):
        # Update the path before passing it to the MPC controller
        start = time.perf_counter()

        current_states = np.array([agv.robot_state for agv in agvs])
        desired_points = np.array([traj[0] for traj in trajs])
        lin_vel, prev_errors, integral_errors = pid_controller(1., 0., 1., desired_points, current_states,
                                                               prev_errors,
                                                               integral_errors)
        angl_vel, a_prev_errors, a_integral_errors = pid_controller_angle(1., 0., 1., desired_points, current_states,
                                                                          a_prev_errors, a_integral_errors)

        lw_speed, rw_speed = _differential_drive_inverse_kinematics(lin_vel, angl_vel, agv.wheel_radius,
                                                                    agv.wheel_distance)
        # find the ratio so max speed is 1
        max_speed = np.max(np.array([np.abs(lw_speed), np.abs(rw_speed)]).reshape(100, 2), axis=1)
        indx = np.argwhere(max_speed > agv.max_wheel_speed)
        lw_speed[indx] /= max_speed[indx]
        rw_speed[indx] /= max_speed[indx]
        simulated_time += dt

        for i, (lw_speed, rw_speed) in enumerate(zip(lw_speed, rw_speed)):
            agv = agvs[i]
            agv.set_wheel_velocity(lw_speed, rw_speed)
            agv.update(dt)
            new_traj = update_path(trajs[i], agv.robot_state[:2, 0], threshold_distance=0.1)
            current_states[i] = agv.robot_state
            if len(new_traj) == 0:
                new_traj = np.array([[agv.x, agv.y, agv.orientation]])
                dones[i] = True
            trajs[i] = new_traj
        end = time.perf_counter()
        total_time = f"Time: {(end - start):.2f} seconds"
        print(total_time)
        if visualize:
            wait_time = visualize_trajectories(path, current_states, trajs, None,
                                               agv_config["wheel_radius"], agv_config["wheel_distance"], dt, total_time,
                                               wait_time=wait_time)
