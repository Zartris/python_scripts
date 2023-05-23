from multiprocessing import Pool

import cv2
import imageio
import numpy as np
import time

from MPC_wheel_control import generate_trajectory, mpc_controller, predict_trajectory, update_path
from agv import AAGV
from pid import pid_controller, pid_controller_angle
from utils import _differential_drive_inverse_kinematics


# (Include your imports and other code here)

def update_agv(agv_id, agv, traj, dt):
    # (Include initialization code here, but use current_state instead of initializing a new AGV)
    optimal_controls, cost = mpc_controller(traj[:agv.horizon],
                                            np.array([agv.x, agv.y]),
                                            np.array([agv.orientation]),
                                            dt, agv.horizon, agv.wheel_radius,
                                            agv.wheel_distance,
                                            agv.lw_speed, agv.rw_speed,
                                            agv.max_wheel_speed, agv.max_acceleration,
                                            agv.R, agv.Q, agv.Rd)
    return optimal_controls


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

    # find longest path:
    longest_path = []
    for i in range(num_agvs):
        if len(agv_trajectories[i]) > len(longest_path):
            longest_path = agv_trajectories[i]

    for point in longest_path:
        x, y = point[:2]
        x = int(scale * x + buffer)
        y = int(scale * y + buffer)
        cv2.circle(img, (x, y), radius, trajectory_color, thickness)

    # Draw trajectory points and robot positions for each AGV
    for i in range(num_agvs):
        robot_position, robot_orientation = agv_states[i][:2, 0], agv_states[i][-1]

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


def generate_circle_points(center, radius, num_points):
    circle_points = []
    angle_step = 2 * np.pi / num_points

    for i in range(num_points):
        angle = i * angle_step
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        circle_points.append((x, y))

    return circle_points


def pid_tunning():
    prev_errors, integral_errors = np.zeros((num_agvs,)), np.zeros((num_agvs,))
    a_prev_errors, a_integral_errors = np.zeros((num_agvs,)), np.zeros((num_agvs,))
    max_wheel_speed = agvs[0].max_wheel_speed
    circle_path = np.array(generate_circle_points((2, 2), 1, 100))
    # circle_path = np.array([[0, 0]])  # tuning:
    circle_paths = [circle_path for _ in range(num_agvs)]
    kp_ang = 10.
    ki_ang = 0.1
    kd_ang = 0.
    kp_lin = 4.
    ki_lin = 0.
    kd_lin = 1.
    current_states = np.array([agv.robot_state for agv in agvs])
    desired_points = np.array([circle_path[0] for circle_path in circle_paths])
    path_index = np.zeros((num_agvs,), dtype=np.int32)
    t = 0
    while True:
        t += 1
        # Update the path before passing it to the MPC controller
        start = time.perf_counter()
        lin_vel, prev_errors, integral_errors = pid_controller(kp_lin, ki_lin, kd_lin, desired_points, current_states,
                                                               prev_errors,
                                                               integral_errors)
        angl_vel, a_prev_errors, a_integral_errors = pid_controller_angle(kp_ang, ki_ang, kd_ang, desired_points,
                                                                          current_states,
                                                                          a_prev_errors, a_integral_errors)

        lw_speed, rw_speed = _differential_drive_inverse_kinematics(lin_vel, angl_vel, agv_config["wheel_radius"],
                                                                    agv_config["wheel_distance"])
        # find the ratio so max speed is 1
        max_speed = np.max(np.array([np.abs(lw_speed), np.abs(rw_speed)]).reshape(num_agvs, 2), axis=1)
        indx = np.argwhere(max_speed > max_wheel_speed)
        lw_speed[indx] /= max_speed[indx]
        rw_speed[indx] /= max_speed[indx]

        for i, (lw_speed, rw_speed) in enumerate(zip(lw_speed, rw_speed)):
            agv = agvs[i]
            agv.set_wheel_velocity(lw_speed, rw_speed)
            agv.update(dt)
            current_states[i] = agv.robot_state
            # if close to the desired point, update the desired point
            if np.linalg.norm(current_states[i][:2, 0] - desired_points[i]) < 0.1:
                path_index[i] += 1
                if path_index[i] >= len(circle_paths[i]):
                    path_index[i] = 0
            desired_points[i] = circle_paths[i][path_index[i]]
        end = time.perf_counter()
        total_time = f"Time: {(end - start) * 1000:.2f} ms"
        print(total_time)

        if visualize and t % 5 == 0:
            visualize_trajectories(circle_path, current_states, circle_paths, None,
                                   agv_config["wheel_radius"], agv_config["wheel_distance"], dt, total_time,
                                   wait_time=wait_time)
        if np.all(dones):
            break


def test_pid():
    prev_errors, integral_errors = np.zeros((num_agvs,)), np.zeros((num_agvs,))
    a_prev_errors, a_integral_errors = np.zeros((num_agvs,)), np.zeros((num_agvs,))
    max_wheel_speed = agvs[0].max_wheel_speed

    # tuning:
    kp_ang = 8. # 6
    ki_ang = 0.0
    kd_ang = 0.1
    kp_lin = 4. # 3
    ki_lin = 0.0
    kd_lin = 0.1
    current_states = np.array([agv.robot_state for agv in agvs])
    desired_points = np.array([traj[0] for traj in trajs])
    for t in range(10000):
        # Update the path before passing it to the MPC controller
        start = time.perf_counter()
        lin_vel, prev_errors, integral_errors = pid_controller(kp_lin, ki_lin, kd_lin, desired_points, current_states,
                                                               prev_errors,
                                                               integral_errors)
        angl_vel, a_prev_errors, a_integral_errors = pid_controller_angle(kp_ang, ki_ang, kd_ang, desired_points,
                                                                          current_states,
                                                                          a_prev_errors, a_integral_errors)

        lw_speed, rw_speed = _differential_drive_inverse_kinematics(lin_vel, angl_vel, agv_config["wheel_radius"],
                                                                    agv_config["wheel_distance"])
        # find the ratio so max speed is 1
        max_speed = np.max(np.array([np.abs(lw_speed), np.abs(rw_speed)]).reshape(num_agvs, 2), axis=1)
        indx = np.argwhere(max_speed > max_wheel_speed)
        lw_speed[indx] /= max_speed[indx]
        rw_speed[indx] /= max_speed[indx]

        for i, (lw_speed, rw_speed) in enumerate(zip(lw_speed, rw_speed)):
            agv = agvs[i]
            agv.set_wheel_velocity(lw_speed, rw_speed)
            agv.update(dt)
            new_traj = update_path(trajs[i], agv.robot_state[:2], threshold_distance=0.1)
            current_states[i] = agv.robot_state
            if len(new_traj) == 0:
                new_traj = np.array([[agv.x, agv.y, agv.orientation]])
                dones[i] = True
            trajs[i] = new_traj
            desired_points[i] = new_traj[0]
        end = time.perf_counter()
        total_time = f"Time: {(end - start) * 1000:.2f} ms"
        print(total_time)

        if visualize:
            visualize_trajectories(path, current_states, trajs, None,
                                   agv_config["wheel_radius"], agv_config["wheel_distance"], dt, total_time,
                                   wait_time=wait_time)
        if np.all(dones):
            break
    imageio.mimsave(f"multi-agent-pid_{num_agvs}.gif", frames, 'GIF', duration=0.05)

    pass


# Call the function
def test_mpc():
    # Create a multiprocessing pool with the desired number of processes
    with Pool() as pool:
        for t in range(10000):
            # Create the input arguments for the update_agv function
            start = time.perf_counter()
            input_args = [(i, agvs[i], trajs[i], dt) for i in range(num_agvs) if len(trajs[i]) > 0]

            # Call the update_agv function for each AGV using the multiprocessing pool
            results = pool.starmap(update_agv, input_args)
            # Update the current states and print the output for each AGV
            optimal_controls_list = []
            for i, (optimal_control) in enumerate(results):
                agv = agvs[i]
                lw_vel, rw_vel = optimal_control[0]
                agv.set_wheel_velocity(lw_speed=lw_vel, rw_speed=rw_vel)
                agv.update(dt)
                new_traj = update_path(trajs[i], agv.robot_state[:2], threshold_distance=0.1)
                optimal_controls_list.append(optimal_control)
                current_states[i] = agv.robot_state
                if len(new_traj) == 0:
                    new_traj = np.array([[agv.x, agv.y, agv.orientation]])
                    dones[i] = True
                trajs[i] = new_traj
            end = time.perf_counter()
            total_time = f"Time: {(end - start):.2f} seconds"
            print(total_time)

            if visualize:
                visualize_trajectories(path, current_states, trajs, optimal_controls_list,
                                       agv_config["wheel_radius"], agv_config["wheel_distance"], dt, total_time,
                                       wait_time=wait_time)
            if np.all(dones):
                break
    imageio.mimsave(f"multi-agent-mpc_{num_agvs}.gif", frames, 'GIF', duration=0.1)


if __name__ == '__main__':
    np.random.seed(2)
    frames = []
    visualize = True
    wait_time = 1
    controller = "MPC"
    controller = "PID"
    # Test the controller with a simple path
    path = np.array([[1., 0.], [0., 1.], [1., 1.], [2., 0.], [3., 1.], [4., 0.]])
    traj = generate_trajectory(path, distance_between_points=0.05)

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

    num_agvs = 1000
    agvs = [AAGV(i, None, agv_config) for i in range(num_agvs)]
    for agv in agvs:
        agv.robot_state = np.array([
            [np.random.uniform(0.5, 1.5)],  # x position
            [np.random.uniform(-0.2, 0.2)],  # y position
            [-np.pi],  # orientation
        ])
    trajs = [traj for _ in range(num_agvs)]
    dt = 1 / 10
    # Initialize the current state for each AGV
    current_states = [agv.robot_state for agv in agvs]
    dones = np.zeros(num_agvs, dtype=np.bool)
    # pid_tunning()
    # Create a multiprocessing pool with the desired number of processes
    if controller == "MPC":
        test_mpc()
    else:
        test_pid()
