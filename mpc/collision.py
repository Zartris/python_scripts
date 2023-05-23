import time

import cv2
import imageio
import numpy as np
from numba import njit, prange, types, gdb, jit
from line_profiler_pycharm import profile
from numba.typed import Dict

from agv import AAGV
from utils import get_orientation_between_points, distance, norm_2d, distance_squared, my_clip_axis1

frames = []


def visualize_bots(agv_states, text, scale=250, radius=5, thickness=-1, wait_time=0, buffer=100):
    global frames
    start = time.perf_counter()
    num_agvs = len(agv_states)

    # Set the dimensions for the visualization window
    max_width = int(np.max(agv_states[:, 0]) * scale) + 2 * radius + buffer * 2
    max_height = int(np.max(agv_states[:, 1]) * scale) + 2 * radius + buffer * 2
    min_width = int(np.min(agv_states[:, 0]) * scale) - 2 * radius - buffer * 2
    min_height = int(np.min(agv_states[:, 1]) * scale) - 2 * radius - buffer * 2
    if save_gif:
        max_width = gif_size
        min_width = -gif_size
        max_height = gif_size
        min_height = -gif_size
    width = max_width - min_width
    height = max_height - min_height
    if width > 3000 or height > 3000:
        print("Too big")
        exit(0)
    # Create a blank image
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw the X and Y axes
    origin = (buffer - min_width, buffer - min_height)
    white = (255, 255, 255)  # White color
    green = (0, 255, 0)  # Green color
    darker_green = (89, 184, 55)  # Darker green color
    gray = (200, 200, 200)  # Gray color

    cv2.line(img, (origin[0], 0), (origin[0], max_height - min_height), white)
    cv2.line(img, (0, origin[1]), (max_width - min_width, origin[1]), white)

    # Draw the robots as circles
    for i in range(num_agvs):
        pos = (int(agv_states[i, 0] * scale) - min_width + buffer, int(agv_states[i, 1] * scale) - min_height + buffer)
        cv2.circle(img, pos, radius, darker_green, thickness, lineType=cv2.LINE_AA)

        # Draw robot orientation line
        orientation_x = pos[0] + int(radius * np.cos(agv_states[i, -1]))
        orientation_y = pos[1] + int(radius * np.sin(agv_states[i, -1]))
        cv2.line(img, pos, (orientation_x, orientation_y), white, 2, lineType=cv2.LINE_AA)

    img = np.flipud(img).astype(np.uint8)

    # compute text height
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, 0.5, 1)
    # Draw text in bottom left corner
    cv2.putText(img, text, (text_height, height - (text_height + 10)), font, 0.5, white, 1, cv2.LINE_AA)
    end = time.perf_counter()
    if save_gif:
        # Save the frame as an image
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB format
        frames.append(frame)  # Add the frame to the list of frames

    # Show the image
    cv2.imshow("Robots Visualization", img)
    key = cv2.waitKey(wait_time)
    if key == ord('q'):
        cv2.destroyAllWindows()
        exit(0)
    if key == ord('p'):
        wait_time = 0 if wait_time == 1 else 1
    return wait_time, end - start


# Make array type.  Type-expression is not supported in jit functions.
int64_array = types.uint64[:]


# @profile
@njit(fastmath=True)
def check_collision_naive(list_of_agents_pos, list_of_radius):
    collisions_mask = np.zeros((len(list_of_agents_pos),))
    collision_pairs = {}
    for i in range(len(list_of_agents_pos[:-1])):
        pos = list_of_agents_pos[i]
        # Distance to all other agents
        agents_pos = list_of_agents_pos[i + 1:]
        radii = list_of_radius[i + 1:]
        deltas = np.abs(agents_pos - pos)
        dists = deltas[:, 0] ** 2 + deltas[:, 1] ** 2
        (idxs,) = np.where(dists <= (radii + list_of_radius[i]) ** 2)
        if len(idxs) > 0:
            collisions_mask[i] = 1
            collisions_mask[i + 1 + idxs] = 1
            collision_pairs[i] = i + 1 + idxs
    return collisions_mask, collision_pairs


@njit(fastmath=True)
def create_endpoints(agents_pos, radii):
    num_agents = len(agents_pos)
    endpoints = np.empty((num_agents * 2, 3), dtype=np.float64)
    for i in prange(num_agents):
        pos, radius = agents_pos[i], radii[i]
        endpoints[i * 2, 0] = pos[0] - radius
        endpoints[i * 2, 1] = i
        endpoints[i * 2, 2] = 0  # 'start' is represented as 0
        endpoints[i * 2 + 1, 0] = pos[0] + radius
        endpoints[i * 2 + 1, 1] = i
        endpoints[i * 2 + 1, 2] = 1  # 'end' is represented as 1
    sorted_indices = np.argsort(endpoints[:, 0])
    return endpoints[sorted_indices]


def testing(agents_pos, radii):
    print("make endpoints")
    endpoints = create_endpoints(agents_pos, radii)
    print("len endpoints", len(endpoints))
    print("sweep and prune")
    _, cp = check_collision_sweep_and_prune(agents_pos, radii, endpoints)
    print("len cp", len(cp))


@njit(fastmath=True)
def check_collision_sweep_and_prune(agents_pos, radii):
    # Create sorted endpoints list
    endpoints = create_endpoints(agents_pos, radii)
    collision_mask = np.zeros((len(agents_pos)), dtype=np.uint8)
    active_list = []
    collision_pairs = Dict.empty(
        key_type=types.int64,
        value_type=int64_array,
    )
    # collision_pairs = {}
    for i in range(endpoints.shape[0]):
        idx, label = int(endpoints[i, 1]), int(endpoints[i, 2])
        if label == 0:  # 'start'
            for active_idx in active_list:
                # Check for collision along the y-axis
                delta_y = np.abs(agents_pos[active_idx][1] - agents_pos[idx][1])
                sum_radii = radii[active_idx] + radii[idx]
                if delta_y <= sum_radii:
                    # Check if the agents are actually colliding
                    if distance_squared(agents_pos[active_idx], agents_pos[idx]) <= sum_radii ** 2:
                        min_id = min(active_idx, idx)
                        max_id = max(active_idx, idx)
                        collision_mask[min_id] = 1
                        collision_mask[max_id] = 1
                        if min_id not in collision_pairs:
                            collision_pairs[min_id] = np.array([max_id], dtype=np.uint64)
                        else:
                            collision_pairs[min_id] = np.concatenate(
                                (collision_pairs[min_id], np.array([max_id], dtype=np.uint64)))
            active_list.append(idx)
        else:  # 'end'
            active_list.remove(idx)
    return collision_mask, collision_pairs


# @profile
@njit(fastmath=True)
def check_collision_spatial_hashing(agent_pos, radii, cell_size=0.5):
    assert max(radii) < cell_size, "Cell size must be larger than the largest radius"
    # Hash agents
    x = agent_pos[:, 0].copy()
    y = agent_pos[:, 1].copy()
    min_x = np.min(x)
    min_y = np.min(y)
    max_x = np.max(x)
    x -= (min_x - radii.max())  # shift to positive
    y -= (min_y - radii.max())  # shift to positive
    grid_width = np.ceil(((max_x + radii.max()) - (min_x - radii.max())) / cell_size)
    # hash_agent_pos_center = np.floor(x / cell_size) + np.floor(y / cell_size) * grid_width
    hash_agent_pos_top_right = (
            np.floor((x + radii) / cell_size) + np.floor((y + radii) / cell_size) * grid_width)
    hash_agent_pos_top_left = (
            np.floor((x - radii) / cell_size) + np.floor((y + radii) / cell_size) * grid_width)
    hash_agent_pos_bottom_right = (
            np.floor((x + radii) / cell_size) + np.floor((y - radii) / cell_size) * grid_width)
    hash_agent_pos_bottom_left = (
            np.floor((x - radii) / cell_size) + np.floor((y - radii) / cell_size) * grid_width)
    # create hash table
    hash_table = {}
    # Dict.empty(
    # key_type=types.int64,
    # value_type=int64_array,
    # )
    for i in range(len(agent_pos)):
        for j, cell_index in enumerate([hash_agent_pos_top_right[i], hash_agent_pos_top_left[i],
                                        hash_agent_pos_bottom_right[i], hash_agent_pos_bottom_left[i]]):
            if cell_index not in hash_table:
                hash_table[cell_index] = np.array([i], dtype=np.uint64)
            else:
                hash_table[cell_index] = np.concatenate(
                    (hash_table[cell_index], np.array([i], dtype=np.uint64)))

    # Check for collisions
    collision_pairs = {}
    collision_mask = np.zeros((len(agent_pos)), dtype=np.uint8)
    for cell_index, agents_in_cell in hash_table.items():
        agents_in_cell = np.sort(agents_in_cell)
        for i in range(len(agents_in_cell)):
            for j in range(i + 1, len(agents_in_cell)):
                agent_i = agents_in_cell[i]
                agent_j = agents_in_cell[j]
                dist = distance_squared(agent_pos[agent_i], agent_pos[agent_j])
                sum_radii = radii[agent_i] + radii[agent_j]
                if dist <= sum_radii ** 2:
                    collision_mask[agent_i] = 1
                    collision_mask[agent_j] = 1
                    if agent_i not in collision_pairs:
                        collision_pairs[agent_i] = np.array([agent_j], dtype=np.uint64)
                    else:
                        if agent_j not in collision_pairs[agent_i]:
                            collision_pairs[agent_i] = np.concatenate(
                                (collision_pairs[agent_i], np.array([agent_j], dtype=np.uint64)))
    # Return collision pairs
    return collision_mask, collision_pairs


def check_collision(agents_pos, radii):
    """ this is just a wrapper for the sweep and prune function"""

    if collision_test == "sweep_and_prune":
        return check_collision_sweep_and_prune(agents_pos, radii)
    elif collision_test == "spatial_hashing":
        return check_collision_spatial_hashing(agents_pos, radii, cell_size=radii.max() * 2)
    elif collision_test == "naive":
        return check_collision_naive(agents_pos, radii)
    else:
        raise ValueError("Unknown collision test method")


warmup_done = False


def warmup():
    global warmup_done
    if warmup_done:
        return
    print("Running Warmup")
    print("Warming up Naive collision method")
    forces = np.zeros((len(current_states), 2))
    start = time.perf_counter()
    _, collision_pairs = check_collision_naive(current_states[:, :2, 0], radius_list)
    print(f"time naive: {time.perf_counter() - start}")
    start_naive = time.perf_counter()
    _, collision_pairs = check_collision_naive(current_states[:, :2, 0], radius_list)
    end_naive = time.perf_counter()
    naive_time = end_naive - start_naive
    print(f"time naive: {naive_time}\n")

    print("Warming up Sweep and Prune (SaP) collision method")
    start = time.perf_counter()
    _, sap_cp = check_collision_sweep_and_prune(current_states[:, :2, 0], radius_list)
    print(f"time SaP: {time.perf_counter() - start}")
    start = time.perf_counter()
    _, sap_cp = check_collision_sweep_and_prune(current_states[:, :2, 0], radius_list)
    end = time.perf_counter()
    sap_time = end - start
    print(f"time SaP: {sap_time}\n")

    print("Warming up spatial hashing (SH) collision method")
    start = time.perf_counter()
    _, sh_cp = check_collision_spatial_hashing(current_states[:, :2, 0], radius_list, cell_size=radius_list.max() * 2)
    print(f"time SH: {time.perf_counter() - start}")
    start = time.perf_counter()
    _, sh_cp = check_collision_spatial_hashing(current_states[:, :2, 0], radius_list, cell_size=radius_list.max() * 2)
    end = time.perf_counter()
    sh_time = end - start
    print(f"time SH: {sh_time}\n")

    naive_num_col = 0
    sap_num_col = 0
    sh_num_col = 0
    for i, j in collision_pairs.items():
        naive_num_col += len(j) + 1
    for i, j in sap_cp.items():
        sap_num_col += len(j) + 1
        s, d = collision_pairs[i], sap_cp[i]
        debug = 0
    for i, j in sh_cp.items():
        sh_num_col += len(j) + 1
    print(f"Naive: {naive_num_col}, SaP: {sap_num_col}, SH: {sh_num_col}")
    # assert sap_num_col == naive_num_col
    # assert sh_num_col == naive_num_col
    print(f"speedup sap: {naive_time / sap_time}")
    print(f"speedup SH: {naive_time / sh_time}")
    forces = handle_collision_njit(collision_pairs, current_states, radius_list, spring_constant, forces)
    # handle_collision_njit.inspect_types()
    print(f"forces 0: {forces[0]}")

    warmup_done = True
    print("Warmup Done")


def naive_collision_test():
    global wait_time, results
    warmup()
    print("Running Naive Collision Test")
    total_collision_check = 0
    total_handle_collision = 0
    last_state = np.copy(current_states[:, :2, 0])
    is_not_same = True
    t = 0
    forces = np.zeros((len(current_states), 2))
    if visualize:
        wait_time, _ = visualize_bots(current_states, text="epoch 0", scale=scale,
                                      radius=int(agv_config["radius"] * scale), buffer=0,
                                      wait_time=wait_time)
    while is_not_same:
        t += 1
        # forces = np.zeros((len(current_states), 2))
        start = time.perf_counter()
        collisions_mask, collision_pairs = check_collision(current_states[:, :2, 0], radius_list)
        end = time.perf_counter()
        total_collision_check += end - start

        # Handle collisions
        start_handle = time.perf_counter()
        forces = handle_collision_njit(collision_pairs, current_states, radius_list, spring_constant, forces)
        end_handle = time.perf_counter()
        total_handle_collision += end_handle - start_handle
        # print(f"forces 0: {forces[0]}")

        # Update states
        current_states[:, :2, 0] += forces
        for state in current_states:
            agv.robot_state = state
        # dampen_forces(forces, 0.1)
        # forces = forces * 1
        # forces[forces < 0.0001] = 0

        # print statistics
        num_collisions = 0
        for i, j in collision_pairs.items():
            num_collisions += len(j) + 1

        is_not_same = num_collisions > 0
        # is_not_same = not np.allclose(current_states[:, :2, 0], last_state)
        last_state = current_states[:, :2, 0].copy()
        print(f"{t}: \n"
              f"\tTime taken to handle collision check: {(end - start) * 1000:0.2f} milliseconds \n"
              f"\tTime taken to handle collisions: {(end_handle - start_handle) * 1000:0.2f} milliseconds\n"
              f"\tNumber of collisions: {num_collisions}"
              )
        if visualize:
            wait_time, draw_time = visualize_bots(current_states,
                                                  text=f"epoch {t}, collisions: {num_collisions}, processing: {((end - start) + (end_handle - start_handle)) * 1000:0.2f} ms",
                                                  scale=scale, radius=int(agv_config["radius"] * scale),
                                                  buffer=0,
                                                  wait_time=wait_time)
            print(f"\tTime taken to draw: {draw_time * 1000:0.2f} milliseconds\n")
    s = f"Collision param: method={collision_test}, spring_constant={spring_constant}\n" \
        f"\tTotal epochs for convergence: {t}\n" \
        f"\tTotal collision check time taken: {total_collision_check * 1000:0.2f} milliseconds\n" \
        f"\tAverage collision check time taken: {total_collision_check / t * 1000:0.2f} milliseconds\n" \
        f"\tTotal handle collision time taken: {total_handle_collision * 1000:0.2f} milliseconds\n" \
        f"\tAverage handle collision time taken: {total_handle_collision / t * 1000:0.2f} milliseconds\n"
    print(s)
    results.append(s)

    if save_gif:
        imageio.mimsave(f"collision_{num_agvs}_s{spring_constant}.gif", frames, 'GIF', duration=0.05)


# @njit
def handle_collision(list_of_collision_pairs, current_states, radius_list, spring_constant):
    new_forces = np.zeros((len(current_states), 2), dtype=np.float64)
    for i, j in list_of_collision_pairs.items():
        current_agent_forces, other_agents_forces = get_repelling_forces(current_states[i, :2, 0],
                                                                         current_states[j, :2, 0],
                                                                         radius_list[i], radius_list[j],
                                                                         spring_constant)
        new_forces[i] -= current_agent_forces
        new_forces[j] += other_agents_forces
    return new_forces


@njit(fastmath=True)
def handle_collision_njit(list_of_collision_pairs, current_states, radius_list, spring_constant, forces):
    new_forces = np.zeros((len(current_states), 2), dtype=np.float64)
    for i, j in list_of_collision_pairs.items():
        current_agent_forces, other_agents_forces = get_repelling_forces(current_states[i, :2, 0],
                                                                         current_states[j, :2, 0],
                                                                         radius_list[i], radius_list[j],
                                                                         spring_constant, forces[i], forces[j])
        new_forces[i] -= current_agent_forces
        new_forces[j] += other_agents_forces
    new_forces = my_clip_axis1(new_forces, a_min=-radius_list, a_max=radius_list)
    return new_forces


@njit(fastmath=True)
def get_repelling_force(agent_pos_1, agent_pos_2, radius_1, radius_2, spring_constant):
    d = distance(agent_pos_1, agent_pos_2)
    overlap = radius_1 + radius_2 - d

    force = spring_constant * overlap
    angle = get_orientation_between_points(agent_pos_1, agent_pos_2)
    return np.cos(angle) * force, np.sin(angle) * force


@njit(fastmath=True)
def get_repelling_forces(agent_pos_1, agent_pos_2: np.ndarray, radius_1, radius_2, spring_constant,
                         agent_1_force, agent_2_forces):
    diff = agent_pos_2 - agent_pos_1
    d = norm_2d(diff)
    overlap = radius_1 + radius_2 - d
    a1_not_moved = int(np.sum(agent_1_force) == 0)
    a2_not_moved = (np.sum(agent_2_forces, axis=1) == 0).astype(np.int64) + 1
    moved_mask = a2_not_moved - a1_not_moved
    force = moved_mask * spring_constant * overlap

    moved_mask[moved_mask == 2] = 0
    angle = np.arctan2(diff[:, 1], diff[:, 0])
    forces = np.column_stack((np.cos(angle) * force, np.sin(angle) * force))
    a1_forces = np.sum(forces * np.column_stack((moved_mask, moved_mask)), axis=0)
    return a1_forces, forces


if __name__ == '__main__':
    results = []
    np.random.seed(2)
    frames = []
    visualize = True
    save_gif = True
    gif_size = 450
    scale = 50
    wait_time = 1
    spring_constant = 1
    collision_test = "sweep_and_prune"
    # Test the controller with a simple path

    agv_config = {
        "wheel_radius": 0.1,
        "wheel_distance": 0.5,
        "max_speed": 1.,
        "max_acceleration": 1,
        "max_deceleration": 1.,
        "max_battery": 1000,
        "move_cost": 0,
        "idle_cost": 0,
        "radius": 0.1,
    }

    num_agvs = 5000
    start_area = 1
    agvs = [AAGV(i, None, agv_config) for i in range(num_agvs)]
    for agv in agvs:
        agv.robot_state = np.array([
            [np.random.uniform(-start_area, start_area)],  # x position
            [np.random.uniform(-start_area, start_area)],  # y position
            [np.random.uniform(-np.pi, np.pi)],  # orientatio
        ])

    dt = 1 / 10
    # Initialize the current state for each AGV
    initial_states = np.array([agv.robot_state for agv in agvs])
    forces = np.zeros((num_agvs, 2))
    radius_list = np.array([agv.radius for agv in agvs])

    # Uncomment for testing different collision methods
    # for method in [
    #     # "naive",
    #     "sweep_and_prune",
    #     # "spatial_hashing"
    # ]:
    #     for i, agv in enumerate(agvs):
    #         agv.robot_state = initial_states[i].copy()
    #     current_states = initial_states.copy()
    #     forces = np.zeros((num_agvs, 2))
    #     collision_test = method
    #     naive_collision_test()

    # Uncomment for testing different spring constants
    for sc in [
        1.0, 1.2, 1.4, 1.6, 1.8, 2.0
    ]:
        frames = []
        for i, agv in enumerate(agvs):
            agv.robot_state = initial_states[i].copy()
        current_states = initial_states.copy()
        forces = np.zeros((num_agvs, 2))
        spring_constant = sc
        naive_collision_test()

    for s in results:
        print(s)
    # check_collision.parallel_diagnostics(level=4)
    #
