import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

from objects.mesh_base import Wireframe, unit_vector
from polynomial_fitting import plot_data
from tranformation.transform import TRotation, Transform
from utils.folder import get_project_root
from k_means_constrained import KMeansConstrained

cache = {}
data = {}


def read_json_file():
    global cache
    project_folder = get_project_root()
    with open(str(Path(project_folder, 'data/out/shortest_path.json')), 'r') as json_file:
        data = json.load(json_file)
    return data


def dist(p_to, p_from=np.array([0, 0, 0])):
    v = p_to - p_from
    return np.linalg.norm(v)


def move_to(current: int, currentDistance: float, visited, node_order: list, max_dist: float, edges, save_key: str):
    visited[current] = 1
    if str(visited) in cache and not cache[str(visited)]:
        return False, None, None
    node_order.append(int(current))
    valid_neighbors = np.where(edges[current] > -1)[0]
    valid_edges = edges[current, valid_neighbors].squeeze()
    indices = np.argsort(valid_edges).astype(int)
    test = valid_edges[indices]
    best_order = node_order
    found_solution = False
    all_visited = True
    print(
        f"\rDepth:{len(node_order)}, order[-10]={node_order if len(node_order) < 11 else node_order[-10:]}, dist={currentDistance}")
    for neighbour_index in indices:
        neighbour_index = valid_neighbors[neighbour_index]
        if visited[neighbour_index] == 1:
            continue
        all_visited = False
        distance = edges[current, neighbour_index]
        newDistance = currentDistance + distance
        if newDistance > max_dist:
            continue
        finished, resulting_dist, resulting_order = move_to(neighbour_index, newDistance, visited.copy(),
                                                            node_order.copy(),
                                                            max_dist, edges, save_key)
        if finished and max_dist > resulting_dist:
            max_dist = resulting_dist
            best_order = resulting_order
            found_solution = True
    if all_visited:

        if len(best_order) != len(edges[0]):
            cache[str(visited)] = False
        else:
            debug = 0
        data["current_path"] = node_order
        data[save_key] = {"cache": cache, "max_dist": max_dist}
        write_shortest_path(data)
        return len(best_order) == len(edges[0]), currentDistance, best_order
    return found_solution, max_dist, best_order


def move_to2(current: int, currentDistance: float, visited, node_order: list, max_dist: float, edges,
             points_to_sections, point_to_index, points):
    # Set visited:
    visited[current] = 1
    node_order.append(int(current))
    # Get end of section:
    p = points[current]
    section = points_to_sections[str(p)]
    p2 = section.get_other_end(p)

    # Set visisted
    current = point_to_index[str(p2)]
    if visited[current] == 0:
        visited[current] = 1
        node_order.append(int(current))
    # update distance travelled:
    currentDistance += section.length
    #
    if str(visited) in cache and not cache[str(visited)]:
        return False, None, None
    indices = np.argsort(edges[current]).astype(int)
    test = edges[current, indices]
    best_order = node_order
    found_solution = False
    all_visited = True
    print(
        f"\rDepth:{len(node_order)}, order[-10]={node_order if len(node_order) < 11 else node_order[-10:]}, dist={currentDistance}, BEST: {max_dist}")
    for neighbour_index in indices:
        if visited[neighbour_index] == 1:
            continue
        all_visited = False
        distance = edges[current, neighbour_index]
        newDistance = currentDistance + distance
        if newDistance > max_dist:
            continue
        finished, resulting_dist, resulting_order = move_to2(neighbour_index, newDistance, visited.copy(),
                                                             node_order.copy(),
                                                             max_dist, edges, points_to_sections, point_to_index,
                                                             points)
        if finished and max_dist > resulting_dist:
            max_dist = resulting_dist
            best_order = resulting_order
            found_solution = True
    if all_visited:
        if len(best_order) != len(edges[0]):
            cache[str(visited)] = False
        else:
            debug = 0
        data["current_path"] = node_order
        # data[save_key] = {"cache": cache, "max_dist": max_dist}
        write_shortest_path(data)
        return len(best_order) == len(edges[0]), currentDistance, best_order
    return found_solution, max_dist, best_order


def get_shortest_path(obj: Wireframe, project_distance_m: float = 0, max_dist=7):
    global data, cache
    key = f"p_{project_distance_m}_md_{max_dist}"
    data = read_json_file()
    cache = data[key]["cache"] if key in data else {}
    best_dist = data[key]["max_dist"] if key in data else float('inf')
    best_path = None
    best_order = None
    centers = obj.center
    normals = obj.normals
    points = centers + normals * project_distance_m
    edges = np.zeros((len(centers) + 1, len(centers) + 1))
    for i, p in enumerate(points):
        for j in range(i + 1, len(points)):
            p2 = points[j]
            d = dist(p, p2)
            edges[i, j] = d
            edges[j, i] = d
    start_node = np.array([-0.035274, -0.137386, 1.745499])
    start_node += np.array([-68, 32, 70])  # Offset
    start_node_cost = [0]
    for p in points:
        d = dist(p, start_node)
        start_node_cost.append(d)
    shortest_path_start = np.argsort(start_node_cost)
    edges[0] = np.array(start_node_cost)
    edges[:, 0] = np.array(start_node_cost).transpose()
    from python_tsp.exact import solve_tsp_dynamic_programming
    from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing
    edges[:, 0] = 0
    permutation, distance = solve_tsp_dynamic_programming(edges)
    permutation4, distance4 = solve_tsp_local_search(edges)
    write_shortest_path({"current_path": (np.array(permutation[1:]) - 1).tolist()})
    permutation2, distance2 = solve_tsp_simulated_annealing(edges)
    permutation3, distance3 = solve_tsp_local_search(edges, x0=permutation)

    debug = 0

    edges[np.where(edges > max_dist)] = -1

    for start_index in shortest_path_start:
        print(f"new_start = {start_index}")
        current = start_index
        currentDistance = start_node_cost[start_index]
        # unvisited = {node_index: None for node_index in range(len(points))}
        visited = np.zeros((len(edges[0])))
        # unvisited[current] = currentDistance
        found_solution, resulting_dist, resulting_order = move_to(current, currentDistance, visited, [], best_dist,
                                                                  edges, f"p_{project_distance_m}_md_{max_dist}")
        if found_solution and resulting_dist < best_dist:
            best_dist = resulting_dist
            best_order = resulting_order


def shortest_path(sections, multiplier=10):
    # Change normals to not go in z
    m_sections = []
    for s in sections:
        m_sections.append(s.scale_using_normal(multiplier))
    distance_traveled = 0
    best_dist = float('inf')
    points_to_sections = {}
    points = []
    for s in m_sections:
        points_to_sections[str(s.p1)] = s
        points_to_sections[str(s.p2)] = s
        points.append(s.p1)
        if np.any(s.p1 != s.p2):
            points.append(s.p2)

    edges = np.zeros((len(points), len(points)))
    for i, p in enumerate(points):
        for j in range(i + 1, len(points)):
            p2 = points[j]
            d = dist(p, p2)
            edges[i, j] = d
            edges[j, i] = d

    start_node = np.array([-0.035274, -0.137386, 1.745499])
    start_node += np.array([-68, 32, 70])  # Offset
    start_node_cost = []
    for p in points:
        d = dist(p, start_node)
        start_node_cost.append(d)
    shortest_path_start = np.argsort(start_node_cost)
    for start_index in shortest_path_start:
        print(f"new_start = {start_index}")
        current = start_index
        currentDistance = start_node_cost[start_index]

        visited = np.zeros((len(points)))
        point_to_index = {}
        for i, p in enumerate(points):
            point_to_index[str(p)] = i
        # unvisited[current] = currentDistance
        found_solution, resulting_dist, resulting_order = move_to2(current, currentDistance, visited, [], best_dist,
                                                                   edges, points_to_sections, point_to_index, points)
        if found_solution and resulting_dist < best_dist:
            best_dist = resulting_dist
            best_order = resulting_order
        break
    print("Done")
    debug = 0


def grouping(obj):
    centers = obj.center
    normals = obj.normals

    grp = {}
    # Group by normals
    string_to_normal = {}
    for i, n in enumerate(normals):
        n += np.zeros(3)
        n = np.round(n, 2)
        if str(n) not in grp:
            grp[str(n)] = []
            string_to_normal[str(n)] = n
        grp[str(n)].append(centers[i])

    grps = []
    lowest_amount_of_triangles = float('inf')
    for k, g in grp.items():
        if len(g) < lowest_amount_of_triangles:
            lowest_amount_of_triangles = len(g)

    for k, g in grp.items():
        if len(g) == lowest_amount_of_triangles:
            # The groups pointing up
            grps.append((string_to_normal[k], g))
        elif len(g) == lowest_amount_of_triangles + 1:
            # the end of the turbines
            # Find the longest dist to between nodes:
            distance = np.full((len(g), len(g)), float("inf"))
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    d = dist(g[i], g[j])
                    distance[i, j] = d
                    distance[j, i] = d
            outlier_index = np.min(distance, axis=0).argmax()
            grps.append((string_to_normal[k], [g[outlier_index]]))
            del g[outlier_index]
            grps.append((string_to_normal[k], g))
        else:
            number_of_groups = len(g) / lowest_amount_of_triangles
            np_g = np.array(g)
            clf = KMeansConstrained(
                n_clusters=int(number_of_groups),
                size_min=lowest_amount_of_triangles,
                size_max=lowest_amount_of_triangles,
                random_state=0
            )
            clf.fit_predict(np_g)
            print(clf.cluster_centers_)
            print(clf.labels_)

            for i in range(int(number_of_groups)):
                sub_grp = []
                for node_index, label in enumerate(clf.labels_):
                    if label == i:
                        sub_grp.append(g[node_index])
                grps.append((string_to_normal[k], sub_grp))

            debug = 0
    # plot_data_color(grps, "yes")
    return grps


def plot_data_color(data, title, save=False, show=True, dpi=600):
    # now lets plot it!
    fig = plt.figure(dpi=dpi)
    ax = Axes3D(fig)
    ax.grid(False)
    ax.set_facecolor('white')
    for d_grp in data:
        d = np.array(d_grp).transpose()
        ax.plot(d[0], d[1], d[2], label='Original Global Path', lw=2,
                c=(np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)))
    ax.legend()
    # plt.xlim(-50, -110)
    plt.savefig(f'data/fig/{title}_plot.png')
    if show:
        plt.show()
        plt.clf()
    return ax


def plot_data_color_sections(sections, title, save=False, show=True, dpi=600):
    # now lets plot it!
    fig = plt.figure(dpi=dpi)
    ax = Axes3D(fig)
    ax.grid(False)
    ax.set_facecolor('white')
    for d_grp in sections:
        d = d_grp.points.transpose()
        ax.plot(d[0], d[1], d[2], label='Original Global Path', lw=2,
                c=(np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)))
    ax.legend()
    # plt.xlim(-50, -110)
    plt.savefig(f'data/fig/{title}_plot.png')
    if show:
        plt.show()
        plt.clf()
    return ax


def write_shortest_path(info):
    # Directly from dictionary
    with open('data/out/shortest_path.json', 'w') as outfile:
        json.dump(info, outfile)


def midpoint(p1, p2):
    return (p1 + p2) / 2


class section:
    def __init__(self, points, normal):
        self.points = np.array(points)
        self.normal = normal
        if len(self.points) > 1:
            indices = np.argsort(self.points[:, 2])
            self.points = self.points[indices]
            self.p1 = midpoint(self.points[0], self.points[1])
            self.p2 = midpoint(self.points[-1], self.points[-2])
        else:
            self.p1 = self.points[0]
            self.p2 = self.points[0]
        self.length = dist(self.p1, self.p2)

    def get_other_end(self, p):
        if np.all(np.isclose(p, self.p1)):
            return self.p2
        return self.p1

    def scale_using_normal(self, multiplier):
        n = self.normal * np.array([1, 1, 0])
        n = unit_vector(n)
        p = self.points + n * multiplier
        return section(p, self.normal)


if __name__ == '__main__':
    wt = Wireframe.from_stl_path('data/in/turbine_v2.stl')
    r = Rotation.from_euler("XYZ", [0, 0, 90], degrees=True).as_matrix()

    r = TRotation().set_matrix(r, "XYZ")
    t = Transform(np.expand_dims(np.array([-80, 0, 20]), axis=1), r,
                  translate_before_rotate=False)
    wt = wt.transform(t)
    gps = grouping(wt)
    fit = []
    points = []
    sections = []
    for n, g in gps:
        sections.append(section(g, n))
    # plot_data_color_sections(sections, "te")
    order = shortest_path(sections)
    debug = 0
