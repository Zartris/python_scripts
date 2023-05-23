import numpy as np
from colorama import Fore
from colorama import Style
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from polynomial_fitting import readfile, readfile_multiple_values, remove_duplicates


def compute_dist(color, px, py, pz):
    p_list = []
    total_dist = 0
    for i in range(1, len(px)):
        start = np.array([px[i - 1], py[i - 1], pz[i - 1]])
        end = np.array([px[i], py[i], pz[i]])
        dist = np.linalg.norm(end - start)
        s = f"{color}start: {start.T} -> end: {end.T}, dist: {dist}{Style.RESET_ALL}"
        print(s)
        p_list.append(s)
        total_dist += dist

    return p_list, total_dist


def plot_data_color_connected(graph1, graph2, title, save=False, show=True, dpi=600):
    # now lets plot it!
    plt.clf()
    fig = plt.figure(dpi=dpi)
    try:
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
    except:
        ax = Axes3D(fig)
    ax.grid(False)
    ax.set_facecolor('white')
    ax.plot(graph1[0].squeeze(), graph1[1].squeeze(), graph1[2].squeeze(), lw=2, c='black')
    ax.plot(graph2[0].squeeze(), graph2[1].squeeze(), graph2[2].squeeze(), lw=2, c='green')
    #
    # last_point = None
    # for d_grp, c in graph1:
    #     d = np.array(d_grp).transpose()
    #     if last_point is not None:
    #         last_point[0].append(d[0, 0])
    #         last_point[1].append(d[1, 0])
    #         last_point[2].append(d[2, 0])
    #         ax.plot(last_point[0], last_point[1], last_point[2], lw=2, c=color)
    #     color = c
    #     ax.plot(d[0], d[1], d[2], lw=2, c=color)
    #     last_point = [[d[0, -1]], [d[1, -1]], [d[2, -1]]]
    ax.legend()
    # plt.xlim(-50, -110)
    # plt.savefig(f'data/fig/{title}_plot.png')
    if show:
        plt.show()
        plt.clf()
    return ax


if __name__ == '__main__':
    px = readfile("data/in/cheat_output/px_160.txt", expand=True)
    py = readfile("data/in/cheat_output/py_160.txt", expand=True)
    pz = readfile("data/in/cheat_output/pz_160.txt", expand=True)
    o_pp, total_dist = compute_dist(Fore.LIGHTYELLOW_EX, px, py, pz)
    print(f"other path planner len: {total_dist}\n\n")
    m_px = readfile("data/out/p162_x.txt", expand=True)
    m_py = readfile("data/out/p162_y.txt", expand=True)
    m_pz = readfile("data/out/p162_z.txt", expand=True)
    #
    # px = readfile("data/in/mp_output/p162_x.txt", expand=True)
    # py = readfile("data/in/mp_output/p162_y.txt", expand=True)
    # pz = readfile("data/in/mp_output/p162_z.txt", expand=True)
    my_pp, total_dist = compute_dist(Fore.LIGHTBLUE_EX, m_px[1:-1], m_py[1:-1], m_pz[1:-1])

    plot_data_color_connected(np.array([px, py, pz]), np.array([m_px[1:-1], m_py[1:-1], m_pz[1:-1]]), "test")
    print(f"our path planner len: {total_dist}\n\n")
    index = 0
    for s1, s2 in zip(o_pp, my_pp):
        print(index, s1)
        print(index, s2)
        index += 1

    values = readfile_multiple_values("data/in/gp_path.txt", expand=False)
    values = remove_duplicates(values, 6)
    pos = values[:, :3]
    angles = values[:, 3:]
    o_total_pp, total_dist = compute_dist(Fore.LIGHTGREEN_EX, pos[:, 0], pos[:, 1], pos[:, 2])
    print(f"Their total path planner len: {total_dist}\n\n")

    full_px = readfile("data/out/test_full_path_x.txt", expand=True)
    full_py = readfile("data/out/test_full_path_y.txt", expand=True)
    full_pz = readfile("data/out/test_full_path_z.txt", expand=True)
    m_total_pp, total_dist = compute_dist(Fore.LIGHTMAGENTA_EX, full_px, full_py, full_pz)
    print(f"My total path planner len: {total_dist}\n\n")
    plot_data_color_connected(np.array([pos[:, 0], pos[:, 1], pos[:, 2]]), np.array([full_px, full_py, full_pz]),
                              "test2")
    # smoothed
    s_full_px = readfile("data/out/full_fit_noiterp_x.txt", expand=True)
    s_full_py = readfile("data/out/full_fit_noiterp_y.txt", expand=True)
    s_full_pz = readfile("data/out/full_fit_noiterp_z.txt", expand=True)
    m_total_fit_pp, total_dist = compute_dist(Fore.LIGHTCYAN_EX, s_full_px, s_full_py, s_full_pz)
    print(f"My total smooth path planner len: {total_dist}\n\n")
    plot_data_color_connected(np.array([pos[:, 0], pos[:, 1], pos[:, 2]]), np.array([s_full_px, s_full_py, s_full_pz]),
                              "test2")
    #
    # index = 0
    # for s1, s2 in zip(o_pp, my_pp):
    #     print(index, s1)
    #     print(index, s2)
    #     index += 1
