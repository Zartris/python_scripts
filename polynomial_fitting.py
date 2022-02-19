import os
import time

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D


def readfile(filename, expand=False):
    file1 = open(filename, 'r')
    Lines = file1.readlines()

    count = 0
    # Strips the newline character
    lines = []
    for line in Lines:
        value = float(line.strip())
        if value == -0:
            value = 0
        lines.append(value)
    lines = np.array(lines)
    if expand:
        lines = np.expand_dims(lines, axis=1)
    return lines


def writefile(filename, data):
    with open(filename, 'w') as file:
        for data_point in data:
            L = [f"{str(data_point)}\n"]
            file.writelines(L)


def polynomial_fit(deg: list, smooth: list):
    from scipy.interpolate import splprep, splev
    px = readfile("data/px_160.txt", expand=True)
    py = readfile("data/py_160.txt", expand=True)
    pz = readfile("data/pz_160.txt", expand=True)

    data = np.concatenate((px, py), axis=1)
    data = np.concatenate((data, pz), axis=1)

    data = data.astype(float)
    # Remove duplicates:
    last_point = None
    points = []
    for point in data:
        if np.all(point == last_point):
            continue
        points.append(point)
        last_point = point

    data = np.array(points)
    data = data.transpose()

    # now we get all the knots and info about the interpolated spline
    titles = []
    data_list = []
    for degrees in deg:
        for smoothnes in smooth:
            titles.append(f"d{degrees}s{smoothnes}")
            tck, u = interpolate.splprep(data, k=degrees, s=smoothnes)
            # here we generate the new interpolated dataset,
            # increase the resolution by increasing the spacing, 500 in this example
            new = interpolate.splev(np.linspace(0, 1, 5000), tck, der=0)
            data_list.append(new)

    # now lets plot it!
    for new, t in zip(data_list, titles):
        plot_data(data, new, t, save=True)
        plot_axis(data, new, t)
        plt.clf()


def trunc(values, decs=0):
    return np.trunc(values * 10 ** decs) / (10 ** decs)


def split_data_into_sections(data):
    # Extract points and normals
    data_points = data[:, :3]
    normals = data[:, 3:]

    # Splitted_points is the resulting list
    splitted_points = []
    tmp_list = []  # tmp_list is the list containing the points that are grouped
    old_n = None
    old_uv = None
    points_to_remember = []
    new_section = False
    for data_point, normal in zip(data_points, normals):
        n = trunc(normal, 3)  # We truncate to estimate the normals instead of using exact values
        if np.any(n != old_n):  # If the normal changes we start a new group
            new_section = True
        elif len(tmp_list) > 1:
            p = tmp_list[-2]
            v = p - data_point
            unit_vector = v / np.linalg.norm(v)
            if old_uv is None:
                old_uv = unit_vector
            else:
                if not np.allclose(old_uv, unit_vector, 0.001):
                    debug = 0
                    print("SHIFT")
                    new_section = True
        if new_section:
            if len(tmp_list) != 0:  # Check if we should add old group
                splitted_points.append(np.array(tmp_list.copy()))
                tmp_list = []
            old_n = n
            new_section = False
        tmp_list.append(data_point)
    return splitted_points


def np_polyfit(deg):
    px = readfile("data/px_160.txt", expand=True)
    py = readfile("data/py_160.txt", expand=True)
    pz = readfile("data/pz_160.txt", expand=True)
    data_points = np.concatenate((px, py, pz), axis=1)
    # data_points = np.concatenate((data_points, pz), axis=1)
    # Normals
    nx = readfile("data/nx_160.txt", expand=True)
    ny = readfile("data/ny_160.txt", expand=True)
    nz = readfile("data/nz_160.txt", expand=True)
    normals = np.concatenate((nx, ny, nz), axis=1)
    data = np.concatenate((data_points, normals), axis=1)
    # Remove duplicates:
    data = remove_duplicates(data)
    split_data = split_data_into_sections(data)
    for dps in split_data:
        t = np.array(range(len(dps)))
        data_points = dps.transpose()
        fit_dx_eq = polyfit_np_axis(t, data_points[0], t, deg)
        fit_dy_eq = polyfit_np_axis(t, data_points[1], t, deg)
        fit_dz_eq = polyfit_np_axis(t, data_points[2], t, deg)
        plot_data(data_points, [fit_dx_eq, fit_dy_eq, fit_dz_eq], "yes")
    debug = 0


def old_polyfit():
    px = readfile("data/px_160.txt", expand=True)
    py = readfile("data/py_160.txt", expand=True)
    pz = readfile("data/pz_160.txt", expand=True)
    data_points = np.concatenate((px, py, pz), axis=1)
    data_points = remove_duplicates(data_points)
    d = []
    cur_point = data_points[0]
    dist = 0
    for data_point in data_points:
        dist += np.linalg.norm(data_point - cur_point)
        debug = 0
        d.append(dist)
        cur_point = data_point
    d = np.array(d)

    data_points = data_points.transpose()
    full_dist = np.linspace(0, dist, 5000)
    # d = np.linspace(0, 1, len(d))

    fit_dx_eq = polyfit_np_axis(d, data_points[0], full_dist, deg)
    fit_dy_eq = polyfit_np_axis(d, data_points[1], full_dist, deg)
    fit_dz_eq = polyfit_np_axis(d, data_points[2], full_dist, deg)

    plot_data(data_points, [fit_dx_eq, fit_dy_eq, fit_dz_eq], "yes")
    debug = 0


def remove_duplicates(data):
    last_point = None
    points = []
    for point in data:
        if np.all(point == last_point):
            continue
        points.append(point)
        last_point = point
    data = np.array(points)
    return data


def polyfit_np_axis(d, data, full_dist, deg):
    fit = np.polyfit(d, data, deg)
    fit_eq = 0
    for i in range(deg):
        fit_eq += np.power(full_dist, (deg - i)) * fit[i]
    fit_eq += fit[-1]
    return fit_eq


def plot_data(data, new, title, save=False):
    # now lets plot it!

    fig = plt.figure(dpi=600)
    ax = Axes3D(fig)
    ax.grid(False)
    ax.set_facecolor('white')
    ax.plot(data[0], data[1], data[2], label='Original Global Path', lw=2, c='blue')
    ax.plot(new[0], new[1], new[2], label='Interpolated Trajectory', lw=2, c='black')
    ax.legend()
    # plt.xlim(-50, -110)
    plt.savefig(f'data/fig/{title}_plot.png')
    plt.show()
    plt.clf()

    if save:
        writefile(f"data/out/px_B_spline_interp_{title}.txt", new[0])
        writefile(f"data/out/py_B_spline_interp_{title}.txt", new[1])
        writefile(f"data/out/pz_B_spline_interp_{title}.txt", new[2])


def plot_axis(data, new, title):
    plt.figure(dpi=600)
    plt.plot(np.linspace(0, 1, len(data[0])), data[0], label=f'Original Global Path - x', lw=2, c='blue')
    plt.plot(np.linspace(0, 1, len(new[0])), new[0], label=f'Interpolated Trajectory - x', lw=2, c='black')
    plt.title('X ' + title)
    plt.savefig(f'data/fig/{title}_x.png')
    plt.clf()
    # plt.show()
    plt.figure(dpi=600)
    plt.plot(np.linspace(0, 1, len(data[1])), data[1], label=f'Original Global Path - y', lw=2, c='blue')
    plt.plot(np.linspace(0, 1, len(new[1])), new[1], label=f'Interpolated Trajectory - y', lw=2, c='black')
    plt.title('Y' + title)
    plt.savefig(f'data/fig/{title}_y.png')
    plt.clf()
    # plt.show()
    plt.figure(dpi=600)
    plt.plot(np.linspace(0, 1, len(data[2])), data[2], label=f'Original Global Path - z', lw=2, c='blue')
    plt.plot(np.linspace(0, 1, len(new[2])), new[2], label=f'Interpolated Trajectory - z', lw=2, c='black')
    plt.title('Z')
    plt.savefig(f'data/fig/{title}_z.png')
    # plt.show()
    plt.clf()


# polynomial_fit([5, 3, 2, 1], [10, 20, 30, 80])

# Save non dub list:
# px = readfile("data/px_160.txt", expand=True)
# py = readfile("data/py_160.txt", expand=True)
# pz = readfile("data/pz_160.txt", expand=True)
# data = np.concatenate((px, py), axis=1)
# data = np.concatenate((data, pz), axis=1)
# # Remove duplicates:
# last_point = None
# points = []
# for point in data:
#     if np.all(point == last_point):
#         continue
#     points.append(point)
#     last_point = point
# data = np.array(points)
# data = data.transpose()
# writefile(f"data/px_nodub_{len(data[0])}.txt", data[0])
# writefile(f"data/py_nodub_{len(data[1])}.txt", data[1])
# writefile(f"data/pz_nodub_{len(data[2])}.txt", data[2])
np_polyfit(1)

# x = 0
# y = 0
# z = 0
# d = 0
#
# deg = 1
# fit = np.polyfit(d, x, deg)
# fit_eq = 0
# for i in range(deg):
#     fit_eq += d ** (deg - i) + fit[i]
#
# fig = plt.figure()
# ax = fig.subplots()
# ax.plot(x, fit_eq, color='r', )

# poly = PolynomialFeatures(degree=3)
# X_t = poly.fit_transform(X)
#
# clf = LinearRegression()
# clf.fit(X_t, y)
# print(clf.coef_)
# print(clf.intercept_)
