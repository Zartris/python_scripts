"""
Adapted from trimesh example raytrace.py
----------------
Install `pyembree` for a speedup (600k+ rays per second)
"""
from scipy.spatial.transform.rotation import Rotation
from stl import mesh

from Visualizer.Display3D import Display3D
from objects.mesh_base import Wireframe
import numpy as np

from polynomial_fitting import plot_axis
from tranformation.transform import TRotation, Transform


def readfile_multiple_values(filename, split_char=" "):
    file1 = open(filename, 'r')
    Lines = file1.readlines()
    offset = np.array([-68, 32, 70])
    rotaion_offset = 180
    count = 0
    # Strips the newline character
    lines = []
    poses = []
    yaws = []
    x_refs = []
    ts = []
    for line in Lines:
        line = line.strip()
        l_values = line.split(split_char)
        poses.append(np.array(l_values[4:7]).astype(float) + offset)
        yaw = np.array([l_values[18]]).astype(float) + rotaion_offset
        yaw = -yaw
        yaws.append(yaw)
        x_refs.append(np.array(l_values[1:4]).astype(float) + offset)
        ts.append(np.array([l_values[0]]).astype(float))
    lines = [np.array(ts), np.array(poses), np.array(yaws), np.array(x_refs)]
    return lines


if __name__ == '__main__':
    view = Display3D(1280, 720)
    wt = Wireframe.from_stl_path('data/in/turbine_v2.stl')
    r = Rotation.from_euler("XYZ", [0, 0, 90], degrees=True).as_matrix()

    r = TRotation().set_matrix(r, "XYZ")
    t = Transform(np.expand_dims(np.array([-80, 0, 20]), axis=1), r,
                  translate_before_rotate=False)

    wt = wt.transform(t)
    cube = Wireframe.from_stl_path('data/in/camera.stl')
    view.add_object("windturbine", wt)
    view.add_object("camera", cube)
    data = None
    data = readfile_multiple_values("data/in/GT_traj.txt", ",")
    # plot_axis(data[1].transpose(),data[1].transpose(),"axis")
    view.run(data)
