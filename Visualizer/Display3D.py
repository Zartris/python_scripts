import pygame, math
import numpy as np

# Radian rotated by a key event
from scipy.spatial.transform.rotation import Rotation

from Visualizer.utils.colors import *
from objects.mesh_base import Wireframe
from tranformation.camera import CameraInfo
from tranformation.transform import TRotation, Transform


class Display3D:
    """ A group of objects which can be displayed on a Pygame screen """

    def __init__(self, width, height, name="Display3D"):
        super().__init__()
        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()

        self.screen = pygame.display.set_mode((width, height))
        self.screen_center = (int(width / 2), int(height / 2))
        pygame.display.set_caption(name)

        self.objects = {}
        self.object_colors = {}
        self.object_to_update = []

        self.displayNodes = False
        self.displayEdges = False
        self.displayFaces = True
        self.displayFPS = True
        self.displaceTraj = True
        self.displayAxis = True

        # self.perspective = False  # 300.

        # CameraInfo -> Zenmuse x3 (dji camera) f= 22 or 35 - https://www.dji.com/dk/zenmuse-x3/info
        self.camera = CameraInfo(focal=22, FOV=94, image_width=1280, image_height=720)
        self.camera_translation = np.array([-80, 0, 80]).astype(float)
        self.camera_rpy = np.array([0., 0., 0.])
        self.camera_view_dir = np.array([1., 0., 0.])
        self.orbit_center = np.array([0, 0, 80]).astype(float)
        self.use_orbit = False
        t, _ = self.compute_transform(self.camera_rpy, self.camera_translation, not self.use_orbit)
        self.camera.update_transform(t)

        self.third_person = True
        self.camera_obj_translation = np.array([-80, 0, 80]).astype(float)
        self.camera_obj_rpy = np.array([0., 0., 0.])

        # Lighting
        # self.use_light = False
        # self.light = Wireframe()
        # self.light.add_nodes([[0, -1, 0]])
        # self.min_light = 0.02
        # self.max_light = 1.0
        # self.light_range = self.max_light - self.min_light

        self.background = BLACK
        self.nodeColour = (250, 250, 250)
        self.nodeRadius = 4
        self.control = 0

        self.left_mouse = False
        self.right_mouse = False
        self.keys_down = []

        self.increment_speed = 5
        self.pause = False
        pygame.init()

    def add_object(self, name, obj, color=WHITE):
        self.objects[name] = obj
        #   If colour is set to None, then wireframe is not displayed
        self.object_colors[name] = color

    def add_object_group(self, object_group):
        # Potential danger of overwriting names
        for name, obj in object_group.wireframes.items():
            self.add_object(name, obj)

    @staticmethod
    def segments(p):
        return zip(p, p[1:] + [p[0]])

    def area(self, p):
        return 0.5 * abs(sum(x0 * y1 - x1 * y0
                             for ((x0, y0), (x1, y1)) in self.segments(p)))

    @staticmethod
    def compute_transform(rpy_in_from, pos_in_from, translate_before_rotate):
        R = Rotation.from_euler("XYZ", rpy_in_from, degrees=True).as_matrix()
        rotate = TRotation().set_matrix(R, "XYZ")
        T_from2to = Transform(np.expand_dims(-pos_in_from, axis=1), rotate,
                              translate_before_rotate=translate_before_rotate)
        T_to2from = T_from2to.inverse()
        return T_from2to, T_to2from

    def show_fps(self):
        fps_text = str(int(self.clock.get_fps()))
        fps_surface = pygame.font.SysFont("Arial", 20).render(fps_text, True, pygame.Color("green"))
        self.screen.blit(fps_surface, (self.width - 30, 10))

    def display(self, verbose=False):
        self.screen.fill(self.background)
        transforms = []
        r = Rotation.from_euler("XYZ", self.camera_rpy, degrees=True).as_matrix()
        if self.use_orbit:
            # Transform to orbit_center
            orbit_rotation = TRotation().set_matrix(r, "XYZ")
            orbit_transform = Transform(np.expand_dims(self.orbit_center, axis=1), orbit_rotation,
                                        translate_before_rotate=False)
            transforms.append(orbit_transform)
            camera_transform = Transform(np.expand_dims(-self.camera_translation, axis=1))
            transforms.append(camera_transform)
        else:
            camera_rotation = TRotation().set_matrix(r, "XYZ")
            camera_transform = Transform(np.expand_dims(-self.camera_translation, axis=1), camera_rotation,
                                         translate_before_rotate=True)
            transforms.append(camera_transform)
        self.camera.update_transform(camera_transform)

        if self.displayAxis:
            p0 = np.array([0, 0, 0])
            px = np.array([100, 0, 0])
            py = np.array([0, 100, 0])
            pz = np.array([0, 0, 100])
            colors = [RED, GREEN, BLUE]
            p0_im = self.camera.transform_world_to_image(p0)
            for c, p in zip(colors, [px, py, pz]):
                p_im = self.camera.transform_world_to_image(p)
                pygame.draw.line(self.screen, c, p0_im, p_im, width=5)

        counter = 0
        for name, obj in self.objects.items():
            o: Wireframe = obj
            if name == "camera":
                if self.third_person:
                    r = Rotation.from_euler("XYZ", self.camera_obj_rpy, degrees=True).as_matrix()

                    cam_r = TRotation().set_matrix(r, "XYZ")
                    cam_t = Transform(np.expand_dims(self.camera_obj_translation, axis=1), cam_r,
                                      translate_before_rotate=False)
                    o = o.transform(cam_t)
                else:
                    continue

            for t in transforms:
                o = o.transform(t)
            if self.displayFaces:
                for index in o.sorted_vertices_ind():
                    vertice = o.vertices[index]
                    if self.camera_view_dir[0] == 1:
                        if vertice[:, 0].max() > 0:
                            if verbose:
                                print("behind me!")
                            continue
                    else:
                        if vertice[:, 0].min() < 0:
                            if verbose:
                                print("behind me!")
                            continue
                    normal = o.normals[index]
                    color = o.vertice_colors[index]

                    towards_us = np.dot(normal, self.camera_view_dir)

                    # Only draw faces that face us
                    if towards_us > 0:
                        points_im = []
                        x = 0
                        y = 0
                        for point in vertice:
                            depht = 0
                            u, v = self.camera.transform_cam_to_image(point)
                            points_im.append((u, v))
                            x += np.clip(u / self.camera.image_width, -1, 1)
                            y += np.clip(v / self.camera.image_height, -1, 1)
                        if abs(x) == 3 or abs(y) == 3:
                            debug = 0
                            continue
                        a = self.area(points_im)
                        if a >= 10000000:
                            print(f"skipped by area: {a}")
                            continue
                        counter += 1
                        if verbose:
                            print(f"drawing: {counter}")
                        pygame.draw.polygon(self.screen, color, points_im, 0)

            # if self.displayEdges:
            #     for (n1, n2) in wireframe.edges:
            #         if self.perspective:
            #             if wireframe.nodes[n1][2] > -self.perspective and nodes[n2][2] > -self.perspective:
            #                 z1 = self.perspective / (self.perspective + nodes[n1][2])
            #                 x1 = self.width / 2 + z1 * (nodes[n1][0] - self.width / 2)
            #                 y1 = self.height / 2 + z1 * (nodes[n1][1] - self.height / 2)
            #
            #                 z2 = self.perspective / (self.perspective + nodes[n2][2])
            #                 x2 = self.width / 2 + z2 * (nodes[n2][0] - self.width / 2)
            #                 y2 = self.height / 2 + z2 * (nodes[n2][1] - self.height / 2)
            #
            #                 pygame.draw.aaline(self.screen, colour, (x1, y1), (x2, y2), 1)
            #         else:
            #             pygame.draw.aaline(self.screen, colour, (nodes[n1][0], nodes[n1][1]),
            #                                (nodes[n2][0], nodes[n2][1]), 1)
            #
            # if self.displayNodes:
            #     for node in nodes:
            #         pygame.draw.circle(self.screen, colour, (int(node[0]), int(node[1])), self.nodeRadius, 0)
        if self.displaceTraj:
            last_point = None
            for i, p in enumerate(self.traj[1]):
                if i % 100 == 0 or i == 0:
                    p_im = self.camera.transform_world_to_image(p)
                    if i == 0:
                        last_point = p_im
                        continue
                    pygame.draw.line(self.screen, RED, last_point, p_im, width=5)
                    last_point = p_im

        if self.displayFPS:
            self.show_fps()
        # pygame.display.flip()
        pygame.display.update()

    def run(self, traj=None):
        """ Display wireframe on screen and respond to keydown events """
        running = True
        key_down = False
        index = 0
        self.displaceTraj = self.displaceTraj if traj is not None else False
        self.traj = traj
        while running:
            self.clock.tick(50)
            if traj is None:
                self.handle_user_input()
                self.camera_rpy[1] = 0
            else:
                self.handle_user_input()
                if self.third_person:
                    self.camera_obj_translation = traj[1][index]
                    self.camera_obj_rpy = np.array([0, 0, traj[2][index][0]])
                    if index % 10 == 0:
                        print(f"STATE: i: {index} t: {self.camera_obj_translation}"
                              f" roll: {self.camera_obj_rpy[0]}, pitch: {self.camera_obj_rpy[1]}, yaw:{self.camera_obj_rpy[2]}")
                else:
                    self.camera_translation = traj[1][index]
                    self.camera_rpy = np.array([0, 0, traj[2][index][0]])
                    if index % 10 == 0:
                        print(f"STATE: i: {index} t: {self.camera_translation}"
                              f" roll: {self.camera_rpy[0]}, pitch: {self.camera_rpy[1]}, yaw:{self.camera_rpy[2]}")
                if index == 1720:
                    debug = 0
            # self.update_objects()
            self.display(index == 1720)
            if not self.pause:
                index += self.increment_speed
        pygame.quit()

    def handle_user_input(self):
        change = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                self.keys_down.append(event.key)
            elif event.type == pygame.KEYUP:
                self.keys_down.remove(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                change = True
                self.handle_mouse_click_input(event, key_down=True)
            elif event.type == pygame.MOUSEBUTTONUP:
                change = True
                self.handle_mouse_click_input(event, key_down=False)
            elif event.type == pygame.MOUSEMOTION:
                if self.right_mouse or self.left_mouse:
                    change = True
                    self.handle_mouse_drag_input(event)
                # Compute tranform of camera:
        for key in self.keys_down:
            change = True
            self.handle_key_input(key)
        if change:
            print(f"STATE: t: {self.camera_translation}"
                  f" roll: {self.camera_rpy[0]}, pitch: {self.camera_rpy[1]}, yaw:{self.camera_rpy[2]}")

    def handle_key_input(self, key):
        if key == pygame.K_ESCAPE:
            pygame.quit()
            exit()
        elif key == pygame.K_DOWN or key == ord('s'):
            translate = self.camera.t_c2w(np.array([-1, 0, 0])).squeeze()
            self.camera_translation = translate
        elif key == pygame.K_UP or key == ord('w'):
            translate = self.camera.t_c2w(np.array([1, 0, 0])).squeeze()
            self.camera_translation = translate
        elif key == pygame.K_LEFT or key == ord('a'):
            translate = self.camera.t_c2w(np.array([0, 1, 0])).squeeze()
            self.camera_translation = translate
        elif key == pygame.K_RIGHT or key == ord('d'):
            translate = self.camera.t_c2w(np.array([0, -1, 0])).squeeze()
            self.camera_translation = translate
        elif key == ord('q'):
            self.camera_rpy[2] += 90
        elif key == ord('e'):
            self.camera_rpy[2] -= 90
        elif key == ord('f'):
            self.use_orbit = not self.use_orbit
            print(f"orbit mode: {self.use_orbit}")
        elif key == ord('t'):
            self.increment_speed += 1
            print(f"speed: {self.increment_speed}")
        elif key == pygame.K_SPACE:
            self.increment_speed = 0
        elif key == ord('y'):
            self.increment_speed -= 1
            print(f"speed: {self.increment_speed}")
        elif key == pygame.K_1:
            self.third_person = True
        elif key == pygame.K_2:
            self.third_person = False

    def handle_mouse_click_input(self, event, key_down: bool):
        if event.button == 1:  # left click
            # Record starting pos
            self.right_mouse = key_down
            pygame.mouse.set_visible(not key_down)
        elif event.button == 3:  # right click
            self.left_mouse = key_down
            pygame.mouse.set_visible(not key_down)
        elif event.button == 4 and key_down:  # Scroll forward
            translate = self.camera.t_c2w(np.array([-1, 0, 0])).squeeze()
            print(f"B camera_translation: {self.camera_translation}, A camera_translation {translate}")
            self.camera_translation = translate
        elif event.button == 5 and key_down:  # Scroll forward
            translate = self.camera.t_c2w(np.array([1, 0, 0])).squeeze()
            print(f"B camera_translation: {self.camera_translation}, A camera_translation {translate}")
            if self.use_orbit:
                self.camera_translation = np.array([translate[0], 0, 0])
            else:
                self.camera_translation = translate

    def handle_mouse_drag_input(self, event):
        new_mouse_x, new_mouse_y = event.pos
        move_x = np.clip(new_mouse_x - self.screen_center[0], -2, 2)
        move_y = np.clip(new_mouse_y - self.screen_center[1], -2, 2)
        if self.right_mouse:
            translate = self.camera.t_c2w(np.array([0, move_x / 5, move_y / 5])).squeeze()
            self.camera_translation = translate
        if self.left_mouse:
            # target = camera.transform_to_camera(np.array([0, 0, 0]))
            # target = np.squeeze(target)[:-1]
            speed = 10
            if self.use_orbit:
                speed = 2
            self.camera_rpy[2] -= move_x / speed
            self.camera_rpy[1] -= move_y / speed

            # camera_translation += np.array([0, move_x / 100, move_y / 100])
            # roll, pitch, yaw = lookat(camera_translation, np.array([0, 0, 0]))
            # roll, pitch, yaw = lookat(np.array([0, 0, 0]), target)
            # print(f"t: {camera_translation} roll: {roll}, pitch: {pitch}, yaw:{yaw}")
        pygame.mouse.set_pos(self.screen_center)
