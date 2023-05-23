import numpy as np
from numba import njit

from utils import linear_to_angular_velocity, _differential_drive_inverse_kinematics, \
    _differential_drive_forward_kinematics, _differential_drive_update_state, my_clip


class AAGV:
    def __init__(self, unique_id, model, agv_config: dict):
        self.unique_id = unique_id
        self.work_state = 'idle'
        self.battery: float = agv_config['max_battery']
        self.max_battery: float = agv_config['max_battery']
        self.task = None
        self.path: list = []
        self._goal = None
        self.pos = None
        self.robot_state = np.array([
            [0.],  # x position
            [0.],  # y position
            [0.],  # orientation
        ])
        self.radius = agv_config['radius']  # Radius of the AGV (m)

        # Robot specific
        self.max_speed = agv_config[
            'max_speed']  # Maximum speed of the AGV (pixel_size / timestep) (1 pixel = 1m and timestep = 0.1s = 10 m/s)
        self.max_wheel_speed = linear_to_angular_velocity(self.max_speed, agv_config['wheel_radius'])

        self.max_acceleration = agv_config[
            'max_acceleration']  # Maximum acceleration of the AGV (pixel_size / timestep^2)
        self.max_deceleration = agv_config[
            'max_deceleration']  # Maximum deceleration of the AGV (pixel_size / timestep^2)
        self.wheel_radius = agv_config['wheel_radius']  # Radius of the wheels (m)
        self.wheel_distance = agv_config['wheel_distance']  # Distance between the wheels (m)
        self.x_dot = np.array([
            [0.],  # Linear velocity (m/s)
            [0.],
            [0.],  # Angular velocity (rad/s)
        ])
        self.wheel_speed = np.array([
            [0.],  # Right wheel speed (rad/s)
            [0.],  # Left wheel speed (rad/s)
        ])
        # Movement cost for each step
        self.movement_cost = agv_config['move_cost']
        self.wait_cost = agv_config['move_cost']
        self.idle_cost = agv_config['idle_cost']
        self.collided = False

        # MPC params
        # self.mpc_control = mpc_controller
        self.horizon = 5
        self.R = np.diag([0.01, 0.01])  # input cost matrix
        self.R = np.diag([0.0, 0.0])  # input cost matrix
        self.Rd = np.diag([0.01, 0.01])  # input difference cost matrix
        self.Q = np.diag([2.0, 2.0, 1.])  # state cost matrix
        self.Qf = self.Q  # state final matrix

    @property
    def goal(self):
        if self._goal is None:
            return self.pos
        return self._goal

    @goal.setter
    def goal(self, value: tuple):
        self._goal = value

    @property
    def orientation(self):
        return self.robot_state[2, 0]

    @orientation.setter
    def orientation(self, value: float):
        self.robot_state[2, 0] = value

    @property
    def x(self):
        return self.robot_state[0, 0]

    @x.setter
    def x(self, value: float):
        self.robot_state[0, 0] = value

    @property
    def y(self):
        return self.robot_state[1, 0]

    @y.setter
    def y(self, value: float):
        self.robot_state[1, 0] = value

    @property
    def linear_velocity(self):
        return self.x_dot[0, 0]

    @linear_velocity.setter
    def linear_velocity(self, value: float):
        self.x_dot[0, 0] = value

    @property
    def angular_velocity(self):
        return self.x_dot[2, 0]

    @angular_velocity.setter
    def angular_velocity(self, value: float):
        self.x_dot[2, 0] = value

    @property
    def rw_speed(self):
        return self.wheel_speed[0, 0]

    @rw_speed.setter
    def rw_speed(self, value: float):
        self.wheel_speed[0, 0] = value

    @property
    def lw_speed(self):
        return self.wheel_speed[1, 0]

    @lw_speed.setter
    def lw_speed(self, value: float):
        self.wheel_speed[1, 0] = value

    def step(self):
        if self.work_state == 'idle':
            self.random_move()
        elif self.work_state == 'moving':
            if self.pos == self.goal:
                self.work_state = 'waiting'
                self.path.pop(0)
                if len(self.path) > 0:
                    self.goal = self.path[0]
                else:
                    self.goal = None
            else:
                self.move_towards(self.goal)
        elif self.work_state == 'waiting':
            self.random_move()
        elif self.work_state == 'charging':
            self.random_move()
        elif self.work_state == 'recharging':
            self.battery = self.max_battery
            self.work_state = 'idle'
        else:
            raise Exception("Unknown state: {}".format(self.work_state))

    # @profile
    def update(self, dt):
        self.wheel_speed = my_clip(self.wheel_speed, -self.max_wheel_speed, self.max_wheel_speed)

        v, omega = self.forward_kinematics()
        self.linear_velocity = v
        self.angular_velocity = omega

        self.update_state(dt)

        w_left, w_right = self.inverse_kinematics()
        self.lw_speed = w_left
        self.rw_speed = w_right

    def update_state(self, dt):
        new_state = _differential_drive_update_state(self.robot_state, self.linear_velocity,
                                                     self.angular_velocity, dt)
        self.robot_state = new_state

    def set_wheel_velocity(self, lw_speed, rw_speed):
        self.rw_speed = rw_speed
        self.lw_speed = lw_speed

        v, omega = self.forward_kinematics()
        self.linear_velocity = v
        self.angular_velocity = omega

    def set_robot_velocity(self, linear_velocity, angular_velocity):
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity

        w_left, w_right = self.inverse_kinematics()
        self.lw_speed = w_left
        self.rw_speed = w_right

    def inverse_kinematics(self):
        return _differential_drive_inverse_kinematics(self.linear_velocity, self.angular_velocity, self.wheel_radius,
                                                      self.wheel_distance)

    def forward_kinematics(self):
        return _differential_drive_forward_kinematics(self.lw_speed, self.rw_speed, self.wheel_radius,
                                                      self.wheel_distance)
