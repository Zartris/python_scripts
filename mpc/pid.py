import numpy as np
import numba
from numba import njit, prange

from utils import norm_2d


@njit(fastmath=True)
def pid_controller(Kp: float, Ki: float, Kd: float, desired_position: np.ndarray,
                   current_states: np.ndarray, prev_errors: np.ndarray, integral_errors: np.ndarray):
    position_error = desired_position[:, :2] - current_states[:, :2, 0]
    # distance_error = np.linalg.norm(position_error, axis=1)
    distance_error = norm_2d(position_error)

    integral_errors += distance_error
    derivative_errors = (distance_error - prev_errors)
    desired_velocity = Kp * distance_error + Ki * integral_errors + Kd * derivative_errors
    return desired_velocity, distance_error, integral_errors


@njit(fastmath=True)
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
