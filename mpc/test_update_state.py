import unittest
import numpy as np

from utils import _differential_drive_update_state


class TestDifferentialDriveUpdateState(unittest.TestCase):

    def test_update_state_with_zero_velocities(self):
        current_state = np.array([[0], [0], [0]], dtype=np.float64)
        linear_velocity = 0.0
        angular_velocity = 0.0
        dt = 1.0
        expected_state = current_state
        updated_state = _differential_drive_update_state(current_state, linear_velocity, angular_velocity, dt)
        np.testing.assert_allclose(updated_state, expected_state)

    def test_update_state_with_nonzero_velocities(self):
        current_state = np.array([[0],
                                  [0],
                                  [0]], dtype=np.float64)
        dts = [0.1, 0.2, 0.5, 1.0]  # different time steps

        linear_velocities = [1.0, 2.0, 3.0]  # different linear velocities

        angular_velocities = [np.pi / 4, np.pi / 2, np.pi]  # different angular velocities
        for dt in dts:
            for linear_velocity in linear_velocities:
                for angular_velocity in angular_velocities:
                    expected_state = self.calculate_expected_state(current_state, linear_velocity, angular_velocity, dt)
                    updated_state = _differential_drive_update_state(current_state, linear_velocity, angular_velocity,
                                                                     dt)
                    print("updated_state: {}".format(updated_state), "expected_state: {}".format(expected_state))
                    np.testing.assert_allclose(updated_state, expected_state, rtol=1e-5)

    @staticmethod
    def calculate_expected_state(current_state, linear_velocity, angular_velocity, dt):
        x, y, theta = current_state.flatten()
        delta_x = dt * linear_velocity * np.cos(theta * dt)
        delta_y = dt * linear_velocity * np.sin(theta * dt)
        x += delta_x
        y += delta_y
        theta += dt * angular_velocity
        return np.array([[x], [y], [theta]], dtype=np.float64)


if __name__ == '__main__':
    unittest.main()
