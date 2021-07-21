import numpy as np
import torch


class Loss:
    def __init__(self, target, angle, coeff):
        coeff[1] = 11
        coeff[3] = 7
        self.target_np = target
        self.angle_np = angle
        self.coeff_np = coeff

        self.target = torch.tensor(target)
        self.angle = torch.tensor(angle)
        self.coeff = torch.tensor(coeff)

    def __call__(self, x):
        _x = x - self.target
        _x[0], _x[1] = (
            (np.cos(self.angle) * _x[0] - np.sin(self.angle) * _x[1]) / 3,
            (np.sin(self.angle) * _x[0] + np.cos(self.angle) * _x[1]) / 3,
        )
        return (
            self.coeff[0] * (_x[0] ** 2 + _x[1] - self.coeff[1]) ** 2
            + self.coeff[2] * (_x[0] + _x[1] ** 2 - self.coeff[3]) ** 2
        )

    def get_contour(self, x, y):
        _x = x - self.target_np[0]
        _y = y - self.target_np[1]

        _x, _y = (
            (np.cos(self.angle_np) * _x - np.sin(self.angle_np) * _y) / 3,
            (np.sin(self.angle_np) * _x + np.cos(self.angle_np) * _y) / 3,
        )
        return (
            self.coeff_np[0] * (_x ** 2 + _y - self.coeff_np[1]) ** 2
            + self.coeff_np[2] * (_x + _y ** 2 - self.coeff_np[3]) ** 2
        )
