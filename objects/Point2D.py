import numpy as np
from functools import total_ordering
import copy

@total_ordering
class Point2D:  # x, y, theta
    def __init__(self, x, y, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta

    def __eq__(self, other):  # ==
        return self.x == other.x and self.y == other.y and self.theta == other.theta

    def __lt__(self, other):  # <
        if self.x == other.x:
            if self.y == other.y:
                return self.theta < other.theta
            else:
                return self.y < other.y
        else:
            return self.x < other.x

    def __add__(self, other):  # +
        x = self.x + other.x
        y = self.y + other.y
        theta = self.theta + other.theta
        return Point2D(x, y, theta)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        theta = self.theta - other.theta
        return Point2D(x, y, theta)

    def __mul__(self, other):
        x = self.x * other
        y = self.y * other
        return Point2D(x, y, self.theta)

    __rmul__ = __mul__

    def __repr__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.theta) + ")"

    def len(self):
        return np.sqrt(self.x ** 2 + self.y ** 2)

    def rotate(self, theta2):
        x = self.x * np.cos(theta2) - self.y * np.sin(theta2)
        y = self.x * np.sin(theta2) + self.y * np.cos(theta2)
        return Point2D(x, y, theta2 + self.theta)

    def unit(self):
        return copy.deepcopy(self) * (1.0 / self.len())

    def getXY(self):
        return self.x, self.y

    def cross(self, other):  # 2次元での外積を求める.
        # \vec{a} \times \vec{b} = a_x b_y - a_y b_x
        return self.x * other.y - other.x * self.y

    def dot(self, other):
        return self.x * other.x + self.y * other.y

