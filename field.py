import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from abc import ABC, ABCMeta, abstractmethod
from functools import total_ordering


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

    def len(self):
        return np.sqrt(self.x ** 2 + self.y ** 2)

    def rotate(self, theta2):
        x = self.x * np.cos(theta2) - self.y * np.sin(theta2)
        y = self.x * np.sin(theta2) + self.y * np.cos(theta2)
        return Point2D(x, y, theta2 + self.theta)

    def getXY(self):
        return self.x, self.y

    def cross(self, other):  # 2次元での外積を求める.
        # \vec{a} \times \vec{b} = a_x b_y - a_y b_x
        return self.x * other.y - other.x * self.y

    def dot(self, other):
        return self.x * other.x + self.y * other.y


class Object(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def check_collision(self, point):  # pointにある点がオブジェクトと衝突していないか判定
        pass

    @abstractmethod
    def check_collision_line_segment(self, point1, point2):  # point1, point2を結ぶ線分とオブジェクトが衝突していないか判定
        pass

    @abstractmethod
    def plot(self, ax):  # 図形をmatplotlibで描画
        pass

    @abstractmethod
    def calc_min_dist_point(self, point):  # pointとの最短距離を計算する
        pass

    @abstractmethod
    def change_center(self, new_center_point):  # 中心の位置を変更する
        pass


class Circle(Object):
    def __init__(self, x, y, r, fill=True, color="black"):
        super().__init__()
        self.center = Point2D(x, y, 0.0)
        self.r = r
        self.fill = fill
        self.color = color

    def check_collision(self, point):
        if type(point) == Point2D:
            if self.fill:
                return (self.center - point).len() <= self.r
            else:
                return (self.center - point).len() == self.r
        elif type(point) == Circle:
            return (self.center - point.center).len() <= self.r
        elif type(point) == Rectangle:
            for tmp in point.vertex:
                if (tmp - self.center).len() <= self.r:
                    return True
            rect1 = Rectangle(point.center.x, point.center.y, point.w + self.r * 2.0, point.h, point.theta)
            rect2 = Rectangle(point.center.x, point.center.y, point.w, point.h + self.r * 2.0, point.theta)
            return rect1.check_collision(self.center) or rect2.check_collision(self.center)
        else:
            print("Error in check collision")
            return True

    def check_collision_line_segment(self, point1, point2):
        vec = point1 - point2
        max_dist = max((point2 - self.center).len(), (point1 - self.center).len())
        min_dist = np.abs(((self.center - point1).cross(vec)) / vec.len())
        if min_dist == self.r or max_dist == self.r:  # 円と線分が接する
            return True
        if (min_dist < self.r) ^ (max_dist < self.r):  # xor. 交差しない事を判定
            return True
        else:
            return False

    def calc_min_dist_point(self, point):  # pointとの最短距離を計算する
        return max((self.center - point).len() - self.r, 0.0)

    def plot(self, ax):
        c = patches.Circle(xy=(self.center.x, self.center.y), radius=self.r, fill=self.fill,
                           fc=self.color, ec=self.color)  # surface color, edge color
        ax.add_patch(c)
        ax.set_aspect("equal")

    def change_center(self, new_center_point):  # 中心の位置を変更する
        self.center = new_center_point


class Rectangle(Object):
    def __init__(self, x, y, w, h, theta, fill=True, color="black", obstacle=True):  # 横w, 縦h, 角度thetaの長方形
        super().__init__()
        self.center = Point2D(x, y, 0.0)
        self.theta = theta
        self.w = w
        self.h = h
        self.fill = fill
        self.color = color
        self.obstacle = obstacle
        self.vertex = [
            Point2D(w / 2.0, h / 2.0).rotate(theta) + self.center,
            Point2D(-w / 2.0, h / 2.0).rotate(theta) + self.center,
            Point2D(-w / 2.0, -h / 2.0).rotate(theta) + self.center,
            Point2D(w / 2.0, -h / 2.0).rotate(theta) + self.center,
        ]

    def check_collision(self, point):
        if not self.obstacle:
            return False

        if type(point) == Point2D:
            vec = (point - self.center).rotate(-self.theta)
            if self.fill:
                return np.abs(vec.x) <= self.w / 2.0 and np.abs(vec.y) <= self.h / 2.0
            else:
                return np.abs(vec.x) == self.w / 2.0 and np.abs(vec.y) == self.h / 2.0
        elif type(point) == Circle:
            for tmp in self.vertex:
                if (tmp - point.center).len() <= self.vertex:
                    return True
            rect1 = Rectangle(self.center.x, self.center.y, self.w + point.r * 2.0, self.h, self.theta)
            rect2 = Rectangle(self.center.x, self.center.y, self.w, self.h + point.r * 2.0, self.theta)
            return rect1.check_collision(point.center) or rect2.check_collision(point.center)
        elif type(point) == Rectangle:
            for vert in point.vertex:
                vec = (vert - self.center).rotate(-self.theta)
                if np.abs(vec.x) == self.w / 2.0 and np.abs(vec.y) == self.h / 2.0:
                    return True
            return False
        else:
            print("Error in check collision")
            return True

    def check_collision_line_segment(self, point1, point2):
        def check_line_segment(point_a1, point_a2, point_b1, point_b2):  # 外積による線分同士の交差判定; Trueなら交差してる
            vec = point_a2 - point_a1
            if vec.cross(point_b1 - point_a1) * vec.cross(point_b2 - point_a1) > 0:
                return False
            vec = point_b2 - point_b1
            if vec.cross(point_a1 - point_b1) * vec.cross(point_a2 - point_b1) > 0:
                return False
            return True

        return (check_line_segment(point1, point2, self.vertex[0], self.vertex[1]) or
                check_line_segment(point1, point2, self.vertex[1], self.vertex[2]) or
                check_line_segment(point1, point2, self.vertex[2], self.vertex[3]) or
                check_line_segment(point1, point2, self.vertex[3], self.vertex[0]))

    def plot(self, ax, non_fill=False):
        vec = Point2D(-self.w / 2.0, -self.h / 2.0).rotate(self.theta) + self.center
        c = patches.Rectangle(xy=(vec.x, vec.y), width=self.w, height=self.h, angle=math.degrees(self.theta),
                              fill=(self.fill and (not non_fill)), fc=self.color, ec=self.color)
        ax.add_patch(c)
        ax.set_aspect("equal")

    def calc_min_dist_point(self, point):  # pointとの最短距離を計算する
        vec = (point - self.center).rotate(-self.theta)  # 回転した座標系でのcenterから見たpointの位置
        if np.abs(vec.x) <= self.w / 2.0:
            return max(abs(vec.y) - self.h / 2.0, 0.0)
        elif np.abs(vec.y) <= self.h / 2.0:
            return max(abs(vec.x) - self.w / 2.0, 0.0)
        else:
            return min([(vert - point).len() for vert in self.vertex])

    def change_center(self, new_center_point):  # 中心の位置を変更する
        for vert in self.vertex:
            vert += new_center_point - self.center
        self.center = new_center_point


class Field:
    def __init__(self, w, h):
        self.obstacles = []
        self.frame = Rectangle(w / 2.0, h / 2.0, w, h, 0.0, fill=True, color="black")

    def check_collision(self, point):
        if not self.frame.check_collision(point):
            return True
        for obs in self.obstacles:
            if obs.check_collision(point):
                return True
        return False

    def check_collision_line_segment(self, point1, point2):
        if not self.frame.check_collision(point1) or not self.frame.check_collision(point2):
            return True
        for obs in self.obstacles:
            if obs.check_collision_line_segment(point1, point2):
                return True
        return False

    def plot_field(self):
        ax = plt.axes()
        self.frame.plot(ax, non_fill=True)
        for tmp in self.obstacles:
            tmp.plot(ax)
        plt.axis("scaled")
        ax.grid(which="major", axis="x", color="blue", alpha=0.4, linestyle="--", linewidth=1)
        ax.grid(which="major", axis="y", color="blue", alpha=0.4, linestyle="--", linewidth=1)
        return ax

    def plot(self):
        plt.cla()
        ax = self.plot_field()
        plt.show()
        return ax

    def plot_path(self, path, start_point=None, target_point=None, ax=None, show=True):
        if ax is None:
            ax = self.plot_field()
        if start_point is not None:
            ax.plot(start_point.x, start_point.y, color="green", marker="x", markersize=10.0)
        if target_point is not None:
            ax.plot(target_point.x, target_point.y, color="red", marker="x", markersize=10.0)
        for point in path:
            ax.plot(point.x, point.y, color="red", marker='.', markersize=1.0)
        for i in range(len(path) - 1):
            p1, p2 = path[i].getXY(), path[i + 1].getXY()
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="grey")
        plt.tight_layout()
        if show:
            plt.show()
        return ax

    def plot_path_control_point(self, path, ctrl_points=None, ax=None, show=True, lined=True):
        if ax is None:
            ax = self.plot_field()
        if ctrl_points is not None:
            for i in range(len(ctrl_points)):
                if i == len(ctrl_points)-1:
                    ax.plot(ctrl_points[i].x, ctrl_points[i].y, color="red", marker="x", markersize=10.0)
                else:
                    ax.plot(ctrl_points[i].x, ctrl_points[i].y, color="green", marker="x", markersize=10.0)
        for point in path:
            ax.plot(point.x, point.y, color="red", marker='.', markersize=1.0)
        if lined:
            for i in range(len(path) - 1):
                p1, p2 = path[i].getXY(), path[i + 1].getXY()
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="grey")
        plt.tight_layout()
        if show:
            plt.show()
        return ax

    # def plot_anime(self, path, start_point, target_point, global_path=None):
    #     plt.cla()
    #     ax = self.plot_field()
    #     ax.plot(start_point.x, start_point.y, color="green", marker="x", markersize=10.0)
    #     ax.plot(target_point.x, target_point.y, color="red", marker="x", markersize=10.0)
    #     if global_path is not None:
    #         for i in range(len(global_path) - 1):
    #             p1, p2 = global_path[i].getXY(), global_path[i + 1].getXY()
    #             ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="grey")
    #     for i in range(len(path)-1):
    #         p1, p2 = path[i].getXY(), path[i + 1].getXY()
    #         ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="blue")
    #     plt.pause(0.001)

    def plot_anime(self, path, start_point, target_point, global_path=None, predict_trajectory=None):
        ax = plt.axes()
        plt.axis("off")
        ax.axis("off")
        self.frame.plot(ax, non_fill=True)
        for tmp in self.obstacles:
            tmp.plot(ax)
        ax.grid(which="major", axis="x", color="blue", alpha=0.4, linestyle="--", linewidth=1)
        ax.grid(which="major", axis="y", color="blue", alpha=0.4, linestyle="--", linewidth=1)

        ax.plot(start_point.x, start_point.y, color="green", marker="x", markersize=10.0)
        ax.plot(target_point.x, target_point.y, color="red", marker="x", markersize=10.0)
        if global_path is not None:
            for i in range(len(global_path) - 1):
                p1, p2 = global_path[i].getXY(), global_path[i + 1].getXY()
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="grey")
        if predict_trajectory is not None:
            for i in range(len(predict_trajectory) - 1):
                p1, p2 = predict_trajectory[i].getXY(), predict_trajectory[i + 1].getXY()
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="cyan")
        for i in range(len(path)-1):
            p1, p2 = path[i].getXY(), path[i + 1].getXY()
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="red")
        ax.plot(path[-1].x, path[-1].y, color="red", marker="x", markersize=8.0)

        ax.set_aspect('equal')
        plt.draw()
        plt.pause(0.001)
        ax.clear()

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)


if __name__ == '__main__':
    field = Field(12, 12)
    field.add_obstacle(Circle(2, 4, 1, True))
    field.add_obstacle(Rectangle(5, 2, 1, 3, np.pi / 4.0, True))
    field.plot()
