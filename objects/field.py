import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from abc import ABC, ABCMeta, abstractmethod
import sys
import os

sys.path.append(os.path.dirname(__file__))
from .Point2D import Point2D


class Object(ABC):
    pos: Point2D
    fill: bool
    color: str

    def __init__(self, pos: Point2D, fill: bool, color: str) -> None:
        self.pos = pos
        self.fill = fill
        self.color = color

    @abstractmethod
    def check_collision(self, obstacle) -> bool:  # obstacleと衝突していないか判定
        pass

    @abstractmethod
    def check_collision_line_segment(self, point1: Point2D, point2: Point2D) -> bool:
        # point1, point2を結ぶ線分とオブジェクトが衝突していないか判定
        pass

    @abstractmethod
    def plot(self, ax):  # 図形をmatplotlibで描画
        pass

    @abstractmethod
    def calc_min_dist_point(self, point: Point2D) -> float:  # pointとの最短距離を計算する
        pass

    @abstractmethod
    def change_pos(self, new_pos: Point2D):  # 中心の位置を変更する
        pass


class Circle(Object):
    def __init__(self, x, y, r, fill=True, color="black"):
        super().__init__(Point2D(x, y, 0.0), fill, color)
        self.r = r

    def check_collision(self, obstacle) -> bool:
        if type(obstacle) == Point2D:
            if self.fill:
                return (self.pos - obstacle).len() <= self.r
            else:
                return (self.pos - obstacle).len() == self.r
        elif type(obstacle) == Circle:
            return (self.pos - obstacle.pos).len() <= (self.r + obstacle.r)
        elif type(obstacle) == Rectangle:
            for tmp in obstacle.get_vertex():
                if (tmp - self.pos).len() <= self.r:
                    return True
            rect1 = Rectangle(obstacle.pos.x, obstacle.pos.y, obstacle.w + self.r * 2.0, obstacle.h, obstacle.theta)
            rect2 = Rectangle(obstacle.pos.x, obstacle.pos.y, obstacle.w, obstacle.h + self.r * 2.0, obstacle.theta)
            return rect1.check_collision(self.pos) or rect2.check_collision(self.pos)
        else:
            print("Error in check collision")
            return True

    def check_collision_line_segment(self, point1: Point2D, point2: Point2D) -> bool:
        vec = point2 - point1
        max_dist = max((point2 - self.pos).len(), (point1 - self.pos).len())
        min_dist = np.abs((vec.cross(self.pos - point1)) / vec.len())
        # TODO : 円と線分の距離の計算
        if min_dist == self.r or max_dist == self.r:  # 円と線分が接する
            return True
        if self.fill:
            if min_dist <= self.r:
                if (self.pos - point2).dot(vec) * (self.pos - point1).dot(vec) < 0.0:
                    return True
                else:
                    return False
        if not self.fill:
            if (min_dist < self.r) ^ (max_dist < self.r):  # xor. 交差しない事を判定
                return True
        return False

    def calc_min_dist_point(self, point: Point2D) -> float:  # pointとの最短距離を計算する
        return max(abs((self.pos - point).len() - self.r), 0.0)

    def plot(self, ax):
        c = patches.Circle(xy=(self.pos.x, self.pos.y), radius=self.r, fill=self.fill,
                           fc=self.color, ec=self.color)  # surface color, edge color
        ax.add_patch(c)
        ax.set_aspect("equal")

    def change_pos(self, new_pos: Point2D) -> None:  # 中心の位置を変更する
        self.pos = new_pos


class Rectangle(Object):
    def __init__(self, x, y, w, h, theta, fill=True, color="black", obstacle=True):  # 横w, 縦h, 角度thetaの長方形
        super().__init__(Point2D(x, y, theta), fill, color)
        self.theta = theta
        self.w = w
        self.h = h
        self.obstacle = obstacle
        self._vertex = [
            Point2D(w / 2.0, h / 2.0),
            Point2D(-w / 2.0, h / 2.0),
            Point2D(-w / 2.0, -h / 2.0),
            Point2D(w / 2.0, -h / 2.0),
        ]

    def _change_coordinate(self, pos, vertex: list[Point2D]) -> iter:
        return iter(map(lambda x: x.rotate(pos.theta) + pos, vertex))

    def get_vertex(self):
        return list(self._change_coordinate(self.pos, self._vertex))

    def check_collision(self, obstacle) -> bool:
        if not self.obstacle:
            return False

        if type(obstacle) == Point2D:
            vec = (obstacle - self.pos).rotate(-self.theta)
            if self.fill:
                return np.abs(vec.x) <= self.w / 2.0 and np.abs(vec.y) <= self.h / 2.0
            else:
                return np.abs(vec.x) == self.w / 2.0 and np.abs(vec.y) == self.h / 2.0
        elif type(obstacle) == Circle:
            vertex = self.get_vertex()
            for i in range(len(vertex)):
                if obstacle.check_collision_line_segment(vertex[i], vertex[(i + 1) % len(vertex)]):
                    return True
            return False
        elif type(obstacle) == Rectangle:
            vertex_self = self.get_vertex()
            vertex_their = obstacle.get_vertex()
            return self.check_collision(vertex_their) or obstacle.check_collision(vertex_self)
        else:
            print("Error in check collision")
            return True

    def check_collision_line_segment(self, point1: Point2D, point2: Point2D) -> bool:
        def check_line_segment(point_a1, point_a2, point_b1, point_b2):  # 外積による線分同士の交差判定; Trueなら交差してる
            vec = point_a2 - point_a1
            if vec.cross(point_b1 - point_a1) * vec.cross(point_b2 - point_a1) > 0:
                return False
            vec = point_b2 - point_b1
            if vec.cross(point_a1 - point_b1) * vec.cross(point_a2 - point_b1) > 0:
                return False
            return True

        vertex = self.get_vertex()
        for i in range(len(vertex)):
            if check_line_segment(point1, point2, vertex[i], vertex[(i + 1) % len(vertex)]):
                return True
        return False

    def plot(self, ax, non_fill=False):
        vec = Point2D(-self.w / 2.0, -self.h / 2.0).rotate(self.theta) + self.pos
        c = patches.Rectangle(xy=(vec.x, vec.y), width=self.w, height=self.h, angle=math.degrees(self.theta),
                              fill=(self.fill and (not non_fill)), fc=self.color, ec=self.color)
        ax.add_patch(c)
        ax.set_aspect("equal")

    def calc_min_dist_point(self, point: Point2D) -> float:  # pointとの最短距離を計算する
        vec = (point - self.pos).rotate(-self.theta)  # 回転した座標系でのcenterから見たpointの位置
        if np.abs(vec.x) <= self.w / 2.0:
            return max(abs(vec.y) - self.h / 2.0, 0.0)
        elif np.abs(vec.y) <= self.h / 2.0:
            return max(abs(vec.x) - self.w / 2.0, 0.0)
        else:
            return min(map(lambda vert: (vert - point).len(), self.get_vertex()))

    def change_pos(self, new_pos: Point2D) -> None:  # 中心の位置を変更する
        self.pos = new_pos


class Field:
    def __init__(self, w, h, center=False):
        # center=Trueの時はフィールドの中心を原点にする。Falseの時は左下の角が原点
        self.obstacles = []
        if center:
            self.frame = Rectangle(0.0, 0.0, w, h, 0.0, fill=True, color="black")
        else:
            self.frame = Rectangle(w / 2.0, h / 2.0, w, h, 0.0, fill=True, color="black")

    def check_collision(self, point) -> bool:
        if not self.frame.check_collision(point):
            return True
        for obs in self.obstacles:
            if obs.check_collision(point):
                return True
        return False

    def check_collision_line_segment(self, point1: Point2D, point2: Point2D) -> bool:
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

    def plot_path(self, path: list[Point2D], start_point=None, target_point=None, ax=None, show=True):
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

    def plot_path_control_point(self, path: list[Point2D], ctrl_points=None, ax=None, show=True, lined=True):
        if ax is None:
            ax = self.plot_field()
        if ctrl_points is not None:
            for i in range(len(ctrl_points)):
                if i == len(ctrl_points) - 1:
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
        for i in range(len(path) - 1):
            p1, p2 = path[i].getXY(), path[i + 1].getXY()
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="red")
        ax.plot(path[-1].x, path[-1].y, color="red", marker="x", markersize=8.0)

        ax.set_aspect('equal')
        plt.draw()
        plt.pause(0.001)
        ax.clear()

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)


def GenNHK2022_Field():
    field = Field(12, 12)
    field.add_obstacle(Rectangle(6.0, 6.0, 9.0, 7.0, 0.0, obstacle=False, fill=False))

    field.add_obstacle(Rectangle(6.0, 6.0, 11.9, 11.9, 0.0, obstacle=True, fill=False))
    field.add_obstacle(Rectangle(6.0, 6.0, 11.9, 11.9, 0.0, obstacle=True, fill=False))

    field.add_obstacle(Rectangle(12.0 - 0.15 / 2.0 - 0.05, 6.0, 0.15, 1, 0.0, obstacle=True, fill=True))
    field.add_obstacle(Rectangle(0.15 / 2.0 + 0.05, 6.0, 0.15, 1, 0.0, obstacle=True, fill=True))

    field.add_obstacle(Rectangle(6.0, 6.0, 0.5 / 2.0, 0.5 / 2.0, 0.0, obstacle=True, fill=True))
    field.add_obstacle(Rectangle(6.0, 6.0, 0.5 / 2.0, 0.5 / 2.0, 0.0, obstacle=True, fill=True))

    field.add_obstacle(Rectangle(6.0, 0.05 + 1 / 2, 1, 1, 0.0, obstacle=False, fill=False))
    field.add_obstacle(Rectangle(6.0, 0.05 + 1.45 + 1 / 2, 1, 1, 0.0, obstacle=False, fill=False))
    field.add_obstacle(Rectangle(6.0, 12.0 - 0.05 - 1 / 2, 1, 1, 0.0, obstacle=False, fill=False))
    field.add_obstacle(Rectangle(6.0, 12.0 - 0.05 - 1.45 - 1 / 2, 1, 1, 0.0, obstacle=False, fill=False))

    field.add_obstacle(Rectangle(6.0, 0.05 + 1.45 + 1 / 2 + 1, 2, 1, 0.0, obstacle=False, fill=False))
    field.add_obstacle(Rectangle(6.0, 12.0 - 0.05 - 1.45 - 1 / 2 - 1, 2, 1, 0.0, obstacle=False, fill=False))

    field.add_obstacle(Rectangle(6.0, 12.0 - 2.5 / 2.0, 12.0, 2.5, 0.0, obstacle=False, fill=False, color="red"))
    field.add_obstacle(Rectangle(6.0, 2.5 / 2.0, 12.0, 2.5, 0.0, obstacle=False, fill=False, color="blue"))
    return field


def GenTestField(num: int):
    if num == 0:
        field = Field(12, 12)
        field.add_obstacle(Circle(2.0, 4.0, 1.0, True))
        field.add_obstacle(Circle(6.0, 6.0, 2.0, True))
        field.add_obstacle(Rectangle(5, 2, 1, 3, np.pi / 4.0, True))
        return field
    elif num == 1:
        field = Field(12, 12)
        field.add_obstacle(Circle(5.0, 6.0, 0.2 + 0.15, True))
        field.add_obstacle(Circle(5.3, 6.0, 0.3 + 0.15, True))
        field.add_obstacle(Circle(5.5, 8.0, 0.15 + 0.15, True))
        field.add_obstacle(Circle(6.2, 6.0, 0.25 + 0.15, True))
        return field
    return Field(12, 12)

