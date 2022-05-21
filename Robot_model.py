from field import Field, Circle, Rectangle, Point2D
import numpy as np
from abc import ABC, abstractmethod
import copy
import math


class Robot_state:
    def __init__(self, coord=Point2D(0.0, 0.0), v0=0.0, v0_theta=0.0, omega0=0.0):
        self.coord = coord
        self.v = v0  # vの符号付きの大きさ
        self.v_theta = v0_theta  # vの機体の座標系に対する角度
        self.omega = omega0  # 角速度


class Robot_model_base(ABC):
    def __init__(self, initial_state):
        self._state = initial_state
        self.objects = []
        pass

    @abstractmethod
    def check_collision(self, obj):  # オブジェクトと衝突していないか判定
        pass

    @abstractmethod
    def plot(self, ax):  # 図形をmatplotlibで描画
        pass


class Robot_model_Circle(Robot_model_base):  # 対向二輪
    def __init__(self, initial_state, r):
        super().__init__(initial_state)
        self._state.v_theta = 0.0  # 速度は常に角度0の方向に向く事にする
        self.objects = [
            Circle(initial_state.coord.x, initial_state.coord.y, r, fill=True, color="green")
        ]
        self.history_pos = [self._state.coord]
        self.history_vel = [self._state.v]
        self.history_omega = [self._state.omega]

    def set_state(self, coord, v=0.0, omega=0.0, v_theta=0.0):
        self._state.coord = coord
        self._state.v = v
        self._state.omega = omega
        self._state.v_theta = v_theta

    def get_state(self):
        return copy.deepcopy(self._state)

    def check_collision(self, obj):
        for tmp in self.objects:
            if tmp.check_collision(obj):
                return True
        return False

    def plot(self, ax):
        for tmp in self.objects:
            tmp.plot(ax)
        ax.set_aspect("equal")

    def predict_state(self, v, omega, dt, num_steps):
        predict_cords = []

        for i in range(num_steps):
            tmp_point = Point2D(
                self._state.coord.x + v * math.cos(self._state.coord.theta + omega * i * dt) * dt,
                self._state.coord.y + v * math.sin(self._state.coord.theta + omega * i * dt) * dt,
                self._state.coord.theta + omega * (i+1) * dt
            )
            predict_cords.append(tmp_point)

        return predict_cords

    def move(self, v, omega, dt):
        self._state.coord.x += v * math.cos(self._state.coord.theta) * dt
        self._state.coord.y += v * math.sin(self._state.coord.theta) * dt
        self._state.coord.theta += omega * dt

        self._state.v = v
        self._state.omega = omega

        self.history_pos.append(self.get_state().coord)
        self.history_vel.append(self.get_state().v)
        self.history_omega.append(self.get_state().omega)


class Robot_model_base2(ABC):  # 形状と次の状態への遷移式のみを保持する
    def __init__(self):
        self.objects = []
        pass

    @abstractmethod
    def check_collision(self, obj, state):  # オブジェクトと衝突していないか判定
        pass

    @abstractmethod
    def plot(self, ax, state):  # 図形をmatplotlibで描画
        pass

    @abstractmethod
    def next_state(self, state):  # 状態遷移の計算
        pass
#
#
#
# class Robot_state_swerve():
#     def __init__(self):
#         self.coord = coord
#         self.steer_poses = np.zeros(4, 2)  # [(l, theta)]
#         self.steer_thetas = np.zeros(4)
#         self.steer_vs = np.zeros(4)
#         self.v = 0.0
#         self.omega = 0.0
#         # STEER_LEFT_FRONT,  // 機体の左前のユニット 0
#         # STEER_LEFT_BACK,
#         # STEER_RIGHT_FRONT,
#         # STEER_RIGHT_BACK,
#
#     def set_velocities(self, steer_thetas,
#                        ):


# class Robot_model_Swerve(Robot_model_base2):  # 独立ステアリング
#     def __init__(self, initial_state, r):
#         super().__init__()
#         self.objects = [
#             Circle(0.0, 0.0, r, fill=True, color="green")
#         ]
#
#     def check_collision(self, obj, state):
#         for tmp in self.objects:
#             tmp.change_center(state.coord)
#             if tmp.check_collision(obj):
#                 return True
#         return False
#
#     def plot(self, ax, state):
#         for tmp in self.objects:
#             tmp.plot(ax)
#         ax.set_aspect("equal")
#
#     def predict_state(self, v, omega, dt, num_steps):
#         predict_cords = []
#
#         for i in range(num_steps):
#             tmp_point = Point2D(
#                 self._state.coord.x + v * math.cos(self._state.coord.theta + omega * i * dt) * dt,
#                 self._state.coord.y + v * math.sin(self._state.coord.theta + omega * i * dt) * dt,
#                 self._state.coord.theta + omega * (i+1) * dt
#             )
#             predict_cords.append(tmp_point)
#
#         return predict_cords
#
#     def next_state(self, state):


# def GenNHK2022R1():


