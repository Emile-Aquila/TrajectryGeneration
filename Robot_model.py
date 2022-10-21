import dataclasses

from field import Field, Circle, Rectangle, Point2D, Object
from typing import Any, Generic, TypeVar
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
                self._state.coord.theta + omega * (i + 1) * dt
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


VelType = TypeVar("VelType")


@dataclasses.dataclass(frozen=True)
class RobotState2(Generic[VelType]):
    pos: Point2D
    vel: VelType


class RobotModel2(ABC):
    _objects: list[Object]

    def __init__(self, objects: list[Object]):
        self._objects = objects

    def check_collision(self, state: RobotState2, obstacles: list[Object]) -> bool:
        tmp_objects = self.get_objects(state)
        for obstacle in obstacles:
            for obj in tmp_objects:
                if obj.check_collision(obstacle):
                    return True
        return False

    def get_objects(self, state: RobotState2) -> list[Object]:
        def calc_pos(pos_: Point2D, obj_: Object) -> Point2D:
            new_pos = pos_ + Point2D(obj_.pos.x, obj_.pos.y, 0.0).rotate(pos_.theta)
            new_pos.theta += obj_.pos.theta
            return new_pos

        def generate_new_obj(obj: Object, pos: Point2D) -> Object:
            ans = copy.deepcopy(obj)
            ans.change_pos(pos)
            return ans

        return list(map(lambda x: generate_new_obj(x, calc_pos(state.pos, x)), self._objects))

    def plot(self, ax):
        for tmp in self._objects:
            tmp.plot(ax)
        ax.set_aspect("equal")


ActType = TypeVar("ActType")


class RobotModel_with_Dynamics(Generic[ActType], RobotModel2):
    def __init__(self, objects: list[Object]):
        super(RobotModel_with_Dynamics, self).__init__(objects)

    @abstractmethod
    def step(self, state: RobotState2, act: ActType, dt: float) -> RobotState2:
        raise NotImplementedError()

    @abstractmethod
    def generate_next_act(self, state: RobotState2, act_pre: ActType, config: Any) -> ActType:
        pass
