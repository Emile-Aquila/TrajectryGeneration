import math
import dataclasses
from objects.field import Point2D, Object
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))
from .Robot_model import RobotState, RobotModel_with_Dynamics


@dataclasses.dataclass(frozen=True)
class V_Omega:
    v: float  # 符号付きのvelの大きさ
    omega: float  # \dot{theta}

    def __add__(self, other):
        return V_Omega(self.v + other.v, self.omega + other.omega)

    def __mul__(self, other: float):
        return V_Omega(self.v * other, self.omega * other)

    __rmul__ = __mul__

    def __truediv__(self, other: float):
        return V_Omega(self.v / other, self.omega / other)

    def __sub__(self, other):
        return V_Omega(self.v - other.v, self.omega - other.omega)

    def __abs__(self):
        return V_Omega(abs(self.v), abs(self.omega))

    def weighted_sum(self, weights: list[float]) -> float:
        return self.v * weights[0] + self.omega * weights[1]


@dataclasses.dataclass
class V_Omega_Config:  # absのmax値
    max_v: float  # [m/s]
    max_omega: float  # [rad/s]

    max_d_v: float  # [m/ss]
    max_d_omega: float  # [rad/ss]

    sigma_v: float
    sigma_omega: float

    dt: float

    def __post_init__(self):
        self.max_v = abs(self.max_v)
        self.max_omega = abs(self.max_omega)

        self.max_d_v = abs(self.max_d_v)
        self.max_d_omega = abs(self.max_d_omega)


class Parallel_TwoWheel_Vehicle_Model(RobotModel_with_Dynamics[V_Omega]):
    def __init__(self, objects: list[Object]):
        super(Parallel_TwoWheel_Vehicle_Model, self).__init__(objects)

    def step(self, state: RobotState[V_Omega], act: V_Omega, dt: float) -> RobotState[V_Omega]:
        vel = (act + state.vel) * 0.5
        new_pos = state.pos + Point2D(vel.v * dt * math.cos(state.pos.theta),
                                      vel.v * dt * math.sin(state.pos.theta),
                                      vel.omega * dt)
        return RobotState[V_Omega](new_pos, act)

    def _clip(self, value, min_value, max_value):
        return max(min(max_value - 1e-3, value), min_value + 1e-3)

    def generate_next_act(self, state_pre: RobotState[V_Omega], act_pre: V_Omega, config: V_Omega_Config) -> V_Omega:
        max_v = min(config.max_v, state_pre.vel.v + config.max_d_v * config.dt)
        min_v = max(-config.max_v, state_pre.vel.v - config.max_d_v * config.dt)

        max_omega = min(config.max_omega, state_pre.vel.omega + config.max_d_omega * config.dt)
        min_omega = max(-config.max_omega, state_pre.vel.omega - config.max_d_omega * config.dt)

        new_v = self._clip(np.random.normal(state_pre.vel.v, config.sigma_v), min_v, max_v)
        new_omega = self._clip(np.random.normal(state_pre.vel.omega, config.sigma_omega), min_omega, max_omega)
        return V_Omega(new_v, new_omega)
