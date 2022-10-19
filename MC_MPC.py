import dataclasses

import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from field import Field, Circle, Rectangle, Point2D, Object
from Robot_model import RobotState2, RobotModel_with_Dynamics
from typing import Any, Generic, TypeVar
from A_star import A_star
from RRT import RRT_star
from scipy import interpolate


@dataclasses.dataclass(frozen=True)
class V_Omega:
    v: float  # 符号付きのvelの大きさ
    omega: float  # \dot{theta}

    def __add__(self, other):
        return V_Omega(self.v + other.v, self.omega + other.omega)

    def __mul__(self, other: float):
        return V_Omega(self.v * other, self.omega * other)

    __rmul__ = __mul__

    def __sub__(self, other):
        return V_Omega(self.v - other.v, self.omega - other.omega)

    def __abs__(self):
        return V_Omega(abs(self.v), abs(self.omega))

    # def __truediv__(self, other: float):
    #     return self * (1.0 / other)

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

    def step(self, state: RobotState2[V_Omega], act: V_Omega, dt: float) -> RobotState2[V_Omega]:
        vel = (act + state.vel) * 0.5
        new_pos = state.pos + Point2D(vel.v * dt * math.cos(state.pos.theta),
                                      vel.v * dt * math.sin(state.pos.theta),
                                      vel.omega * dt)
        return RobotState2[V_Omega](new_pos, act)

    def _clip(self, value, min_value, max_value):
        return max(min(max_value - 1e-3, value), min_value + 1e-3)

    def generate_next_act(self, state_pre: RobotState2[V_Omega], act_pre: V_Omega, config: V_Omega_Config) -> V_Omega:
        max_v = min(config.max_v, state_pre.vel.v + config.max_d_v * config.dt)
        min_v = max(-config.max_v, state_pre.vel.v - config.max_d_v * config.dt)

        max_omega = min(config.max_omega, state_pre.vel.omega + config.max_d_omega * config.dt)
        min_omega = max(-config.max_omega, state_pre.vel.omega - config.max_d_omega * config.dt)

        new_v = self._clip(np.random.normal(state_pre.vel.v, config.sigma_v), min_v, max_v)
        new_omega = self._clip(np.random.normal(state_pre.vel.omega, config.sigma_omega), min_omega, max_omega)
        return V_Omega(new_v, new_omega)


class MCMPC_Config:
    def __init__(self):
        self.act_config = V_Omega_Config(max_v=1.0, max_omega=90.0 * math.pi / 180.0,
                                         max_d_v=8.0, max_d_omega=180.0 * math.pi / 180.0,
                                         sigma_v=0.9, sigma_omega=90.0 * math.pi / 180.0 * 0.9, dt=0.1)
        self.vel_weight = [1.0, 0.0]
        self.d_vel_weight = [1.0, 1.0]
        self.predict_step_num = 10  # 1回の予測に用いるステップ数
        self.iteration_num = 40  # 1回の制御周期あたりの予測計算の回数

        self.num_trajectories_for_calc = 5
        self.id_search_num = 10

        self.reach_dist = 0.2


class MCMPC:
    model: RobotModel_with_Dynamics
    config: MCMPC_Config
    field: Field

    def __init__(self, model: RobotModel_with_Dynamics, config: MCMPC_Config, field: Field):
        self.model = model
        self.config = config
        self.field = field

    def _calc_nearest_index(self, point: Point2D, global_path: list[Point2D], start_index: int) -> int:
        min_dist = float("inf")
        min_idx = start_index
        for i in range(start_index, min(len(global_path), start_index + self.config.id_search_num)):
            if min_dist > (global_path[i] - point).len():
                min_dist = (global_path[i] - point).len()
                min_idx = i
        return min_idx

    def _check_reach_point(self, state: RobotState2, global_path: list[Point2D], global_path_id: int) -> bool:
        if (global_path[global_path_id] - state.pos).len() < self.config.reach_dist:
            return True
        else:
            return False

    def _calc_eval_func(self, state: RobotState2, trajectory: list[RobotState2], global_path: list[Point2D],
                        global_path_idx: int) -> float:
        traj = np.insert(np.array(trajectory), 0, state)
        vels = np.insert(np.array([stat.vel for stat in trajectory]), 0, state.vel)
        d_vels = np.abs(vels[1:-1] - vels[0:-2]) * (1.0 / self.config.act_config.dt)

        for stat in traj:
            if self.field.check_collision(stat.pos):
                return -100.0 * len(traj)
        score = 0.0
        ave_d_vel = np.sum(d_vels) * (1.0 / len(d_vels))
        ave_vel = np.sum(vels) * (1.0 / len(vels))
        path_id = global_path_idx

        for i, tmp in enumerate(traj):
            path_id = min(self._calc_nearest_index(tmp.pos, global_path, path_id), len(global_path) - 1)
            score -= (global_path[path_id] - tmp.pos).len() + ave_d_vel.weighted_sum(self.config.d_vel_weight) * 0.005 \
                     - ave_vel.weighted_sum(self.config.vel_weight) * 0.001
            if path_id > 0:
                target_vec = global_path[path_id] - global_path[path_id - 1]
                d_angle = tmp.pos.theta - math.atan2(target_vec.y, target_vec.x)
                while d_angle > math.pi*2.0 or d_angle <= 0.0:
                    if d_angle > math.pi*2.0:
                        d_angle -= math.pi*2.0
                    else:
                        d_angle += math.pi*2.0
                d_angle = min(abs(d_angle), math.pi*2.0 - abs(d_angle))
                score -= d_angle * 0.1
        # print("score {}".format(score))
        return score

    def _predict_trajectory(self, state: RobotState2, act_pre) -> tuple[list[RobotState2], list]:
        state = copy.deepcopy(state)
        act = copy.deepcopy(act_pre)
        trajectory = list[RobotState2]([state])
        acts = list[type(act_pre)]([act])
        for _ in range(self.config.predict_step_num):
            act = self.model.generate_next_act(state, act, self.config.act_config)
            state = self.model.step(state, act, self.config.act_config.dt)

            acts.append(act)
            trajectory.append(state)
        return trajectory[1:-1], acts[1:-1]

    def calc_step(self, state: RobotState2, act_pre, global_path: list[Point2D], global_path_idx: int) \
            -> tuple[Any, list[RobotState2]]:
        trajectories = list[list[RobotState2]]([])
        actions = list[list[type(act_pre)]]([])

        scores = list[tuple]([])
        for i in range(self.config.iteration_num):
            traj, acts = self._predict_trajectory(state, act_pre)
            actions.append(acts)
            trajectories.append(traj)
            scores.append((self._calc_eval_func(state, traj, global_path, global_path_idx), i))
        scores.sort()
        scores.reverse()
        ans_scores = np.array([score for score, _ in scores[0:self.config.num_trajectories_for_calc]])
        ans_trajs_act = np.array([actions[i][0] for _, i in scores[0:self.config.num_trajectories_for_calc]])
        exp_score_sum = np.sum(np.exp(ans_scores))
        ans_act = np.sum([act * np.exp(score) * (1.0 / (exp_score_sum+1e-20)) for act, score in zip(ans_trajs_act, ans_scores)])

        return ans_act, trajectories[scores[0][1]]

    def calc_trajectory(self, initial_state: RobotState2, act_pre, global_path: list[Point2D]) -> list[Point2D]:
        state = copy.deepcopy(initial_state)
        act = copy.deepcopy(act_pre)
        ans = [state.pos]
        for _ in range(1000):
            idx = self._calc_nearest_index(state.pos, global_path, 0)
            act, predictive_path = self.calc_step(state, act, global_path, min(idx + 1, len(global_path) - 1))
            state = self.model.step(state, act, self.config.act_config.dt)
            ans.append(state.pos)
            if self._check_reach_point(state, global_path, len(global_path) - 1):
                break
            self.field.plot_anime(ans, start_point, target_point, global_path, [tmp.pos for tmp in predictive_path])
        return ans


if __name__ == '__main__':
    field = Field(12, 12)
    field.add_obstacle(Circle(5.0, 5.5, 0.2 + 0.15, True))
    field.add_obstacle(Circle(5.5, 6.0, 0.3 + 0.15, True))
    field.add_obstacle(Circle(5.5, 7.0, 0.15 + 0.15, True))
    field.add_obstacle(Circle(6.2, 6.0, 0.25 + 0.15, True))
    #
    # field = GenNHK2022_Field()
    # field.add_obstacle(Rectangle(5, 2, 1, 3, np.pi / 4.0, True))
    # field.plot()

    start_point = Point2D(1.0, 1.0)
    # target_point = Point2D(6.0, 7.0)
    target_point = Point2D(6.0, 6.5)
    dist, path_global_pre = A_star(field, start_point, target_point, 0.2, show=False)

    # rrt = RRT_star(field, 1.0, 0.05, 0.1)
    # dist, path_global2, _ = rrt.planning(start_point, target_point, 600, show=False, star=True)

    # dwa_config = DWA_Config()
    # init_state_dwa = Robot_model.Robot_state(coord=Point2D(1.0, 1.0, theta=math.pi/2.0))
    # r_model_dwa = Robot_model.Robot_model_Circle(initial_state=init_state_dwa, r=0.1)
    #
    # dwa = DWA(r_model_dwa, dwa_config, field)
    # total_path = dwa.calc_path(path_global_pre, init_state_dwa)

    print(dist)
    path_global = path_global_pre[::4]
    path_global.append(path_global_pre[-1])
    field.plot_path(path_global, start_point, target_point, show=True)
    # field.plot_path(path_global2, start_point, target_point, show=True)
    # field.plot_path(total_path, start_point, target_point, show=True)

    mcmpc_config = MCMPC_Config()
    init_state = RobotState2(Point2D(1.0, 1.0, math.pi/2.0), V_Omega(0.0, 0.0))
    r_model = Parallel_TwoWheel_Vehicle_Model([Circle(x=0.0, y=0.0, r=0.1)])

    # スプライン補間
    xs = ([tmp.x for tmp in path_global])
    ys = ([tmp.y for tmp in path_global])
    print([xs, ys])
    # tck, u = interpolate.splprep([xs, ys], k=3, s=0)
    # u = np.linspace(0, 1, num=100, endpoint=True)
    # spline = interpolate.splev(u, tck)
    # path_glob = []
    # for x, y in zip(spline[0], spline[1]):
    #     path_glob.append(Point2D(x, y))
    # スプライン補間ここまで
    # field.plot_path(path_glob, start_point, target_point, show=True)
    field.plot_path(path_global, start_point, target_point, show=True)

    mcmpc = MCMPC(r_model, mcmpc_config, field)
    final_path = mcmpc.calc_trajectory(init_state, V_Omega(0.0, 0.0), path_global)
    # final_path = mcmpc.calc_trajectory(initial_state, path_glob)
    for tmp in final_path:
        print(tmp.x, tmp.y)
    field.plot_path(final_path, start_point, target_point, show=True)
