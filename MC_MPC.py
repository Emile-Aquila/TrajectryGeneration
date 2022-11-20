import numpy as np
import math
import copy
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import time
from operator import itemgetter
import sys
import os

sys.path.append(os.path.dirname(__file__))
from objects.field import Field, Circle, Point2D, GenTestField, Rectangle
from models.Robot_model import RobotState, RobotModel_with_Dynamics
from models.DynamicsModel import V_Omega, V_Omega_Config, Parallel_TwoWheel_Vehicle_Model
from A_star import A_star


class MCMPC_Config:
    def __init__(self):
        self.act_config = V_Omega_Config(max_v=1.0, max_omega=90.0 * math.pi / 180.0,
                                         max_d_v=8.0, max_d_omega=180.0 * math.pi / 180.0,
                                         sigma_v=0.7, sigma_omega=90.0 * math.pi / 180.0 * 0.7, dt=0.12)
        self.vel_weight = [1.0, -0.05]
        self.d_vel_weight = [1.0, 1.0]
        self.predict_step_num = 8  # 1回の予測に用いるステップ数
        self.iteration_num = 40  # 1回の制御周期あたりの予測計算の回数

        self.num_trajectories_for_calc = 3
        self.id_search_num = 7

        self.reach_dist = 0.3


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
        if self._check_reach_point(point, global_path, start_index):
            start_index = min(start_index + 2, len(global_path) - 1)
        min_idx = start_index
        for i in range(start_index, min(len(global_path), start_index + self.config.id_search_num)):
            if min_dist > (global_path[i] - point).len():
                min_dist = (global_path[i] - point).len()
                min_idx = i
        return min_idx

    def _check_reach_point(self, pos: Point2D, global_path: list[Point2D], global_path_id: int) -> bool:
        if (global_path[global_path_id] - pos).len() < self.config.reach_dist:
            return True
        else:
            return False

    def _eval_trajectory(self, state: RobotState, trajectory: list[RobotState], global_path: list[Point2D],
                         global_path_idx: int) -> float:
        traj = np.insert(np.array(trajectory), 0, state)
        vels = np.insert(np.array([stat.vel for stat in trajectory]), 0, state.vel)
        d_vels = np.abs(vels[1:-1] - vels[0:-2]) * (1.0 / self.config.act_config.dt)

        score = 0.0
        for stat in traj:
            if self.field.check_collision(stat.pos):
                return -100.0 * len(traj)
        for stat in traj:
            if self.model.check_collision(stat, field.obstacles):
                score -= 100.0
        ave_d_vel = np.sum(d_vels) * (1.0 / len(d_vels))
        ave_vel = np.sum(vels) * (1.0 / len(vels))

        score += ave_vel.weighted_sum(self.config.vel_weight) * 0.01 * len(traj)
        score -= ave_d_vel.weighted_sum(self.config.d_vel_weight) * 0.003 * len(traj)
        for i, tmp in enumerate(traj):
            path_id = min(self._calc_nearest_index(tmp.pos, global_path, global_path_idx), len(global_path) - 1)
            score -= (global_path[path_id] - tmp.pos).len() * 0.5 / max(1, path_id - global_path_idx)
            if path_id > 0:
                target_vec = global_path[path_id] - global_path[path_id - 1]
                d_angle = abs(tmp.pos.theta - math.atan2(target_vec.y, target_vec.x)) % (math.pi * 2.0)
                d_angle = min(abs(d_angle), math.pi * 2.0 - abs(d_angle))
                score -= d_angle * 0.1
        return score

    def _predict_trajectory(self, state: RobotState, act_pre) -> tuple[list[RobotState], list]:
        state = copy.deepcopy(state)
        act = copy.deepcopy(act_pre)
        trajectory = list[RobotState]([state])
        acts = list[type(act_pre)]([act])
        for _ in range(self.config.predict_step_num):
            act = self.model.generate_next_act(state, act, self.config.act_config)
            state = self.model.step(state, act, self.config.act_config.dt)

            acts.append(act)
            trajectory.append(state)
        return trajectory[1:-1], acts[1:-1]

    def calc_step(self, state: RobotState, act_pre, global_path: list[Point2D], global_path_idx: int) \
            -> tuple[Any, list[RobotState]]:
        def gen_traj(index):
            traj, acts = self._predict_trajectory(state, act_pre)
            score = self._eval_trajectory(state, traj, global_path, global_path_idx)
            return [score, acts[0], traj]

        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="thread") as executor:
            results = executor.map(gen_traj, range(self.config.iteration_num))
        results = list(results)

        results.sort(key=itemgetter(0))
        results.reverse()
        results = np.array(results[:self.config.num_trajectories_for_calc])

        exp_scores = np.exp(results[:, 0].astype("float32"))
        ans_act = np.average(results[:, 1], weights=exp_scores)
        return ans_act, results[0][2]

    def calc_trajectory(self, initial_state: RobotState, act_pre, global_path: list[Point2D], show: bool) \
            -> list[Point2D]:
        state = copy.deepcopy(initial_state)
        act = copy.deepcopy(act_pre)
        ans = [state.pos]
        global_path_idx = 0
        for i in range(1000):
            global_path_idx = self._calc_nearest_index(state.pos, global_path, global_path_idx)
            act, predictive_path = self.calc_step(state, act, global_path, min(global_path_idx, len(global_path) - 1))
            state = self.model.step(state, act, self.config.act_config.dt)
            ans.append(state.pos)
            if (state.pos - global_path[-1]).len() < 0.15:
                break
            if show and (i % 2 == 0):
                # self.field.plot_anime(ans[-min(20, len(ans)):-1], start_point, target_point, global_path,
                #                       [tmp.pos for tmp in predictive_path], model=self.model.get_objects(state))
                self.field.plot_anime(ans, start_point, target_point, global_path,
                                      [tmp.pos for tmp in predictive_path], model=self.model.get_objects(state))
        return ans


if __name__ == '__main__':
    field = GenTestField(0)
    field.plot_field()

    start_point, target_point = Point2D(0.5, 0.5), Point2D(8.0, 8.0)
    r_model = Parallel_TwoWheel_Vehicle_Model(
        [Rectangle(x=0.0, y=0.0, w=0.3, h=0.6, theta=math.pi/2.0)]
    )
    dist, path_global = A_star(field, start_point, target_point, r_model, check_length=0.1, unit_dist=0.2,show=True)
    field.plot_path(path_global, start_point, target_point, show=True)

    mcmpc_config = MCMPC_Config()
    init_state = RobotState(Point2D(start_point.x, start_point.y, math.pi / 2.0), V_Omega(0.0, 0.0))
    mcmpc = MCMPC(r_model, mcmpc_config, field)

    start_time = time.process_time()
    final_path = mcmpc.calc_trajectory(init_state, V_Omega(0.0, 0.0), path_global, False)  # MCMPC
    print(time.process_time() - start_time)

    field.plot_path(final_path, start_point, target_point, show=True)
