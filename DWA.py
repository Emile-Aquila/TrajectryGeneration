import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from objects.field import Field, Circle, Point2D, Object, GenTestField
from models.Robot_model import RobotState
from models.DynamicsModel import V_Omega, Parallel_TwoWheel_Vehicle_Model
from A_star import A_star


class DWA_Config:
    def __init__(self):
        self.max_v = 1.0  # [m/s]
        self.min_v = -1.0  # [m/s]
        self.max_omega = 360.0 * math.pi / 180.0  # [rad/s]
        self.min_omega = -360.0 * math.pi / 180.0  # [rad/s]

        self.max_d_v = 8.0  # [m/ss]
        self.max_d_omega = 180.0 * math.pi / 180.0  # [rad/ss]

        self.v_resolution = 0.1  # [m/s]
        self.omega_resolution = 1.0 * math.pi / 180.0  # [rad/s]

        self.sampling_time = 0.1  # [s] Time tick for motion prediction
        self.predict_step_num = 20  # [無次元] 予測するstep数

        self.score_gain_obs = 0.05
        self.score_gain_vel = 1.0
        self.score_gain_angle = 1.0 * 0.1
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked

        self.score_gain_dist = 0.1


class Path:
    def __init__(self, trajectory, v, omega):
        self.traj = trajectory
        self.v = v
        self.omega = omega


class DWA:
    robot_model: Parallel_TwoWheel_Vehicle_Model
    config: DWA_Config
    field: Field

    def __init__(self, robot_model: Parallel_TwoWheel_Vehicle_Model, dwa_config: DWA_Config, field: Field):
        self.robot_model = robot_model
        self.config = dwa_config
        self.field = field

    def _calc_vel_range(self, state: RobotState[V_Omega], config: DWA_Config) -> (float, float, float, float):
        range_omega = config.sampling_time * config.max_d_omega
        max_omega = min(state.vel.omega + range_omega, config.max_omega)
        min_omega = max(state.vel.omega - range_omega, config.min_omega)

        range_v = config.sampling_time * config.max_d_v
        max_v = min(state.vel.v + range_v, config.max_v)
        min_v = max(state.vel.v - range_v, config.min_v)
        return max_omega, min_omega, max_v, min_v

    def _predict_path(self, state: RobotState[V_Omega], config: DWA_Config) \
            -> list[(V_Omega, list[RobotState[V_Omega]])]:
        max_omega, min_omega, max_v, min_v = self._calc_vel_range(state, config)  # 速度と角速度の範囲を算出
        paths = list[(V_Omega, list[RobotState[V_Omega]])]([])  # 予測される経路の集合
        for omega in np.arange(min_omega, max_omega, config.omega_resolution):
            for v in np.arange(min_v, max_v, config.v_resolution):
                new_state = copy.deepcopy(state)
                path = list[RobotState[V_Omega]]([])
                for _ in range(config.predict_step_num):
                    new_state = self.robot_model.step(new_state, V_Omega(v=v, omega=omega), config.sampling_time)
                    path.append(new_state)
                paths.append((V_Omega(v, omega), path))
        return paths

    def _calc_score_angle(self, path: list[RobotState[V_Omega]], target_pos: Point2D) -> float:
        # 180 - (最終的なロボットの方位とゴールの方向の差)
        def angle_range_corrector(angle: float) -> float:  # 角度補正
            if angle > math.pi:
                while angle > math.pi:
                    angle -= 2 * math.pi
            elif angle < -math.pi:
                while angle < -math.pi:
                    angle += 2 * math.pi
            return angle

        angle_to_goal = math.atan2((target_pos - path[-1].pos).y, (target_pos - path[-1].pos).x)
        score = abs(angle_range_corrector(angle_to_goal - path[-1].pos.theta))  # score計算
        return math.pi - score

    def _get_near_obs(self, obstacles: list[Object], point: Point2D, max_dist: float) -> (list[Object], list[float]):
        # 距離の近い(max_dist以下の)障害物を求める
        ans_obs, ans_dists = [], []
        for obs in obstacles:
            tp = obs.calc_min_dist_point(point)
            if max_dist >= tp:
                ans_obs.append(obs)
                ans_dists.append(tp)
        return ans_obs, ans_dists

    def _calc_score_obstacle(self, state: RobotState[V_Omega], path: list[RobotState[V_Omega]],
                             obstacles: list[Object], max_dist: float):  # 障害物のスコアを求める。 返り値は score, boolean(衝突があるか)
        near_obs = self._get_near_obs(obstacles, state.pos, max_dist)
        total_min_dist = max_dist
        for stat in path:
            dist_obs = [obs.calc_min_dist_point(stat.pos) for obs in near_obs[0]]
            if len(dist_obs) == 0:
                continue
            for obs in near_obs[0]:
                if obs.check_collision(stat.pos):
                    return 0.0, True
            min_dist = min(dist_obs)
            if min_dist == 0.0:
                return 0.0, True
            if total_min_dist > min_dist:
                total_min_dist = min_dist
        return total_min_dist, False

    def _eval_path(self, state: RobotState[V_Omega], act: V_Omega, path: list[RobotState[V_Omega]],
                   obstacles: list[Object], target_pos: Point2D, obs_max_dist=1.0) -> float:
        obs_score, conflict = self._calc_score_obstacle(state, path, obstacles, max_dist=obs_max_dist)
        if conflict:
            return -float("inf")
        score_angle = self._calc_score_angle(path, target_pos)
        score = self.config.score_gain_angle * score_angle + self.config.score_gain_vel * act.v \
                + self.config.score_gain_obs * obs_score \
                - (path[-1].pos - target_pos).len() * self.config.score_gain_dist
        return score

    def calc_path(self, global_path, initial_state: RobotState[V_Omega], show=True):
        def terminate_condition(target_pos: Point2D, stat: RobotState[V_Omega]):
            if (target_pos - stat.pos).len() < 0.5:
                return True
            else:
                return False

        local_path = list[Point2D]([])
        state = copy.deepcopy(initial_state)
        for i in range(1, len(global_path)):
            while not terminate_condition(global_path[i], state):
                paths: list[(V_Omega, list[RobotState[V_Omega]])] = self._predict_path(state, self.config)  # pathを生成
                list_score_ids = list[(float, int)]([])
                for index, path in enumerate(paths):
                    action, traj = path
                    score = self._eval_path(state, action, traj, self.field.obstacles, global_path[i], obs_max_dist=0.3)
                    list_score_ids.append((score, index))
                max_score, optimal_id = max(list_score_ids)  # 最も評価値が高いpathを選択

                if paths[optimal_id] is None:
                    return local_path
                state = self.robot_model.step(state, paths[optimal_id][0], self.config.sampling_time)
                local_path.append(state.pos)
                if len(local_path) >= 500:
                    return local_path
        if show:
            ax = self.field.plot_field()
            for node in local_path:
                ax.plot(node.x, node.y, color="red", marker='.', markersize=1.0)
            plt.show()
        return local_path


if __name__ == '__main__':
    field = GenTestField(0)
    field.plot()

    start_point = Point2D(0.1, 0.1)
    target_point = Point2D(8.0, 8.0)
    dist, path_global_pre = A_star(field, start_point, target_point, None, check_length=0.1, unit_dist=0.3, show=False)

    path_global = path_global_pre[::4]
    path_global.append(path_global_pre[-1])
    field.plot_path(path_global, start_point, target_point, show=True)

    # ここからDWA
    dwa_config = DWA_Config()
    init_state = RobotState[V_Omega](Point2D(0.1, 0.1, math.pi / 2.0), V_Omega(0.0, 0.0))
    r_model = Parallel_TwoWheel_Vehicle_Model([Circle(0.0, 0.0, r=0.1, fill=True)])

    dwa = DWA(r_model, dwa_config, field)
    total_path = dwa.calc_path(path_global, init_state)
    print(total_path)
    field.plot_path(total_path, start_point, target_point, show=True)