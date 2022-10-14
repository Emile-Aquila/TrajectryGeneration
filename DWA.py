import numpy as np
import math
import copy
import time
import matplotlib.pyplot as plt
from field import Field, Circle, Rectangle, Point2D
from RRT import RRT_star
from Robot_model import Robot_model_Circle, Robot_state
import sys
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
    def __init__(self, robot_model, dwa_config, field):
        self.robot_model = robot_model
        self.config = dwa_config
        self.field = field

    def _calc_vel_range(self, state, config):
        range_omega = config.sampling_time * config.max_d_omega
        max_omega = min(state.omega + range_omega, config.max_omega)
        min_omega = max(state.omega - range_omega, config.min_omega)

        range_v = config.sampling_time * config.max_d_v
        max_v = min(state.v + range_v, config.max_v)
        min_v = max(state.v - range_v, config.min_v)
        # print(state.v, range_v)
        # print("min, max: {}, {}, {}, {}".format(max_omega, min_omega, max_v, min_v))
        return max_omega, min_omega, max_v, min_v

    def _predict_path(self, state, config):
        max_omega, min_omega, max_v, min_v = self._calc_vel_range(state, config)  # 速度と角速度の範囲を算出
        paths = []  # 予測される経路の集合
        # print("min, max: {}, {}, {}, {}".format(max_omega, min_omega, max_v, min_v))
        for omega in np.arange(min_omega, max_omega, config.omega_resolution):
            for v in np.arange(min_v, max_v, config.v_resolution):
                new_path = Path(self.robot_model.predict_state(v, omega, config.sampling_time, config.predict_step_num),
                                v, omega)
                paths.append(new_path)
        return paths

    def _calc_score_angle(self, path, target_pos, state):  # 180 - (最終的なロボットの方位とゴールの方向の差)
        def angle_range_corrector(angle):  # 角度補正
            if angle > math.pi:
                while angle > math.pi:
                    angle -= 2 * math.pi
            elif angle < -math.pi:
                while angle < -math.pi:
                    angle += 2 * math.pi
            return angle

        angle_to_goal = math.atan2(target_pos.y - path.traj[-1].y, target_pos.x - path.traj[-1].x)
        score = angle_to_goal - path.traj[-1].theta  # score計算
        # score = angle_to_goal - state.coord.theta  # score計算
        score = abs(angle_range_corrector(score))  # 角度の補正(ぐるぐるの防止)
        return math.pi - score

    def _calc_score_vel(self, path):
        return path.v

    def _get_near_obs(self, obstacles, point, max_dist):  # 距離の近い(max_dist以下の)障害物を求める
        ans_obs = []
        ans_dists = []
        for obs in obstacles:
            tp = obs.calc_min_dist_point(point)
            if max_dist >= tp:
                ans_obs.append(obs)
                ans_dists.append(tp)
        return ans_obs, ans_dists

    def _calc_score_obstacle(self, path, obstacles, max_dist):  # 障害物のスコアを求める。 返り値は score, boolean(衝突があるか)
        near_obs = self._get_near_obs(obstacles, self.robot_model.get_state().coord, max_dist)
        total_min_dist = max_dist
        for point in path.traj:
            obses = [obs.calc_min_dist_point(point) for obs in near_obs[0]]
            # obses = near_obs[0]
            if len(obses) == 0:
                continue
            for obs in near_obs[0]:
                if obs.check_collision(point):
                    return 0.0, True
            min_dist = min(obses)
            if min_dist == 0.0:
                return 0.0, True
            if total_min_dist > min_dist:
                total_min_dist = min_dist
        return total_min_dist, False

    def _eval_path(self, paths, obstacles, target_pos, obs_max_dist=1.0):
        scores_angle, scores_vel, scores_obs, paths_available = [], [], [], []
        for path in paths:  # pathを評価
            obs_score, conflict = self._calc_score_obstacle(path, obstacles, max_dist=obs_max_dist)
            if conflict:  # 衝突が存在する
                continue
            scores_obs.append(obs_score)
            scores_angle.append(self._calc_score_angle(path, target_pos, self.robot_model.get_state()))
            scores_vel.append(self._calc_score_vel(path))
            paths_available.append(path)

        # for scores in [scores_angle, scores_vel, scores_obs]:
        #     scores = min_max_normalize(scores)
        max_score = sys.float_info.min
        ans_path = None
        for score_angle, score_vel, score_obs, path in zip(scores_angle, scores_vel, scores_obs, paths_available):
            tmp_score = self.config.score_gain_angle * score_angle + self.config.score_gain_vel * score_vel \
                        + self.config.score_gain_obs * score_obs \
                        - (path.traj[-1] - target_pos).len() * self.config.score_gain_dist
            if tmp_score > max_score:
                max_score = tmp_score
                ans_path = path
        return ans_path

    def calc_path(self, global_path, initial_state, show=True):
        def terminate_condition(target_pos, robot_model):
            if (target_pos - robot_model.get_state().coord).len() < 0.5:
                return True
            else:
                return False

        local_path = []
        self.robot_model.set_state(coord=initial_state.coord, v=initial_state.v, omega=initial_state.omega)
        for i in range(1, len(global_path)):
            while not terminate_condition(global_path[i], self.robot_model):
                paths = self._predict_path(self.robot_model.get_state(), self.config)
                optimal_path = self._eval_path(paths, self.field.obstacles, global_path[i], obs_max_dist=0.3)
                if optimal_path is None:
                    return local_path
                self.robot_model.move(optimal_path.v, optimal_path.omega, self.config.sampling_time)
                local_path.append(self.robot_model.get_state().coord)
                if len(local_path) >= 500:
                    return local_path
        if show:
            ax = self.field.plot_field()
            for node in local_path:
                ax.plot(node.x, node.y, color="red", marker='.', markersize=1.0)
            plt.show()
        return local_path


if __name__ == '__main__':
    field = Field(12, 12)
    field.add_obstacle(Circle(2.0, 4.0, 1.0, True))
    field.add_obstacle(Circle(6.0, 6.0, 2.0, True))
    field.add_obstacle(Rectangle(5, 2, 1, 3, np.pi / 4.0, True))
    field.plot()

    start_point = Point2D(0.1, 0.1)
    target_point = Point2D(8.0, 8.0)
    # rrt = RRT_star(field, 1.0, 0.05, 0.1)
    dist, path_global_pre = A_star(field, start_point, target_point, 0.3, show=False)
    print(dist)
    # field.plot_path(path, start_point, target_point)
    path_global = path_global_pre[::4]
    path_global.append(path_global_pre[-1])
    field.plot_path(path_global, start_point, target_point, show=True)

    # ここからDWA
    dwa_config = DWA_Config()
    initial_state = Robot_state(coord=Point2D(0.1, 0.1, math.pi/2.0))
    robot_model = Robot_model_Circle(initial_state=initial_state, r=0.1)

    dwa = DWA(robot_model, dwa_config, field)
    total_path = dwa.calc_path(path_global, initial_state)
    print(total_path)
    field.plot_path(total_path, start_point, target_point, show=True)
    field.plot_path(path_global, start_point, target_point, show=True)





