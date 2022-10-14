import numpy as np
import math
import copy
import time
import matplotlib.pyplot as plt
from field import Field, Circle, Rectangle, Point2D, GenNHK2022_Field
# import Robot_model
from abc import ABC, abstractmethod
from A_star import A_star
from RRT import RRT_star
from scipy import interpolate
from DWA import DWA_Config, DWA


class Robot_state:
    def __init__(self, coord=Point2D(0.0, 0.0), v0=0.0, omega0=0.0):
        self.coord = coord
        self.v = v0  # vの符号付きの大きさ
        self.omega = omega0  # 角速度


class Robot_model_base(ABC):
    def __init__(self):
        self.objects = []

    @abstractmethod
    def check_collision(self, state, obj):  # オブジェクトと衝突していないか判定
        pass

    @abstractmethod
    def plot(self, ax, state):  # 図形をmatplotlibで描画
        pass

    @abstractmethod
    def move(self, state, v, omega, dt):  # ロボットを動かす
        pass


class Robot_model_Circle(Robot_model_base):  # 対向二輪
    def __init__(self, r):
        super().__init__()
        self.objects = [
            Circle(0.0, 0.0, r, fill=True, color="green")
        ]

    def check_collision(self, state, obj):
        for tmp in self.objects:
            tmp.change_center(state.coord)  # ロボットのオブジェクトの座標合わせ
            if tmp.check_collision(obj):
                return True
        return False

    def plot(self, ax, state):
        for tmp in self.objects:
            tmp.change_center(state.coord)
            tmp.plot(ax)
        ax.set_aspect("equal")

    def move(self, state, v, omega, dt):
        ans_state = copy.deepcopy(state)
        ans_state.coord.x += (v+state.v)/2.0 * math.cos(state.coord.theta) * dt
        ans_state.coord.y += (v+state.v)/2.0 * math.sin(state.coord.theta) * dt
        ans_state.coord.theta += omega * dt
        while ans_state.coord.theta > math.pi*2.0:
            ans_state.coord.theta -= math.pi*2.0
        while ans_state.coord.theta < -math.pi*2.0:
            ans_state.coord.theta += math.pi*2.0
        ans_state.v = v
        ans_state.omega = omega
        return ans_state


class MCMPC_Config:
    def __init__(self):
        self.max_v = 1.0  # [m/s]
        self.min_v = -1.0  # [m/s]
        self.max_omega = 90.0 * math.pi / 180.0  # [rad/s]
        self.min_omega = -90.0 * math.pi / 180.0  # [rad/s]

        self.max_d_v = 8.0  # [m/ss]
        self.max_d_omega = 180.0 * math.pi / 180.0  # [rad/ss]

        self.dt = 0.1  # [s]
        self.predict_step_num = 10  # 1回の予測に用いるステップ数
        self.iteration_num = 40  # 1回の制御周期あたりの予測計算の回数

        self.sigma_v = self.max_v*0.9
        self.sigma_omega = self.max_omega*0.9

        self.num_trajectories_for_calc = 5
        self.id_search_num = 10

        self.reach_dist = 0.2


class MCMPC:
    def __init__(self, model, config, field):
        self.model = model
        self.config = config
        self.field = field

    def _calc_nearest_index(self, point, global_path, start_index):
        min_dist = float("inf")
        min_idx = start_index
        for i in range(start_index, min(len(global_path), start_index+self.config.id_search_num)):
            if min_dist > (global_path[i] - point).len():
                min_dist = (global_path[i] - point).len()
                min_idx = i
        return min_idx

    def _check_reach_point(self, state, global_path, global_path_id):
        if (global_path[global_path_id] - state.coord).len() < self.config.reach_dist:
            return True
        else:
            return False

    def _calc_eval_func(self, state, trajectory, global_path, global_path_idx):
        traj = np.insert(np.array(trajectory), 0, state)
        vs = np.insert(np.array([tmp.v for tmp in trajectory]), 0, state.v)
        omegas = np.insert(np.array([tmp.omega for tmp in trajectory]), 0, state.omega)
        d_v = np.abs(vs[1:-1] - vs[0:-2])/self.config.dt
        d_omega = np.abs(omegas[1:-1] - omegas[0:-2])/self.config.dt
        if np.max(vs) > self.config.max_v or np.min(vs) < self.config.min_v \
                or np.max(omegas) > self.config.max_omega or np.min(omegas) < self.config.min_omega:
            print("a", vs, omegas)
            return 100.0 * len(traj)
        if np.max(d_v) > self.config.max_d_v \
                or np.max(d_omega) > self.config.max_d_omega:
            print("b")
            return 100.0 * len(traj)
        for tmp in traj:
            if self.field.check_collision(tmp.coord):
                return 100.0 * len(traj)
        score = 0.0
        sum_d_v = np.average(d_v)
        sum_d_omega = np.average(d_omega)
        path_id = global_path_idx

        sum_v = np.average(vs)

        for i, tmp in enumerate(traj):
            # id_tmp = self._calc_nearest_index(tmp.coord, global_path, path_id)
            # score += (global_path[id_tmp] - tmp.coord).len() + sum_d_v * 0.005 + sum_d_omega * 0.005
            # if path_id > 0:
                # target_vec = global_path[path_id] - global_path[path_id-1]
                # score += abs(tmp.coord.theta - math.atan2(target_vec.y, target_vec.x)) * 0.1

            path_id = min(self._calc_nearest_index(tmp.coord, global_path, path_id), len(global_path)-1)
            score += (global_path[path_id] - tmp.coord).len() + sum_d_v * 0.005 + sum_d_omega * 0.005 - sum_v * 0.001
            if path_id > 0:
                target_vec = global_path[path_id] - global_path[path_id-1]
                score += abs(tmp.coord.theta - math.atan2(target_vec.y, target_vec.x)) * 0.1
        return score

    def _clip(self, value, min_value, max_value):
        return max(min(max_value-1e-3, value), min_value+1e-3)

    def _predict_trajectory(self, state):
        state = copy.deepcopy(state)
        trajectory = [state]
        dt = self.config.dt
        for _ in range(self.config.predict_step_num):
            v = self._clip(np.random.normal(trajectory[-1].v, self.config.sigma_v), self.config.min_v, self.config.max_v)
            omega = self._clip(np.random.normal(trajectory[-1].omega, self.config.sigma_omega), self.config.min_omega, self.config.max_omega)
            v = self._clip(v, trajectory[-1].v-self.config.max_d_v*dt, trajectory[-1].v+self.config.max_d_v*dt)
            omega = self._clip(omega, trajectory[-1].omega-self.config.max_d_omega*dt, trajectory[-1].omega+self.config.max_d_omega*dt)

            state = self.model.move(state, v, omega, self.config.dt)

            trajectory.append(state)
        return trajectory[1:-1]

    def calc_step(self, state, global_path, global_path_idx):
        trajectories = []
        scores = []
        for i in range(self.config.iteration_num):
            traj = self._predict_trajectory(state)
            trajectories.append(traj)
            scores.append((self._calc_eval_func(state, traj, global_path, global_path_idx), i))
        scores.sort()
        ans_scores = np.array([score for score, _ in scores[0:self.config.num_trajectories_for_calc]])
        ans_trajs_v = np.array([trajectories[i][0].v for _, i in scores[0:self.config.num_trajectories_for_calc]])
        ans_trajs_omega = np.array(
            [trajectories[i][0].omega for _, i in scores[0:self.config.num_trajectories_for_calc]])
        # print(ans_trajs_v, ans_trajs_omega)
        ans_v = np.average(ans_trajs_v, axis=0, weights=np.exp(-ans_scores))
        ans_omega = np.average(ans_trajs_omega, axis=0, weights=np.exp(-ans_scores))
        # print(ans_scores, ans_v, ans_omega)

        return ans_v, ans_omega, trajectories[scores[0][1]]

    def calc_trajectory(self, initial_state, global_path):
        state = copy.deepcopy(initial_state)
        ans = [state.coord]
        for _ in range(1000):
            idx = self._calc_nearest_index(state.coord, global_path, 0)
            v, omega, predictive_path = self.calc_step(state, global_path, min(idx+1, len(global_path)-1))
            state = self.model.move(state, v, omega, self.config.dt)
            ans.append(state.coord)
            if self._check_reach_point(state, global_path, len(global_path)-1):
                break
            self.field.plot_anime(ans, start_point, target_point, global_path, [tmp.coord for tmp in predictive_path])
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
    initial_state = Robot_state(coord=Point2D(1.0, 1.0, math.pi/2.0))
    robot_model = Robot_model_Circle(r=0.1)

    # スプライン補間
    xs = ([tmp.x for tmp in path_global])
    ys = ([tmp.y for tmp in path_global])
    print([xs, ys])
    tck, u = interpolate.splprep([xs, ys], k=3, s=0)
    u = np.linspace(0, 1, num=100, endpoint=True)
    spline = interpolate.splev(u, tck)
    path_glob = []
    for x, y in zip(spline[0], spline[1]):
        path_glob.append(Point2D(x, y))
    # スプライン補間ここまで
    field.plot_path(path_glob, start_point, target_point, show=True)

    mcmpc = MCMPC(robot_model, mcmpc_config, field)
    final_path = mcmpc.calc_trajectory(initial_state, path_global)
    # final_path = mcmpc.calc_trajectory(initial_state, path_glob)
    for tmp in final_path:
        print(tmp.x, tmp.y)
    field.plot_path(final_path, start_point, target_point, show=True)
