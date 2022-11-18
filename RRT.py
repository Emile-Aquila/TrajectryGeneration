import operator

import numpy as np
import random
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.append(os.path.dirname(__file__))
try:
    from .objects.field import Field, Point2D, Circle, GenTestField
    from .models.Robot_model import RobotModel, RobotState
    from .utils.utils import print_process_time
except:
    from objects.field import Field, Point2D, Circle, GenTestField
    from models.Robot_model import RobotModel, RobotState
    from utils.utils import print_process_time


class Tree:
    nodes: list[Point2D | None]
    edges: list[(int, float)]  # 親の頂点, 辺の長さ
    size: int

    def __init__(self, NodeSize: int):
        self.nodes = [None] * (NodeSize + 10)
        self.edges = [None] * (NodeSize + 10)
        self.size = 0

    def append(self, point: Point2D, parent_id: int):
        self.nodes[self.size] = point
        if 0 <= parent_id < self.size:
            self.edges[self.size] = (parent_id, (self.nodes[parent_id] - point).len())
            self.size += 1
        elif parent_id == -1:
            self.edges[self.size] = (parent_id, 0.0)
            self.size += 1
        else:
            print("Tree ERROR")

    def dist(self, node_id: int) -> float:
        tmp_id, ans = node_id, 0.0
        while tmp_id != -1:
            tmp_id, edge_len = self.edges[tmp_id]
            ans += edge_len
        return ans


class RRT:
    eps: float
    check_length: float  # modelの衝突判定時に直線区間の何メートル区切りで衝突判定を行うか
    goal_sample_rate: float
    robot_model: RobotModel | None

    def __init__(self, field: Field, robot_model: RobotModel | None, eps=1.0, goal_sample_rate=0.05, check_length=0.1):
        self.field = field
        self.eps = eps
        self.check_length = check_length
        self.goal_sample_rate = goal_sample_rate
        self.robot_model = robot_model
        vertex = field.frame.get_vertex()
        self.max_xy = (max(vertex, key=lambda v: v.x).x, max(vertex, key=lambda v: v.y).y)
        self.min_xy = (min(vertex, key=lambda v: v.x).x, min(vertex, key=lambda v: v.y).y)

    def _calc_nearest_neighbor(self, tree: Tree, point: Point2D) -> (int, Point2D):
        min_id = np.fromiter(map(lambda idx: (point - tree.nodes[idx]).len(), range(tree.size)), dtype=float).argmin()
        return min_id, tree.nodes[min_id]

    def _reconnect_nodes(self, tree: Tree, new_node_id: int, non_collision_ids: list[int], start_point: Point2D):
        # nodeの再接続.
        pass

    def _find_neighborhood(self, tree: Tree, new_node: Point2D) -> list[int]:
        # new_node付近の点の集合を求める
        pass

    def _check_collision(self, node_pre: Point2D, new_node: Point2D, check_length: float) -> bool:
        if self.field.check_collision_line_segment(node_pre, new_node):
            return True
        else:
            if self.robot_model is None:
                return False
            if self.robot_model.check_collision(RobotState[None](node_pre, None), self.field.obstacles) \
                    or self.robot_model.check_collision(RobotState[None](new_node, None), self.field.obstacles):
                return True
            unit_vec = (new_node - node_pre) * (1.0 / (new_node - node_pre).len())
            for i in range(int((node_pre - new_node).len() // check_length)):
                pos = node_pre + unit_vec * (i + 1) * check_length
                if self.robot_model.check_collision(RobotState[None](pos, None), self.field.obstacles):
                    return True
            return False

    def planning(self, start_point: Point2D, target_point: Point2D, try_num=100, star=True, show=True) \
            -> (float | None, list[Point2D] | None, list[Point2D] | None):
        # star = Trueの時はRRT*
        tree = Tree(try_num + 2)
        tree.append(start_point, -1)

        while tree.size < try_num:
            if random.uniform(0.0, 1.0) < self.goal_sample_rate:
                tmp_point = target_point
            else:
                tmp_point = Point2D(random.uniform(self.min_xy[0], self.max_xy[0]),
                                    random.uniform(self.min_xy[1], self.max_xy[1]))

            nearest_id, nearest_vert = self._calc_nearest_neighbor(tree, tmp_point)
            new_node = ((tmp_point - nearest_vert) * self.eps) + nearest_vert
            if not self._check_collision(nearest_vert, new_node, self.check_length):
                if star:  # RRT* (reconnect)
                    neighborhood_ids = self._find_neighborhood(tree, new_node)  # 衝突なしの近傍
                    if len(neighborhood_ids) == 0:
                        neighborhood_ids = [nearest_id]

                    pre_node_id = min(neighborhood_ids,
                                      key=lambda idx: tree.dist(idx) + (tree.nodes[idx] - new_node).len()
                                      )

                    tree.append(new_node, pre_node_id)
                    self._reconnect_nodes(tree, tree.size - 1, neighborhood_ids, start_point)
                else:
                    tree.append(new_node, nearest_id)

        nearest_node, nearest_id = None, 0
        dist_ids = sorted([((target_point - tree.nodes[i]).len() + tree.dist(i), i) for i in range(tree.size)],
                          key=operator.itemgetter(0))
        ans_dist = None
        for tmp_dist, node_id in dist_ids:
            if not self._check_collision(tree.nodes[node_id], target_point, self.check_length):
                nearest_id = node_id
                ans_dist = tmp_dist
                break
        if ans_dist is None:
            return None, None, None
        ans_path = [target_point]
        tmp_id = nearest_id
        while tmp_id != -1:
            ans_path.append(tree.nodes[tmp_id])
            tmp_id = tree.edges[tmp_id][0]
        ans_path.reverse()
        if show:
            ax = self.field.plot_field()
            for node in tree.nodes[0:tree.size]:
                ax.plot(node.x, node.y, color="red", marker='.', markersize=1.0)
            plt.show()
        return ans_dist, ans_path, tree.nodes[0:tree.size]


class RRT_star(RRT):
    R: float

    def __init__(self, field: Field, robot_model: RobotModel | None, R: float, eps=1.0, goal_sample_rate=0.05,
                 check_length=0.1):
        super().__init__(field, robot_model, eps, goal_sample_rate, check_length)
        self.R = R

    def _find_neighborhood(self, tree: Tree, new_node: Point2D) -> list[int]:
        # r近傍にある衝突なしの点を計算する
        r = self.R * np.power(np.log(tree.size) / tree.size, 0.5)  # r = R{{log(N)/N}^(1/d))
        dists = np.fromiter(map(lambda idx: (new_node - tree.nodes[idx]).len(), range(tree.size)), dtype=float)
        r_neighbor = np.array(range(tree.size))[dists < r]

        ans = list(
            filter(lambda idx: not self._check_collision(tree.nodes[idx], new_node, self.check_length), r_neighbor))
        return ans

    def _reconnect_nodes(self, tree: Tree, new_node_id: int, non_collision_ids: list[int], start_point: Point2D):
        new_node = tree.nodes[new_node_id]
        for node_id in non_collision_ids:
            node = tree.nodes[node_id]
            tmp_dist = tree.dist(new_node_id) + (node - new_node).len()
            if tree.dist(node_id) > tmp_dist:
                tree.edges[node_id] = (new_node_id, (node - new_node).len())


if __name__ == '__main__':
    field = GenTestField(0)
    field.plot()

    r_model = RobotModel([Circle(x=0.0, y=0.0, r=0.1, fill=True)])
    start_pt = Point2D(0.1, 0.1)
    target_pt = Point2D(8.0, 8.0)

    start_time = time.process_time()
    # rrt = RRT(field, None, eps=0.15, goal_sample_rate=0.05, check_length=0.1)  # RRT*
    rrt = RRT_star(field, None, R=15.0, eps=0.2, goal_sample_rate=0.05)  # RRT*
    dist, path, _ = rrt.planning(start_pt, target_pt, 200, show=False)
    time_all = time.process_time() - start_time
    print(time_all)
    field.plot_path(path, start_pt, target_pt, show=True)
