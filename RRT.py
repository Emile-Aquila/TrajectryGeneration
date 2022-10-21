import numpy as np
import random
from field import Field, Point2D, Rectangle, Circle
from Robot_model import RobotModel2, RobotState2
import matplotlib.pyplot as plt


class RRT:
    eps: float
    check_length: float  # modelの衝突判定時に直線区間の何メートル区切りで衝突判定を行うか
    goal_sample_rate: float
    robot_model: RobotModel2 | None

    def __init__(self, field: Field, robot_model: RobotModel2 | None, eps=1.0, goal_sample_rate=0.05, check_length=0.1):
        # self.kdtree = ss.cKDTree
        self.field = field
        self.eps = eps
        self.check_length = check_length
        self.goal_sample_rate = goal_sample_rate
        self.robot_model = robot_model
        vertex = field.frame.get_vertex()
        self.max_xy = (max(vertex, key=lambda v: v.x).x, max(vertex, key=lambda v: v.y).y)
        self.min_xy = (min(vertex, key=lambda v: v.x).x, min(vertex, key=lambda v: v.y).y)

    def _calc_nearest_neighbor(self, nodes: list[Point2D], point: Point2D) -> (Point2D, float):
        ans, min_dist = None, None
        for node in nodes:
            if ans is None or min_dist > (node - point).len():
                ans = node
                min_dist = (node - point).len()
        return ans, min_dist

    def _get_dist(self, tree: dict, node: Point2D, start_point: Point2D) -> float:  # TODO : バグありそう
        if node is start_point:
            return 0.0
        if tree[node.getXY()][0] is None:
            return 0.0
        dist = 0.0
        while tree[node.getXY()][0] is not None:
            dist += tree[node.getXY()][1]
            node = tree[node.getXY()][0]
            # print(node.getXY())
        return dist

    def _reconnect_nodes(self, nodes, tree, new_node, dist_new_node, start_point):
        # nodeの再接続. ただしnew_nodeは頂点集合及びグラフに入れていない状態で使う事
        pass

    def _check_collision(self, node_pre: Point2D, new_node: Point2D, check_length: float) -> bool:
        if self.field.check_collision(new_node):
            return True
        elif self.field.check_collision_line_segment(node_pre, new_node):
            return True
        else:
            if self.robot_model is None:
                return False
            if self.robot_model.check_collision(RobotState2[None](node_pre, None), self.field.obstacles) \
                    or self.robot_model.check_collision(RobotState2[None](new_node, None), self.field.obstacles):
                return True
            unit_vec = (new_node - node_pre) * (1.0 / (new_node - node_pre).len())
            for i in range(int((node_pre-new_node).len() // check_length)):
                pos = node_pre+unit_vec*(i+1)*check_length
                if self.robot_model.check_collision(RobotState2[None](pos, None), self.field.obstacles):
                    return True
            return False

    def planning(self, start_point: Point2D, target_point: Point2D, try_num=100, star=True, show=True):
        # star = Trueの時はRRT*
        tree = dict()
        tree[start_point.getXY()] = (None, 0.0)  # 接続元の頂点, 辺の長さ
        nodes = [start_point]  # 頂点集合

        while len(nodes) < try_num:
            if random.uniform(0.0, 1.0) < self.goal_sample_rate:
                tmp_point = target_point
            else:
                tmp_point = Point2D(random.uniform(self.min_xy[0], self.max_xy[0]),
                                    random.uniform(self.min_xy[1], self.max_xy[1]))
            nearest_node, _ = self._calc_nearest_neighbor(nodes, tmp_point)
            new_node = ((tmp_point - nearest_node) * self.eps) + nearest_node
            if not self._check_collision(nearest_node, new_node, self.check_length):
                if star:  # RRT* (reconnect)
                    def f_total_dist(nod):  # 根からの距離
                        return self._get_dist(tree, nod, start_point) + (nod - new_node).len()

                    pre_node = min(filter(lambda nod: not self._check_collision(nod, new_node, self.check_length), nodes)
                                   , key=f_total_dist)
                    self._reconnect_nodes(nodes, tree, new_node, f_total_dist(pre_node), start_point)
                    # ^ new_pointをnodes, treeに入れる前に使う事.
                    nodes.append(new_node)
                    tree[new_node.getXY()] = (pre_node, (pre_node - new_node).len())
                else:
                    nodes.append(new_node)
                    tree[new_node.getXY()] = (nearest_node, (nearest_node - new_node).len())
        nearest_node, dist = self._calc_nearest_neighbor(nodes, target_point)
        sorted_node = sorted(nodes, key=lambda x: (x - target_point).len())
        for node in sorted_node:
            if not self._check_collision(node, target_point, self.check_length):
                nearest_node = node
                break

        ans_path = [target_point, nearest_node]
        while ans_path[-1] is not start_point:
            ans_path.append(tree[ans_path[-1].getXY()][0])
            dist += tree[ans_path[-1].getXY()][1]
        ans_path.reverse()
        if show:
            ax = self.field.plot_field()
            for node in nodes:
                ax.plot(node.x, node.y, color="red", marker='.', markersize=1.0)
            plt.show()
        return dist, ans_path, nodes


class RRT_star(RRT):
    R: float

    def __init__(self, field: Field, robot_model: RobotModel2 | None, R: float, eps=1.0, goal_sample_rate=0.05, check_length=0.1):
        super().__init__(field, robot_model, eps, goal_sample_rate, check_length)
        self.R = R

    def _reconnect_nodes(self, nodes: list[Point2D], tree: dict, new_node: Point2D, dist_new_node: float,
                         start_point: Point2D):
        r = self.R * np.power(np.log(len(nodes)) / len(nodes), 0.5)  # r = R{{log(N)/N}^(1/d))
        for node in nodes:
            if (node - new_node).len() >= r or self._check_collision(node, new_node, self.check_length):
                continue
            tmp_dist = dist_new_node + (node - new_node).len()
            if self._get_dist(tree, node, start_point) > tmp_dist:
                tree[node.getXY()] = (new_node, tmp_dist)


if __name__ == '__main__':
    field = Field(12, 12)
    field.add_obstacle(Circle(2.0, 4.0, 1.0, True))
    field.add_obstacle(Circle(6.0, 6.0, 2.0, True))
    field.add_obstacle(Rectangle(5, 2, 1, 3, np.pi / 4.0, True))
    field.plot()

    r_model = RobotModel2([Circle(x=0.0, y=0.0, r=0.1, fill=True)])
    start_pt = Point2D(0.1, 0.1)
    target_pt = Point2D(8.0, 8.0)
    rrt = RRT_star(field, r_model, 1.0, 0.15, 0.05, 0.1)
    dist, path, _ = rrt.planning(start_pt, target_pt, 200)
    print(dist)
    # field.plot_path(path, start_point, target_point)
    field.plot_path(path, start_pt, target_pt, show=True)
