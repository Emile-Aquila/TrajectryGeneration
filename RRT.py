import numpy as np
import random
from field import Field, Point2D, Rectangle, Circle
import matplotlib.pyplot as plt


class RRT:
    def __init__(self, field, eps=1.0, goal_sample_rate=0.05):
        # self.kdtree = ss.cKDTree
        self.field = field
        self.eps = eps
        self.goal_sample_rate = goal_sample_rate
        self.max_xy = (
            self.field.frame.center.x + self.field.frame.w / 2.0, self.field.frame.center.y + self.field.frame.h / 2.0)
        self.min_xy = (
            self.field.frame.center.x - self.field.frame.w / 2.0, self.field.frame.center.y - self.field.frame.h / 2.0)

    def _calc_nearest_neighbor(self, nodes, point):
        ans, min_dist = None, None
        for node in nodes:
            if ans is None or min_dist > (node - point).len():
                ans = node
                min_dist = (node - point).len()
        return ans, min_dist

    def _get_dist(self, tree, node, start_point):  # TODO : バグありそう
        if node is start_point:
            return 0.0
        if tree[node.getXY()][0] is None:
            return None
        dist = 0.0
        while tree[node.getXY()][0] is not None:
            dist += tree[node.getXY()][1]
            node = tree[node.getXY()][0]
            # print(node.getXY())
        return dist

    def _reconnect_nodes(self, nodes, tree, new_node, dist_new_node, start_point):
        # nodeの再接続. ただしnew_nodeは頂点集合及びグラフに入れていない状態で使う事
        pass

    def planning(self, start_point, target_point, try_num=100, star=True, show=True):  # star = Trueの時はRRT*
        tree = dict()
        tree[start_point.getXY()] = (None, 0.0)  # 接続元の頂点, 辺の長さ
        nodes = [start_point]  # 頂点集合

        while len(nodes) < try_num:
            if random.uniform(0.0, 1.0) < self.goal_sample_rate:
                tmp_point = target_point
            else:
                tmp_point = Point2D(random.uniform(self.min_xy[0], self.max_xy[0]),
                                    random.uniform(self.min_xy[1], self.max_xy[1]))
            nn_node, _ = self._calc_nearest_neighbor(nodes, tmp_point)
            new_point = (tmp_point - nn_node) * self.eps + nn_node
            if not (self.field.check_collision(new_point) or self.field.check_collision_line_segment(nn_node, new_point)):
                if star:  # RRT*
                    tmp_array = []
                    for node in nodes:
                        if self.field.check_collision_line_segment(node, new_point):
                            continue
                        tmp_array.append((self._get_dist(tree, node, start_point)+(node-new_point).len(), node))
                    tmp = min(tmp_array)
                    self._reconnect_nodes(nodes, tree, new_point, tmp[0], start_point)
                    # new_pointをnodes, treeに入れる前に使う事.
                    nodes.append(new_point)
                    tree[new_point.getXY()] = (tmp[1], (tmp[1] - new_point).len())
                else:
                    nodes.append(new_point)
                    tree[new_point.getXY()] = (nn_node, (nn_node - new_point).len())
        nn_node, dist = self._calc_nearest_neighbor(nodes, target_point)
        sorted_node = sorted(nodes, key=lambda x: (x-target_point).len())
        for node in sorted_node:
            if not self.field.check_collision_line_segment(node, target_point):
                nn_node = node
                break
        ans_path = [target_point, nn_node]
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
    def __init__(self, field, R, eps=1.0, goal_sample_rate=0.05):
        super().__init__(field, eps, goal_sample_rate)
        self.R = R

    def _reconnect_nodes(self, nodes, tree, new_node, dist_new_node, start_point):
        r = self.R * np.power(np.log(len(nodes)) / len(nodes), 0.5)  # r = R{{log(N)/N}^(1/d))
        for node in nodes:
            if (node - new_node).len() >= r or self.field.check_collision_line_segment(node, new_node):
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

    start_point = Point2D(0.1, 0.1)
    target_point = Point2D(8.0, 8.0)
    rrt = RRT_star(field, 1.0, 0.05, 0.1)
    dist, path, _ = rrt.planning(start_point, target_point, 200)
    print(dist)
    # field.plot_path(path, start_point, target_point)
    field.plot_path(path, start_point, target_point, show=True)
