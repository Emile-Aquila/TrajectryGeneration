import numpy as np
import time
import matplotlib.pyplot as plt
import heapq
import sys

sys.path.append('./')
from objects.field import Field, Circle, Point2D, GenTestField
from models.Robot_model import RobotModel, RobotState


def gen_motion_model(unit_dist: float) -> list[Point2D]:
    return [
        Point2D(unit_dist, 0.0),
        Point2D(-unit_dist, 0.0),
        Point2D(0.0, unit_dist),
        Point2D(0.0, -unit_dist),
        Point2D(unit_dist, unit_dist),
        Point2D(unit_dist, -unit_dist),
        Point2D(-unit_dist, unit_dist),
        Point2D(-unit_dist, -unit_dist),
    ]


def A_star(field: Field, start_point: Point2D, target_point: Point2D,
           robot_model: RobotModel | None, check_length=0.01,
           unit_dist=0.05, show=True) -> (float, list[Point2D]):
    motion_model = gen_motion_model(unit_dist)
    queue = [((target_point - start_point).len(), 0.0, start_point, None)]  # score, dist, point, point_prev
    checked_node = dict()  # key: point,  value: (dist, point_prev)

    def A_check_collision(node_pre: Point2D, new_node: Point2D) -> bool:
        if field.check_collision(new_node):
            return True
        elif field.check_collision_line_segment(node_pre, new_node):
            return True
        else:
            if robot_model is None:
                return False
            if robot_model.check_collision(RobotState[None](node_pre, None), field.obstacles) \
                    or robot_model.check_collision(RobotState[None](new_node, None), field.obstacles):
                return True
            unit_vec = (new_node - node_pre) * (1.0 / (new_node - node_pre).len())
            for i in range(int((node_pre - new_node).len() // check_length)):
                pos = node_pre + unit_vec * (i + 1) * check_length
                if robot_model.check_collision(RobotState[None](pos, None), field.obstacles):
                    return True
            return False

    def finish_terminate(point_now: Point2D, max_dist: float) -> bool:
        if (point_now - target_point).len() < max_dist * np.sqrt(2) / 2.0:
            return True
        else:
            return False

    def eval_func(point_now: Point2D, point_prev: Point2D) -> float:
        return checked_node[point_prev.getXY()][0] + (point_now - point_prev).len() + (target_point - point_now).len()

    terminate_point = None
    while len(queue) != 0:
        score, dist, point, point_prev = heapq.heappop(queue)  # score, dist, point, point_prev
        if (point.getXY() in checked_node) and (dist >= checked_node[point.getXY()][0]):
            continue
        checked_node[point.getXY()] = (dist, point_prev)
        if finish_terminate(point, unit_dist):
            terminate_point = point
            break
        for vec in motion_model:
            if A_check_collision(point, point + vec):
                continue
            new_node = (eval_func(point + vec, point), dist + vec.len(), point + vec, point)
            if (point + vec).getXY() not in checked_node:
                heapq.heappush(queue, new_node)

    # ここまで解の探索
    if len(queue) == 0:
        print("Can't archive goal")
        return []
    ans_path = [terminate_point]
    while ans_path[-1] != start_point:
        ans_path.append(checked_node[ans_path[-1].getXY()][1])
    ans_path.reverse()
    if show:
        ax = field.plot_field()
        for point in checked_node.keys():
            ax.plot(point[0], point[1], color="red", marker='.', markersize=1.0)
        plt.show()
    return checked_node[terminate_point.getXY()][0], ans_path


if __name__ == '__main__':
    field = GenTestField(0)
    field.plot()

    start_point = Point2D(0.1, 0.1)
    target_point = Point2D(8.0, 8.0)
    r_model = RobotModel([Circle(x=0.0, y=0.0, r=0.2, fill=True)])

    start_time = time.process_time()
    dist, path = A_star(field, start_point, target_point, r_model, check_length=0.1, unit_dist=0.1, show=True)  # A*
    print(time.process_time() - start_time)
    print(dist)
    field.plot_path(path, start_point, target_point, show=True)
