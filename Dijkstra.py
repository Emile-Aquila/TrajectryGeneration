import numpy as np
from objects.field import Field, Circle, Rectangle, Point2D
import heapq


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


def dijkstra(field: Field, start_point: Point2D, target_point: Point2D, unit_dist=0.05) -> (float, list[Point2D]):
    motion_model = gen_motion_model(unit_dist)
    queue = [(0.0, start_point, None)]  # dist, point, point_prev
    checked_node = dict()  # key: point,  value: (dist, point_prev)

    def finish_terminate(point_now, max_dist):
        if (point_now - target_point).len() < max_dist * np.sqrt(2) / 2.0:
            return True
        else:
            return False

    terminate_node = None
    while True:
        present_node = heapq.heappop(queue)
        if present_node[1].getXY() in checked_node:
            continue
        checked_node[present_node[1].getXY()] = (present_node[0], present_node[2])
        if finish_terminate(present_node[1], unit_dist):
            terminate_node = present_node
            break
        for vec in motion_model:
            new_node = (present_node[0] + vec.len(), present_node[1] + vec, present_node[1])
            if field.check_collision(new_node[1]) or field.check_collision_line_segment(new_node[1], new_node[2]):
                continue
            if new_node[1].getXY() not in checked_node:
                heapq.heappush(queue, new_node)
    # ここまで解の探索
    ans_path = [terminate_node[1]]
    while ans_path[-1] != start_point:
        ans_path.append(checked_node[ans_path[-1].getXY()][1])
    ans_path.reverse()
    return terminate_node[0], ans_path


if __name__ == '__main__':
    field = Field(12, 12)
    field.add_obstacle(Circle(2.0, 4.0, 1.0, True))
    field.add_obstacle(Circle(6.0, 6.0, 2.0, True))
    field.add_obstacle(Rectangle(5, 2, 1, 3, np.pi / 4.0, True))
    field.plot()

    start_point = Point2D(0.1, 0.1)
    target_point = Point2D(8.0, 8.0)
    dist, path = dijkstra(field, start_point, target_point, 0.1)
    print(dist)
    field.plot_path(path, start_point, target_point)
