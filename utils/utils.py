from objects.Point2D import Point2D
from scipy import interpolate
import numpy as np


def spline(global_path: list[Point2D]) -> list[Point2D]:  # スプライン補間
    xs = ([tmp.x for tmp in global_path])
    ys = ([tmp.y for tmp in global_path])
    print([xs, ys])
    tck, u = interpolate.splprep([xs, ys], k=3, s=0)
    u = np.linspace(0, 1, num=100, endpoint=True)
    spline = interpolate.splev(u, tck)
    ans_path = []
    for x, y in zip(spline[0], spline[1]):
        ans_path.append(Point2D(x, y))
    return ans_path
