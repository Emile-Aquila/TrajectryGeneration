from scipy import interpolate
import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from objects.Point2D import Point2D


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


def print_process_time(f):  # 計測デコレータ
    def tmp(*args, **kwargs):
        start_time = time.process_time()
        return_val = f(*args, **kwargs)  # 処理実行
        end_time = time.process_time()
        elapsed_time = end_time - start_time
        print(f.__name__, elapsed_time)
        return return_val
    return tmp
