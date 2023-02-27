import numpy as np
import math
import scipy.integrate
from objects.field import Point2D, GenNHK2022_Field
from utils.utils import spline


def psi(phi_v, phi_u, s):
    return phi_v * s + phi_u * (s ** 2.0)


def f(phi_v, phi_u, a=0.0, b=1.0, n=1000, re_im=False):
    # int_a^b exp(i * (phi_v * s + phi_u * s^2)) ds
    s = np.linspace(a, b, n)
    phis_cos = np.cos(psi(phi_v, phi_u, s))
    phis_sin = np.sin(psi(phi_v, phi_u, s))
    re = scipy.integrate.simps(y=phis_cos, x=s)
    im = scipy.integrate.simps(y=phis_sin, x=s)
    if not re_im:
        angle = np.arctan2(im, re)
        absolute = np.sqrt((re ** 2.0) + (im ** 2.0))
        return absolute, angle
    else:
        return re, im


def change_angle(angle):
    while np.abs(angle) > math.pi * 2.0:
        if angle > 0.0:
            angle -= math.pi * 2.0
        else:
            angle += math.pi * 2.0
    return min(np.abs(angle), math.pi * 2.0 - np.abs(angle))


def section_clothoid(p0, p1, phi0, phi1, dl=0.1, d_phi=0.1):
    sum_phi = phi1 - phi0
    psi_ = np.arctan2((p1 - p0).y, (p1 - p0).x) - phi0
    if psi_ != 0.0:
        phis = np.arange(-10.0 * np.abs(psi_), 10.0 * np.abs(psi_), d_phi)
    else:
        phis = np.arange(-10.0 * np.abs(math.pi / 5.0), 10.0 * np.abs(math.pi / 5.0), d_phi)
    phi_v = phis[np.argmin([change_angle(psi_ - f(phi, sum_phi - phi)[1]) for phi in phis])]
    phi_u = sum_phi - phi_v
    lambd, psi2 = f(phi_v, phi_u)
    print(psi_ - psi2)
    l, h = (p1 - p0).len(), (p1 - p0).len() / lambd
    if h < dl:
        point_num = 10
    else:
        point_num = math.floor(h / dl)
    # p1 - p0 = he^{i phi_0} \int_0^1 exp(i (phi_v s + phi_u s^2)) ds
    ans = [p0]
    for i in range(point_num):
        a, b = float(i / point_num), float((i + 1) / point_num)
        if i == (point_num - 1):
            b = 1.0
        tmp_lambda, tmp_psi = f(phi_v, phi_u, a, b, n=100, re_im=False)
        delta_vec = Point2D((tmp_lambda * h) * np.cos(tmp_psi + phi0), (tmp_lambda * h) * np.sin(tmp_psi + phi0), ) + \
                    ans[-1]
        ans.append(delta_vec)
    return ans


def clothoid(points, start_angle, target_angle):
    angles = [start_angle]
    for i in range(1, len(points) - 1):
        delta_vec = (points[i + 1] - points[i - 1])
        angle = np.arctan2(delta_vec.y, delta_vec.x)
        angles.append(angle)
    angles.append(target_angle)
    ans = [points[0]]
    for i in range(len(points) - 1):
        tmp_path = section_clothoid(ans[-1], points[i + 1], angles[i], angles[i + 1], dl=0.1, d_phi=0.005)
        ans.extend(tmp_path)
    return ans


def set_angle(points, first_angle):
    points[0].theta = first_angle
    for id in range(1, len(points)):
        points[id].theta = math.atan2(points[id].y - points[id - 1].y, points[id].x - points[id - 1].x)
    return points


if __name__ == '__main__':
    field = GenNHK2022_Field()
    field.plot()

    start_point = Point2D(6.0, 2.0, 0.0)
    target_point = Point2D(11.5, 6.0, 0.0)

    pointss = [
        start_point,
        Point2D(10.0, 2.0),
        Point2D(11.0, 3.0),
        target_point
    ]
    path = clothoid(pointss, 0.0, math.pi / 2.0)
    field.plot_path_control_point(path, pointss, show=True)
    path = set_angle(path[0::2], 0.0)
    for point in path:
        print("{{{}, {}, {}}},".format(point.x, point.y, point.theta))
    print(len(path))

    # path_glob = spline(pointss)
    # field.plot_path_control_point(path_glob, pointss, show=True)
