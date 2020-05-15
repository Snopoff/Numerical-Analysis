# Вариант 9

"""
Метод универсальной дифференциальной прогонки для линейных уравнений второго порядка

    y''(x) + p(x)y'(x) = q(x)y(x) + f(x)
    a0y'(a) + b0y(a) = A  : a0^2 + b0^2 > 0
    a1y'(b) + b1y(b) = B  : a1^2 + b1^2 > 0
"""


import numpy as np
from typing import Callable
import itertools
from collections import OrderedDict as od


filename = 'data.txt'


def get_data():
    """
    Get data method
    Returns list of parameters
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    conds = list(map(lambda x: float(x), lines[0].split(" ")))
    b_cond1, b_cond2 = conds[:3], conds[3:]
    interval = list(map(lambda x: float(x), lines[1].split(" ")))
    flat_X = list(map(lambda x: int(
        x) if x != '0.0' and float(x).is_integer() else float(x), [i for item in lines[2:] for i in item.rstrip().split(" ")]))
    X = od(dict(itertools.zip_longest(*[iter(flat_X)] * 2, fillvalue="")))
    return b_cond1, b_cond2, interval, X


def write_data(X, Y, Yprime):
    with open('rez.txt', 'w') as f:
        for key, value in X.items():
            f.write(" ".join([str(key), str(value),
                              str(Y[value]), str(Yprime[value]), "\n"]))


def RungeKutta(f: Callable, h: np.double, x: np.double, y=0) -> np.double:
    """
    Runge-Kutta method of fourth order, also named as Kutta-Merson method
        Parameters:
        ----------
        f : RHS of given ODE
        h : step
        x : float number x: y: x -> y(x)
        y : desirable function
    """
    k1 = h * f(x, y)
    k2 = h * f(x + h/3, y + k1/3)
    k3 = h * f(x + h/3, y + k1/6 + k2/6)
    k4 = h * f(x + h/2, y + k1/8 - 3*k3/8)
    k5 = h * f(x + h, y + k1/2 - 3*k3/2 + 2*k4)
    return y + (k1 + 3*k3 + 4*k4 + 2*k5) / 10


def p(x):
    return x+1


def q(x):
    return 2


def f(x):
    return 2


def forward(X: dict, b_cond1: list, interval: list):
    """
    Прямая прогонка: нужно решить следующие 2 системы:
    -------------
    z1' = -z1^2 - p(x)z1 + q(x)
    z1(a) = - a0 / b0
    -------------
    z2' = -z2 * [z1 + p(x)] + f(x)
    z2(a) = A / b0
    """

    a0, b0, A = b_cond1
    a, b, n = interval
    h = (b - a) / n

    Z1 = od({a: -a0 / b0})
    Z2 = od({a: A / b0})

    # Solve first given differential equation
    def func(x, z): return - z**2 - p(x)*z + q(x)

    for i in range(0, len(X)-1):
        # x = next(reversed(Z1))
        x = X[i]
        z = Z1[x]
        Z1[X[i+1]] = RungeKutta(func, h, x, z)

    # solve second given differential equation
    def func(z1): return lambda x, z: - z * z1 + p(x) + f(x)

    for i in range(0, len(X)-1):
        # x = next(reversed(Z2))
        x = X[i]
        z = Z2[x]
        z1 = Z1[x]
        Z2[X[i+1]] = RungeKutta(func(z1), h, x, z)
    return Z1, Z2


def backward(X: dict, b_cond2: list, interval: list, Z1: dict, Z2: dict):
    """
    Обратная прогонка: нужно решить следующую задачу Коши:
        y' = z1 * y + z2
        y(b) = (B - b1*z2(b)) / (a1 + b1*z1(b))
    """

    a1, b1, B = b_cond2
    a, b, n = interval
    h = (b - a) / n

    Y = od({b: (B - b1*Z2[b]) / (a1 + b1*Z1[b])})

    def func(z1, z2):
        return lambda x, y: z1*y + z2

    for i in range(len(X)-1, 0, -1):
        # x = next(reversed(Z2))
        x = X[i]
        y = Y[x]
        z1 = Z1[x]
        z2 = Z2[x]
        Y[X[i-1]] = RungeKutta(func(z1, z2), h, x, y)
    return od(reversed(Y.items()))


def solve():
    b_cond1, b_cond2, interval, X = get_data()
    Z1, Z2 = forward(X, b_cond1, interval)
    Y = backward(X, b_cond2, interval, Z1, Z2)
    Yprime = {}
    for x in X.values():
        Yprime[x] = Z1[x] * Y[x] + Z2[x]

    write_data(X, Y, Yprime)


def main():
    solve()


if __name__ == '__main__':
    main()
