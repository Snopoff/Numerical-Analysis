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
    flat_X = [i for item in lines[2:] for i in item.rstrip().split(" ")]
    X = od(dict(itertools.zip_longest(*[iter(flat_X)] * 2, fillvalue="")))
    return b_cond1, b_cond2, interval, X


def write_data(X):
    with open('rez.txt', 'w') as f:
        f.write(str(X))


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

    def func(x, z): return - z**2 - p(x)*z + q(x)

    for i in range(1, int(n)):
        x = next(reversed(Z1))
        z = Z1[x]
        Z1[x + h] = RungeKutta(func, h, x, z)

    def help_func(x): return Z1[x] + p(x)
    def func(x, z): return - z * help_func(x) + f(x)

    for i in range(1, int(n)):
        x = next(reversed(Z2))
        z = Z2[x]
        print(z)
        Z2[x + h] = RungeKutta(func, h, x, z)
    return Z2


def solve():
    b_cond1, b_cond2, interval, X = get_data()
    print(forward(X, b_cond1, interval))


def main():
    solve()


if __name__ == '__main__':
    main()
