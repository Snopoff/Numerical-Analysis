# Задача 11; Вариант 3
# %%
"""
Решение задачи Коши с заданной точностью
с автоматическим выбором максимальной длины шага

y' = f(x,y) : x ∈ [a,b]
y(c) = y_c : c ∈ {a,b}

"""

from typing import Callable
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

filename = 'data.txt'


def get_data(name: str) -> list:
    """
    Get data method
    Returns list of parameters
    """

    with open(name) as f:
        lines = f.readlines()[2:]
    first, second = lines
    parameters = [None]*2
    parameters[0] = list(map(lambda x: float(x), first.split(" ")))
    parameters[1] = list(map(lambda x: float(x), second.split(" ")))
    return parameters


def give_data(X: list, Y: list, count_eps: int, count_min: int, count_max: int):
    """
    Write output in the file
    """
    with open('output.txt', "w+") as f:
        for i in range(len(X)):
            f.write('f({:.8f}) = {:.8f}\n'.format(X[i], Y[i]))
        f.write('{}, {}, {}, {}'.format(
            len(X), count_eps, count_min, count_max))


def estimate(f: Callable, h: float, x: float, y=0) -> float:
    """
    Runge-Kutta method for solving ODE

        Parameters:
        ----------
        f : RHS of given ODE
        h : step
        x : float number x: y: x -> y(x)
        y : desirable function
    """

    k1 = h * f(x, y)
    k2 = h * f(x + h/2, y + k1/2)
    return y + k2


def optimize(f: Callable, h: float, x: float, y=0) -> float:
    """
    Runge-Kutta method for optimizing solution

        Parameters:
        ----------
        f : RHS of given ODE
        h : step
        x : float number x: y: x -> y(x)
        y : desirable function
    """

    k1 = h * f(x, y)
    k2 = h * f(x + h/2, y + k1/2)
    k3 = h * f(x + h/2, k2/2)
    k4 = h * f(x + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6


def solve(f: Callable) -> list:
    """
    ODE Solver using Runge-Kutta method of 2nd order

        Parameters:
        ----------
        f : RHS of given ODE
    """
    count_min = 0
    count_max = 0
    data = get_data(filename)
    a, b, c, y_c = data[0]
    h_min, h_max, eps = data[1]
    h = (b - a) / 10
    steer = 1
    if c == b:
        steer = -1
    X = []
    Y = [y_c]
    Eps = []
    x = c
    while a <= x and x <= b:
        X.append(x)
        y = Y[-1]
        Y.append(estimate(f, h, x, y))
        y_hat = optimize(f, h, x, y)
        x = x + steer * h
        eps_n = abs(Y[-1] - y_hat)
        Eps.append(eps_n)
        h_e = np.power(eps_n/eps, 0.25) * h
        if h_e < h_min:
            h = h_min
            count_min += 1
        elif h_e > h_max:
            h = h_max
            count_max += 1
        else:
            h = h_e
    count_eps = sum(e < eps for e in Eps)
    return X, Y[:-1], count_eps, count_min, count_max


def f(x: float, y=0) -> float:
    """
    RHS of given ODE
    """

    return 12*x**2


def F(x: float, y=0) -> float:
    """
    Such function that dF = fdt
    """
    return 4*x**3


def main():
    X, Y, count_eps, count_min, count_max = solve(f)
    for i in range(len(X)):
        print('f({:.8f}) = {:.8f}'.format(X[i], Y[i]))

    give_data(X, Y, count_eps, count_min, count_max)
    ''' --plot is probably needed here-- '''

    data = get_data(filename)
    a, b, c, y_c = data[0]
    x = np.linspace(a, b, len(X))
    y = F(x)
    plt.plot(x, y, "k-", label='Точное решение')
    plt.plot(X, Y, "r--", label='Численное решение')
    plt.legend(loc='best')
    plt.title('График заданной функции')
    plt.savefig('graphs.png')
    plt.show()


if __name__ == "__main__":
    main()


# %%
