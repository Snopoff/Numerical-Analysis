# Задача 11; Вариант 3

"""
Решение задачи Коши с заданной точностью 
с автоматическим выбором максимальной длины шага

y' = f(x,y) : x ∈ [a,b]
y(c) = y_c : c ∈ {a,b}

"""

import numpy as np
from typing import Callable

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


def estimate(f: Callable, h: float, x: float, y=0):
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


def optimize(f: Callable, h: float, x: float, y=0):
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


def step() -> float:
    pass


def solve(f: Callable):
    """ 
    ODE Solver using Runge-Kutta method of 2nd order  

        Parameters:
        ----------
        f : RHS of given ODE 
    """

    data = get_data(filename)
    a, b, c, y_c = data[0]
    h_min, h_max, eps = data[1]
    h = (b - a) / 10
    steer = 1
    if c == b:
        steer = -1
    X = [c]
    Y = [y_c]
    x = c
    while a <= x and x <= b:
        y = Y[-1]
        Y.append(estimate(f, h, x, y))
        y_hat = optimize(f, h, x, y)
        x = x + steer * h
        X.append(x)
        eps_n = abs(Y[-1] - y_hat)
        h_e = np.power(eps_n/eps, 0.25) * h
        if h_e < h_min:
            h = h_min
        elif h_e > h_max:
            h = h_max
        else:
            h = h_e
    return X, Y


def f(x: float, y=0) -> float:
    """
    RHS of given ODE
    """

    return x**2


def main():
    X, Y = solve(f)
    for i in range(len(X)):
        print('f({:.4f}) = {:.4f}'.format(X[i], Y[i]))
    ''' --plot is probably needed here-- '''


if __name__ == "__main__":
    main()
