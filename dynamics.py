"""
This file holds the methods required to generate the system matrices of the linear quadratic control systems.

@author: Lucas Weitering
"""

import numpy as np


def boeing():
    A = np.array([[0.99, 0.03, -0.02, -0.32],
                  [0.01, 0.47, 4.7, 0],
                  [0.02, -0.06, 0.4, 0],
                  [0.01, -0.04, 0.72, 0.99]])
    B = np.array([[0.01, 0.99],
                  [-3.44, 1.66],
                  [-0.83, 0.44],
                  [-0.47, 0.25]])
    n, m = B.shape
    Q = np.eye(n)
    R = np.eye(m)
    return A, B, Q, R, "Boeing"


def uav():
    A = np.array([[1, 0.5, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0.5],
                  [0, 0, 0, 1]])
    B = np.array([[0.125, 0],
                  [0.5, 0],
                  [0, 0.125],
                  [0, 0.5]])
    n, m = B.shape
    Q = np.array([[1, 0, 0, 0],
                  [0, .1, 0, 0],
                  [0, 0, 2, 0],
                  [0, 0, 0, .2]])
    R = np.eye(m)
    return A, B, Q, R, "UAV"


def cart_pole():
    m = 1  # Mass of pendulum
    M = 5  # Mass of cart
    L = 2  # Length of pendulum
    g = -10  # Gravitational acceleration
    d = 1  # Dissipation on the cart

    A = np.array([[0, 1, 0, 0],
                  [0, -d/M, -m*g/M, 0],
                  [0, 0, 0, 1],
                  [0, -d/(M*L), -(m+M)*g/(M*L), 0]])
    B = np.array([[0], [1/M], [0], [1/(M*L)]])
    n, m = B.shape
    Q = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 250, 0],
                  [0, 0, 0, 500]])
    R = np.eye(m)
    return A, B, Q, R, "CartPole"

