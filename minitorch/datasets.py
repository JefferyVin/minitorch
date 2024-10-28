import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """
    I am able to converge with 30 points 2 hidden layers, opt to using slightly high LR of .5, converged around 130 epochs
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """
    Same hyperparameters as above, converged around 100 epochs
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """
    I wasn't able to converge consistently with the previous hyperparameters (My fist try converged somehow), gradient boomed
    After added another hidden layer (3) I've noticed the model converging easier
    I do notice the 0.5 learning rate couldnt escape the local minima tho but I do believe It's already a high learning rate so I did not change it
    I bet the ReLU is causing the gradient booms

    I've increased the model hidden layer to 10 and it's converging very consistently (my theory is the poorly initialized parameters is disregarded as the well initialized parameters carries, its just pure RNG anyways), no longer stuck in local minima as easily anymore, however gradient booming still exists
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """
    For xor I felt 30 sample points are not enough to represent the function therefore I increased the sample points to 60 (I tested 30 sample points and indeed it works suboptimally)
    60 works
    (tried lowering hidden size, it did not go well)
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """
    60 sample points work but larger sample is better
    increasing hidden to 20 makes the circle more smooth but i'm worried about overfitting
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """
    Well, 10 hidden layers seems to do a bad job, so I bumped it to 100, also lowered learning rate as well as increased epoch to 0.05 and 150000 (eventually converged pretty well) however I'm disappointed cause I believe this function could be represented easier
    Conclusion:
    Failing to converge - increase hidden size :)
    nan loss - lower learning rate
    If you are worried about the low sample rate causes overfitting/bad representation, then increase the sample size
    Yep thats about it I dont think its worth over thinking about shitty MLP layers, they do work but they cant under stand easy things
    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
