# ersetze heassian in pde 
import time
from typing import Callable, List

#import cupy as cp
# speedup ~3
import deepxde as dde
import numpy as np
import numpy as cp
import tensorflow as tf


def add_ones(X: cp.ndarray) -> cp.ndarray:
    return np.c_[np.ones((X.shape[0], 1)), X]

class Geometry:
    def __init__(self) -> None:
        pass

    def exact_sol(self, r: np.ndarray) -> np.ndarray:
        ...

    def get_X(self, num_dom: int, num_bnd: int) -> np.ndarray:
        ...

    def phi(r) -> np.ndarray:
        ...

    def pde(r, T) -> float:
        ...

class RectHole(Geometry):
    a = 0.4
    b = 1.0

    def __init__(self) -> None:
        a = RectHole.a
        b = RectHole.b
        self.geom: dde.geometry.Geometry = dde.geometry.Rectangle([-b, -b], [b, b]) - dde.geometry.Disk([0, 0], a / 2)

        self.bcs: List[dde.icbc.boundary_conditions.BC] = []
        self.bcs.append(dde.DirichletBC(geom=self.geom, func=RectHole._exact_sol, on_boundary=RectHole._ver_boundary))
        self.bcs.append(dde.DirichletBC(geom=self.geom, func=RectHole._exact_sol, on_boundary=RectHole._int_boundary))
        self.bcs.append(dde.NeumannBC(geom=self.geom, func=RectHole._exact_grad_n, on_boundary=RectHole._hor_boundary))

    def _ver_boundary(r, on_boundary):
        x, _ = r
        return on_boundary and np.isclose(x**2, RectHole.b**2)

    def _int_boundary(r, on_boundary):
        x, y = r
        return on_boundary and np.isclose((x**2 + y**2), (RectHole.a / 2)**2)

    def _exact_grad_n(r):
        a = RectHole.a
        b = RectHole.b
        x = r[:, 0:1]
        _ = r[:, 1:]
        return b * (2 - a / (x**2 + b**2)**0.5)

    def _hor_boundary(r, on_boundary):
        _, y = r
        return on_boundary and np.isclose(y**2, RectHole.b**2)

    def _exact_sol(r: np.ndarray) -> np.ndarray:
        x = r[:, 0:1]
        y = r[:, 1:]
        return ((x**2 + y**2)**0.5 - (RectHole.a / 2))**2

    def pde(self, r, T):
        dT_x = dde.grad.jacobian(T, r, i = 0)
        dT_xx = dde.grad.jacobian(dT_x, r, i = 0)
        dT_y = dde.grad.jacobian(T, r, i = 1)
        dT_yy = dde.grad.jacobian(dT_y, r, i = 1)
        return dT_xx + dT_yy

    def phi(self, r: np.ndarray) -> float:
        return ((RectHole.a/np.sqrt(r**2)) - 4).sum()

    def get_X(self, num_dom: int, num_bnd: int) -> np.ndarray:
        self.data = dde.data.PDE(self.geom, RectHole.pde, self.bcs, num_domain=num_dom, num_boundary=num_bnd, num_test=0, solution=RectHole._exact_sol)
        return self.data.train_x_all

    def exact_sol(self, r: np.ndarray) -> np.ndarray:
        return RectHole._exact_sol(r)

class ELM:
    def __init__(self, layers: List[int], activation_func: Callable, L: int) -> None:
        """Initialize elm.

        Args:
            layers (List[int]): layersizes without input and output layer
            activation_func (Callable): _description_
            L (int): number of features in X
        """
        for i in range(len(layers)):
            layers[i] -= 1
        self.layers: List[int] = [L] + layers + [1]
        self.activation_func: Callable = activation_func
        self.L: int = L

        self.weights_: List[np.ndarray] = []
        for i in range(1, len(self.layers) - 1):
            self.weights_.append(cp.random.random_sample((self.layers[i - 1] + 1, self.layers[i])) - 0.5)
        self.weights_.append(cp.random.random_sample((self.layers[-2], self.layers[-1])) - 0.5)

    def forward_prop(self, X: cp.ndarray) -> List[cp.ndarray]:
        H = []
        for weights in self.weights_[:-1]:
            X = add_ones(X)
            X = cp.dot(X, weights)
            X = self.activation_func(X)
            H.append(X)
        H.append(cp.dot(X, self.weights_[-1]))
        return H

    def fit(self, X: cp.ndarray, y: cp.ndarray) -> None:
        a_l = self.forward_prop(X)
        H = a_l[-2]
        H = add_ones(H)
        s = time.perf_counter()
        H_inv = cp.linalg.pinv(H, rcond=1e-10)
        self.inv_time: float = time.perf_counter() - s
        self.weights_[-1] = cp.dot(H_inv, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward_prop(X)[-1]

    def mse(self, X: cp.ndarray, y: cp.ndarray) -> float:
        y_hat = self.forward_prop(X)[-1]
        return ((y - y_hat)**2).mean()

    def rms(self, X: cp.ndarray, y: cp.ndarray) -> float:
        return self.mse(X, y)**0.5

    def fit_to_pde(self, geom: Geometry, X: np.ndarray):
        b = np.apply_along_axis(func1d=geom.phi, axis=1, arr=X)
        hl = self.forward_prop(X_train)[-2]
        A = geom.pde(X, hl)
        self.weights_[-1] = np.linalg.pinv(A, rcond=1e-8).dot(b)
        ...

def pde_appr(elm: ELM, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x1 = x.copy()
    x2 = x.copy()
    x1[0] += eps
    x2[0] -= eps
    second_der_x = (elm.derv_i(x1, 0, -2) - elm.derv_i(x2, 0, -2))/(2*eps)
    y1 = x.copy()
    y2 = x
    y1[0] += eps
    y2[0] -= eps
    second_der_y = (elm.derv_i(y1, 1, -2) - elm.derv_i(y2, 1, -2))/(2*eps)
    return second_der_x + second_der_y




if __name__=="__main__":
    g = RectHole()
    #X_train = cp.array(g.get_X(num_dom=400, num_bnd=0), dtype=cp.float64)
    X_train = cp.array(g.get_X(num_dom=2048, num_bnd=512), dtype=cp.float64)
    #X_train = cp.array(g.get_X(num_dom=16384, num_bnd=4096), dtype=cp.float64)
    y_train = cp.array(g.exact_sol(X_train), dtype=cp.float64)

    X_test = cp.array(g.get_X(num_dom=1001, num_bnd=0), dtype=cp.float64)
    #X_test = cp.array(g.get_X(num_dom=301, num_bnd=101), dtype=cp.float64)
    y_test = cp.array(g.exact_sol(X_test), dtype=cp.float64)

    print("start training")
    start = time.perf_counter()
    layers = [32, 128, 512]
    elm = ELM(layers = layers, activation_func=cp.tanh, L=2)
    elm.fit_to_pde(g, X_train)
    elm.fit(X_train, y_train)
    end_training = time.perf_counter()
    train_rms = elm.rms(X_train, y_train)
    test_rms = elm.rms(X_test, y_test)
    end_eval = time.perf_counter()
    print(f"RMS deviation training: {train_rms}")
    print(f"RMS deviation test: {test_rms}")
    print(f"training in seconds: {end_training - start}")
    print(f"inv time: {elm.inv_time}")
    print(f"evaluation in seconds: {end_eval - end_training}")