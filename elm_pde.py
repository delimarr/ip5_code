import numpy as np
import deepxde as dde

from typing import List, Callable

class Geometry:
    def __init__(self) -> None:
        pass

    def exact_sol(self, r: np.ndarray) -> np.ndarray:
        ...

    def get_X(self, num_dom: int, num_bnd: int) -> np.ndarray:
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

    def _pde(r, T):
        dT_xx = dde.grad.hessian(T, r, i=0, j=0)
        dT_yy = dde.grad.hessian(T, r, i=1, j=1)
        x = r[:, 0:1]
        y = r[:, 1:]
        return dT_xx + dT_yy + RectHole.a / (x**2 + y**2)**0.5 - 4

    def get_X(self, num_dom: int, num_bnd: int, num_test: int) -> np.ndarray:
        self.data = dde.data.PDE(self.geom, RectHole._pde, self.bcs, num_domain=num_dom, num_boundary=num_bnd, num_test=num_test, solution=RectHole._exact_sol)
        return self.data.train_x_all, self.data.test()

    def exact_sol(self, r: np.ndarray) -> np.ndarray:
        return RectHole._exact_sol(r)

class PdeELM:
    def __init__(self, geometry: Geometry, layers: List[int], num_dom: int, num_bnd: int, num_test: int) -> None:
        self.geometry: Geometry = geometry
        self.layers: List[int] = layers
        self.num_dom: int = num_dom
        self.num_bnd: int = num_bnd
        self.num_test: int = num_test

        self.weights: List[float] = []
        for i in range(1, len(self.layers) - 1):
            self.weights_.append(np.random.random_sample((self.layers[i - 1] + 1, self.layers[i])) - 0.7)

        


if __name__=="__main__":
    
    elm = PdeELM(
         )


