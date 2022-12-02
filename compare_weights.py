import os
# comment out to enable gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import timeit
import numpy as np
import deepxde as dde
from typing import List, Callable

dde.config.real.set_float64()
dde.config.set_default_float("float64")
from deepxde.backend import tf
#tf.function(jit_compile=True)

b = 1.0
a = 0.4
nb_grid_points = 101

from deepxde.nn.tensorflow.fnn import activations
from deepxde.nn.tensorflow.fnn import initializers
from deepxde.nn.tensorflow.fnn import regularizers

class FNN2(dde.nn.tensorflow.nn.NN):
    """Fully-connected neural network, with activation function in output layer."""
    def __init__(
        self,
        layer_sizes,
        activation,
        kernel_initializer,
        regularization=None,
        dropout_rate=0,
    ):
        super().__init__()
        self.regularizer = regularizers.get(regularization)
        self.dropout_rate = dropout_rate

        self.denses = []
        activation = activations.get(activation)
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=123)
        for units in layer_sizes[1:-1]:
            self.denses.append(
                tf.keras.layers.Dense(
                    units,
                    activation=activation,
                    kernel_initializer=initializer,
                    kernel_regularizer=self.regularizer,
                )
            )
            if self.dropout_rate > 0:
                self.denses.append(tf.keras.layers.Dropout(rate=self.dropout_rate))

        self.denses.append(
            tf.keras.layers.Dense(
                layer_sizes[-1],
                activation=activation,
                kernel_initializer=initializer,
                kernel_regularizer=self.regularizer,
            )
        )

    def call(self, inputs, training=False):
        y = inputs
        if self._input_transform is not None:
            y = self._input_transform(y)
        for f in self.denses:
            y = f(y, training=training)
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y

class Geometry:
    def __init__(self, num_dom: int, num_bnd: int, num_tst: int) -> None:
        self.num_dom: int = num_dom
        self.num_bnd: int = num_bnd
        self.num_tst: int = num_tst
        self.geom: dde.geometry.Geometry
        self.bcs: List[dde.icbc.boundary_conditions.BC]
        self.data_dom: dde.data.Data
        self.data_bcs: List[dde.data.Data]

    def exact_sol(r: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _pde(r, T):
        raise NotImplementedError

    def _phi(r):
        raise NotImplementedError

    def _full_pde(r, T):
        raise NotImplementedError


class RectHole(Geometry):
    a = 0.4
    b = 1.0

    def __init__(self, num_dom: int, num_bnd: int, num_tst: int) -> None:
        a = RectHole.a
        b = RectHole.b

        super.__init__(num_dom, num_bnd, num_tst)

        self.geom: dde.geometry.Geometry = dde.geometry.Rectangle([-b, -b], [b, b]) - dde.geometry.Disk([0, 0], a / 2)

        self.bcs: List[dde.icbc.boundary_conditions.BC] = []
        self.bcs.append(dde.DirichletBC(geom=self.geom, func=RectHole.exact_sol, on_boundary=RectHole._ver_boundary))
        self.bcs.append(dde.DirichletBC(geom=self.geom, func=RectHole.exact_sol, on_boundary=RectHole._int_boundary))
        self.bcs.append(dde.NeumannBC(geom=self.geom, func=RectHole._exact_grad_n, on_boundary=RectHole._hor_boundary))

        self.data_dom = dde.data.PDE(self.geom, RectHole._pde, [], num_domain=num_dom, num_boundary=0, num_test=num_tst, solution=None)
        self.data_bcs = []
        for bc in self.bcs:
            self.data_bcs.append(dde.data.PDE(self.geom, RectHole._pde, [bc], num_domain=0, num_boundary=num_bnd, num_test=num_tst, solution=None))

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

    def exact_sol(r: np.ndarray) -> np.ndarray:
        x = r[:, 0:1]
        y = r[:, 1:]
        return ((x**2 + y**2)**0.5 - (RectHole.a / 2))**2

    def _pde(r, T):
        dT_xx = dde.grad.hessian(T, r, i=0, j=0)
        dT_yy = dde.grad.hessian(T, r, i=1, j=1)
        return dT_xx + dT_yy

    def _phi(r):
        x = r[:, 0:1]
        y = r[:, 1:]
        return RectHole.a / (x**2 + y**2)**0.5 - 4

    def _full_pde(r, T):
        return RectHole._pde(r, T) + RectHole._phi(r)

class ExactELM:
    def __init__(self) -> None:
        pass

class PdeELM:
    def __init__(self) -> None:
        pass

def compare_elm(exact: ExactELM, pde: PdeELM) -> bool:
    return False