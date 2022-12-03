# use backend tensorflow
import os
# comment out to enable gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import deepxde as dde
from typing import List, Tuple

dde.config.real.set_float64()
dde.config.set_default_float("float64")
from deepxde.backend import tf

if dde.backend.backend_name != 'tensorflow':
    raise Exception("set backend tensorflow with: python -m deepxde.backend.set_default_backend tensorflow")

b = 1.0
a = 0.4
nb_grid_points = 101

from deepxde.nn.tensorflow.fnn import activations
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
        activation = activations.get(activation)
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=123)

        self.denses = []
        for units in layer_sizes[2:]:
            self.denses.append(
                tf.keras.layers.Dense(
                    units,
                    activation=activation,
                    use_bias=False,
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
    def __init__(self, num_dom: int, num_bnd: int) -> None:
        self.num_dom: int = num_dom
        self.num_bnd: int = num_bnd
        self.geom: dde.geometry.Geometry
        self.bcs: List[dde.icbc.boundary_conditions.BC]
        self.data_dom: dde.data.Data
        self.data_bcs: List[dde.data.Data]

    def exact_sol(r: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def pde(r, T):
        raise NotImplementedError

    def phi(r):
        raise NotImplementedError

    def full_pde(r, T):
        raise NotImplementedError


class RectHole(Geometry):
    a = 0.4
    b = 1.0

    def __init__(self, num_dom: int, num_bnd: int) -> None:
        a = RectHole.a
        b = RectHole.b

        super().__init__(num_dom, num_bnd)

        self.geom: dde.geometry.Geometry = dde.geometry.Rectangle([-b, -b], [b, b]) - dde.geometry.Disk([0, 0], a / 2)

        self.bcs: List[dde.icbc.boundary_conditions.BC] = []
        self.bcs.append(dde.DirichletBC(geom=self.geom, func=RectHole._exact_sol, on_boundary=RectHole._ver_boundary))
        self.bcs.append(dde.DirichletBC(geom=self.geom, func=RectHole._exact_sol, on_boundary=RectHole._int_boundary))
        #self.bcs.append(dde.NeumannBC(geom=self.geom, func=RectHole._exact_grad_n, on_boundary=RectHole._hor_boundary))

        self.data_dom = dde.data.PDE(self.geom, RectHole.pde, [], num_domain=num_dom, num_boundary=0, num_test=101, solution=None)
        self.data_bcs = []
        for bc in self.bcs:
            self.data_bcs.append(dde.data.PDE(self.geom, RectHole.pde, [bc], num_domain=0, num_boundary=num_bnd, num_test=101, solution=__class__._exact_sol))

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

    def exact_sol(self, r: np.ndarray) -> np.ndarray:
        return __class__._exact_sol(r)

    def pde(r, T):
        dT_xx = dde.grad.hessian(T, r, i=0, j=0)
        dT_yy = dde.grad.hessian(T, r, i=1, j=1)
        return dT_xx + dT_yy

    def _phi(r):
        x = r[:, 0:1]
        y = r[:, 1:]
        return RectHole.a / (x**2 + y**2)**0.5 - 4

    def phi(self, r):
        return __class__._phi(r)

    def full_pde(r, T):
        return RectHole._pde(r, T) + RectHole._phi(r)

class PdeELM:
    def __init__(self, net: FNN2, geom: Geometry) -> None:
        self.net: FNN2 = net
        self.geom: Geometry = geom

    def _vec_grad(r, T) -> List[np.ndarray]:
        d_xy = []
        for k in range(T.shape[1]):
            d_xy.append(dde.grad.hessian(T, r, component=k, i=0, j=0) + dde.grad.hessian(T, r, component=k, i=1, j=1))
        return d_xy

    def fit_dom(self) -> np.ndarray:
        data = self.geom.data_dom
        self.model_: dde.Model = dde.Model(data, self.net)
        X, _, _ = data.train_next_batch()
        self.model_.compile(optimizer='adam', lr=0)

        # build loss matrix
        pde_pred = self.model_.predict(X, operator=PdeELM._vec_grad)
        A = np.empty((pde_pred[0].shape[0], len(pde_pred)))
        for k, v in enumerate(pde_pred):
                A[:,k] = v.flatten()

        # build b = -phi
        b = -self.geom.phi(X)

        #A_inv = np.linalg.pinv(A, rcond=1e-10)
        #self.w_out = A_inv.dot(b)
        return (A, b)

    def fit(self) -> Tuple[np.ndarray, np.ndarray]:
        A_dom, b_dom = self.fit_dom()
        A = A_dom
        b = b_dom
        for data_bc in self.geom.data_bcs:
            m = dde.Model(data_bc, self.net)
            m.compile(optimizer='adam', lr=0)
            X, b_bc, _ = data_bc.train_next_batch()
            A_bc = m.predict(X)
            A = np.concatenate([A, A_bc])
            b = np.concatenate([b, b_bc])
        A_inv = np.linalg.pinv(A, rcond=1e-10)
        self.w_out = A_inv.dot(b)
        return (A, b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        hl = self.model_.predict(X)
        return hl.dot(self.w_out)

    def get_weights(self) -> List[np.ndarray]:
        weights = []
        for d in self.net.denses:
            for w in d.weights:
                weights.append(w.numpy())
        weights.append(self.w_out)
        return weights

class ExactELM:
    def __init__(self, pde_elm: PdeELM) -> None:
        self.pde_elm: PdeELM = pde_elm
        self.geom: Geometry = pde_elm.geom
        self.weights = []

        rng = np.random.default_rng(123)
        for d in pde_elm.net.denses:
            for w in d.weights:
                self.weights.append(w.numpy())
        self.weights.append(rng.random((self.weights[-1].shape[1], 1)) - 0.5)

    def forward_prop(self, X: np.ndarray) -> List[np.ndarray]:
        H = []
        for weights in self.weights[:-1]:
            X = np.dot(X, weights)
            X = np.tanh(X)
            H.append(X)
        H.append(np.dot(X, self.weights[-1]))
        return H

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward_prop(X)[-1]

    def fit(self, X: np.ndarray) -> None:
        y = self.geom.exact_sol(X)
        a_l = self.forward_prop(X)
        H = a_l[-2]
        H_inv = np.linalg.pinv(H, rcond=1e-10)
        self.weights[-1] = np.dot(H_inv, y)

def print_dev_dom(elm) -> None:
    X, _, _ = elm.geom.data_dom.train_next_batch()
    y_pred = elm.predict(X)
    y_true = elm.geom.exact_sol(X)
    deviation = rms(y_pred, y_true)
    print(f"{elm.__class__.__name__} RMS Deviation train:\t{deviation}")
    Xt, _, _ = elm.geom.data_dom.test()
    y_pred_test = elm.predict(Xt)
    y_true_test = elm.geom.exact_sol(Xt)
    deviation_test = rms(y_pred_test, y_true_test)
    print(f"{elm.__class__.__name__} RMS Deviation test:\t{deviation_test}\n")

def rms(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return ((y_pred - y_true)**2).mean()**0.5

def compare_elm_inp_weights(exact: ExactELM, pde: PdeELM) -> bool:
    for w_e, w_p in zip(exact.weights[:-1], pde.get_weights()[:-1]):
        dif = np.isclose(w_e, w_p)
        if dif.sum() != dif.size:
            return False
    return True

if __name__=="__main__":
    num_dom = 2048
    num_bnd = 512
    num_tst = 101

    inp_dim = 2
    layers = [inp_dim] + [32, 128, 512]
    net = FNN2([2] + layers, "tanh", "Glorot uniform")

    geom = RectHole(num_dom=num_dom, num_bnd=num_bnd)
    X, _, _ = geom.data_dom.train_next_batch()

    pde_elm = PdeELM(net, geom)
    A_pde, b = pde_elm.fit()
    #A_pde, b = pde_elm.fit_dom()

    exact_elm = ExactELM(pde_elm)
    exact_elm.fit(X)

    print(f"\nNot trained weights are the same: {compare_elm_inp_weights(exact=exact_elm, pde=pde_elm)}\n")
    print_dev_dom(pde_elm)
    print_dev_dom(exact_elm)

    exact_w = exact_elm.weights[-1]
    pde_w = pde_elm.get_weights()[-1]
    exact_y = exact_elm.predict(X)
    pde_y = pde_elm.predict(X)

    # calculate loss matrix with help of exact solution: w x b^-1 = A^-1
    b_inv = np.linalg.pinv(b, rcond=1e-10)
    w = exact_w
    A_inv = w.dot(b_inv)
    A = np.linalg.pinv(A_inv, rcond=1e-10)

    rtol = 1e-6
    close = np.isclose(A, A_pde, rtol=rtol)
    print(f"loss matrix is the same (tolerance: {rtol}):\t{close.sum() == A.size}")
    ...
