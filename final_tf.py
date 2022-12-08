import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import time
from typing import List, Tuple, Callable

import deepxde as dde
dde.config.set_random_seed(7913)

if dde.backend.backend_name != 'tensorflow':
    raise Exception("set backend tensorflow with: python -m deepxde.backend.set_default_backend tensorflow")
import numpy as np

gpu = tf.config.list_physical_devices('GPU')
print(gpu)
GPU_FLG: bool = False
if len(gpu) == 0:
    GPU_FLG = False

class Geometry:
    def __init__(self, num_dom: int, num_bnd: int, num_tst: int) -> None:
        self.num_dom: int = num_dom
        self.num_bnd: int = num_bnd
        self.num_tst: int = num_tst
        self.geom: dde.geometry.Geometry
        self.bcs: List[dde.icbc.boundary_conditions.BC]
        self.data_dom: dde.data.Data
        self.data_test: dde.data.Data
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

    def __init__(self, num_dom: int, num_bnd: int, num_tst: int) -> None:
        a = RectHole.a
        b = RectHole.b

        super().__init__(num_dom, num_bnd, num_tst)

        self.geom: dde.geometry.Geometry = dde.geometry.Rectangle([-b, -b], [b, b]) - dde.geometry.Disk([0, 0], a / 2)

        self.bcs: List[dde.icbc.boundary_conditions.BC] = []
        self.bcs.append(dde.DirichletBC(geom=self.geom, func=RectHole._exact_sol, on_boundary=RectHole._ver_boundary))
        self.bcs.append(dde.DirichletBC(geom=self.geom, func=RectHole._exact_sol, on_boundary=RectHole._int_boundary))
        self.bcs.append(dde.NeumannBC(geom=self.geom, func=RectHole._exact_grad_n, on_boundary=RectHole._hor_boundary))
        """
        # code snippet boundary conditions from notebook
        bcs = {'Dirichlet': [[None, ver_boundary], [None, int_boundary]], 'Neumann': [[exact_grad_n, hor_boundary]]}
        D_bcs = []
        for bc in bcs['Dirichlet']:
            if bc[0] is None:
                f = exact_sol
            else:
                f = bc[0]
            D_bcs.append(dde.DirichletBC(geom, f, bc[1]))

        N_bcs = []
        for bc in bcs['Neumann']:
            N_bcs.append(dde.NeumannBC(geom, bc[0], bc[1]))
        """

        self.data_dom = dde.data.PDE(self.geom, RectHole.pde, [], num_domain=num_dom, num_boundary=0, num_test=num_tst, solution=None)
        self.data_bcs = []
        for bc in self.bcs:
            self.data_bcs.append(dde.data.PDE(self.geom, RectHole.pde, [bc], num_domain=0, num_boundary=num_bnd, num_test=num_tst, solution=__class__._exact_sol))

        self.data_test = dde.data.PDE(self.geom, RectHole.full_pde, self.bcs, num_domain=num_dom, num_boundary=num_bnd, num_test=num_tst, solution=RectHole._exact_sol)

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
    def __init__(self, layers: List[int], seed: Tuple[int, int] = None) -> None:
        """Initialize PdeELM.

        Args:
            layers (List[int]): hiddenlayer sizes without bias
            seed (Tuple[int, int], optional): Seed for weight, bias init. Defaults to None.
        """
        self.model: tf.keras.Sequential = _init_model(layers=layers, seed=seed)

    def fit(self, X_dom: np.ndarray, phi_dom: np.ndarray, data_bcs: List[Tuple[np.ndarray, np.ndarray]] = []) -> None:
        """Fit model to PDE-Loss and given boundary conditions.

        Args:
            X_dom (np.ndarray): Input on domain.
            phi_dom (np.ndarray): not linear part of pde
            data_bcs (List[Tuple[np.ndarray, np.ndarray]], optional): Containes tuple of points on boundary and solution. Defaults to [].
        """
        X_dom_tf = tf.convert_to_tensor(X_dom)
        loss = _get_loss(self.model, X_dom_tf)
        # TO DO: replace zeros with coefficent of T(x) * HL in PDE
        A = np.c_[np.zeros((loss.shape[0], 1), dtype=np.float64), loss]
        b = -phi_dom
        for X_bc, y_bc in data_bcs:
            A_bc = self.model(X_bc)
            A_bc = np.c_[np.ones((A_bc.shape[0], 1), dtype=np.float64), A_bc]
            A = np.concatenate([A, A_bc])
            b = np.concatenate([b, y_bc])
        A_inv = np.linalg.pinv(A, rcond=1e-10)
        self.w_out_ = A_inv.dot(b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict output, use only after fit!

        Args:
            X (np.ndarray): Input Matrix

        Returns:
            np.ndarray: y predicted
        """
        hl = self.model(X)
        hl = np.c_[np.ones((hl.shape[0], 1)), hl]
        return hl.dot(self.w_out_)

class ExactELM:
    def __init__(self, pde_elm: PdeELM) -> None:
        self.pde_elm: PdeELM = pde_elm
        self.weights = []

        pde_w = pde_elm.model.get_weights()
        for i in range(int(len(pde_w) / 2)):
            w = pde_w[2*i]
            b = pde_w[2*i + 1].reshape(1, w.shape[1])
            self.weights.append(np.concatenate([b, w]))
        rng = np.random.default_rng(123)
        self.weights.append(rng.random((self.weights[-1].shape[1] + 1, 1)) - 0.5)

    def forward_prop(self, X: np.ndarray) -> List[np.ndarray]:
        H = []
        for weights in self.weights[:-1]:
            X = np.c_[np.ones((X.shape[0], 1)), X]
            X = np.dot(X, weights)
            X = np.tanh(X)
            H.append(X)
        X = np.c_[np.ones((X.shape[0], 1)), X]
        H.append(np.dot(X, self.weights[-1]))
        return H

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward_prop(X)[-1]

    def fit(self, X_dom: np.ndarray, X_bc: List[np.ndarray], exact_sol: Callable) -> None:
        X = X_dom
        for bc in X_bc:
            X = np.concatenate([X, bc])
        y = exact_sol(X)
        a_l = self.forward_prop(X)
        H = a_l[-2]
        H = np.c_[np.ones((H.shape[0], 1)), H]
        H_inv = np.linalg.pinv(H, rcond=1e-10)
        self.weights[-1] = np.dot(H_inv, y)
    
def _init_model(layers: List[int], seed: Tuple[int, int] = None) -> tf.keras.Sequential:
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(2))
    seed_w = seed_b = None
    if seed is not None:
        seed_w, seed_b = seed
    initializer_w = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=seed_w)
    initializer_b = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=seed_b)
    for layer in layers:
        model.add(tf.keras.layers.Dense(layer,
                    activation=tf.keras.activations.get('tanh'),
                    kernel_initializer=initializer_w,
                    use_bias=True,
                    bias_initializer=initializer_b))
    return model

def _get_loss(model: tf.keras.Sequential, X_r: tf.Tensor) -> np.ndarray:
    with tf.GradientTape(persistent=True) as tape:
        x, y = X_r[:, 0:1], X_r[:,1:2]

        tape.watch(x)
        tape.watch(y)

        u = model(tf.stack([x[:,0], y[:,0]], axis=1))
        A = np.empty((X_r.shape[0], u.shape[1]))

        for k in range(u.shape[1]):
            hl = u[:,k]
            # TO DO: generalize derivatives
            u_x = tape.gradient(hl, x)
            u_y = tape.gradient(hl, y)
            u_xx = tape.gradient(u_x, x)
            u_yy = tape.gradient(u_y, y)

            # TO DO call PDE with generalized derivatives
            col = (u_xx + u_yy).numpy()
            A[:, k] = col.flatten()
    del tape
    return A

def print_dev_dom(elm: PdeELM, geom: Geometry) -> None:
    """Print deviation for trained elm on given geometry.

    Args:
        elm (PdeELM): pde elm
        geom (Geometry): geometry
    """
    X, _, _ = geom.data_test.train_next_batch()
    y_pred_train = elm.predict(X)
    y_true_train = geom.exact_sol(X)
    deviation_train = rms(y_pred_train, y_true_train)
    print(f"{elm.__class__.__name__} RMS Deviation train:\t{deviation_train}\n")

    Xt, _, _ = geom.data_test.test()
    y_pred_test = elm.predict(Xt)
    y_true_test = geom.exact_sol(Xt)
    deviation_test = rms(y_pred_test, y_true_test)
    print(f"{elm.__class__.__name__} RMS Deviation test:\t{deviation_test}\n")

def rms(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate deviation from exact solution (root mean squared error).

    Args:
        y_pred (np.ndarray): predicted values
        y_true (np.ndarray): exact values

    Returns:
        float: rms
    """
    return ((y_pred - y_true)**2).mean()**0.5

def main() -> None:
    num_dom = 250
    num_bnd = 90
    num_tst = 340
    layers = [8, 32, 128, 256]

    geom = RectHole(num_dom=num_dom, num_bnd=num_bnd, num_tst=num_tst)
    X_dom, _, _ = geom.data_dom.train_next_batch()
    phi_dom = geom.phi(X_dom)
    data_bcs = []
    X_bc = []
    for data_bc in geom.data_bcs:
        X, b_bc, _ = data_bc.train_next_batch()
        data_bcs.append((X, b_bc))
        X_bc.append(X)
    
    pde_elm = PdeELM(layers, seed=(123, 456))
    s = time.perf_counter()
    pde_elm.fit(X_dom, phi_dom, data_bcs)
    print(f"\nTraining in seconds: {time.perf_counter() - s}")
    print_dev_dom(pde_elm, geom)

    exact_elm = ExactELM(pde_elm)
    s = time.perf_counter()
    exact_elm.fit(X_dom, X_bc, geom.exact_sol)
    print(f"\nTraining in seconds: {time.perf_counter() - s}")
    print_dev_dom(exact_elm, geom)
    ...



if __name__=="__main__":
    if GPU_FLG:
        main()
    else:
        with tf.device('/CPU:0'):
            main()