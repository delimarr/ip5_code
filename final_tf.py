"""first working elm, also executable on gpu"""
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import deepxde as dde
dde.config.set_random_seed(7913)
if dde.backend.backend_name != 'tensorflow':
    raise Exception("set backend tensorflow with: python -m deepxde.backend.set_default_backend tensorflow")

import numpy as np
import time
from typing import List, Tuple, Callable
import timeit
from memory_profiler import profile

gpu = tf.config.list_physical_devices('GPU')
GPU_FLG: bool = True
if len(gpu) == 0:
    GPU_FLG = False

DTYPE: str = "float32"
dde.config.set_default_float(DTYPE)
tf.keras.backend.set_floatx(DTYPE)
if DTYPE == "float32":
    dde.config.real.set_float32()
    DTYPE = np.float32
elif DTYPE == "float64":
    dde.config.real.set_float64()
    DTYPE = np.float64
else:
    raise Exception("Choose DTYPE between float32 and float64")

class Geometry:
    def __init__(self, num_dom: int, num_bnd: int, num_tst: int) -> None:
        self.num_dom: int = num_dom
        self.num_bnd: int = num_bnd
        self.num_tst: int = num_tst
        self.geom: dde.geometry.Geometry
        self.dirichlet_bcs: List[dde.icbc.boundary_conditions.BC]
        self.data_dom: dde.data.Data
        self.data_test: dde.data.Data
        self.dirichlet_bcs: List[dde.data.Data]

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

        self.dirichlet_bcs: List[dde.icbc.boundary_conditions.DirichletBC] = []
        self.dirichlet_bcs.append(dde.DirichletBC(geom=self.geom, func=RectHole._exact_sol, on_boundary=RectHole._ver_boundary))
        self.dirichlet_bcs.append(dde.DirichletBC(geom=self.geom, func=RectHole._exact_sol, on_boundary=RectHole._int_boundary))

        self.neumann_bcs: List[dde.icbc.boundary_conditions.NeumannBC] = []
        self.neumann_bcs.append(dde.NeumannBC(geom=self.geom, func=RectHole._exact_grad_n, on_boundary=RectHole._hor_boundary))

        self.data_dom = dde.data.PDE(self.geom, RectHole.pde, [], num_domain=num_dom, num_boundary=0, num_test=num_tst, solution=None)

        self.data_dirichlet: List[dde.data.PDE] = []
        for bc in self.dirichlet_bcs:
            self.data_dirichlet.append(dde.data.PDE(self.geom, RectHole.pde, [bc], num_domain=0, num_boundary=num_bnd, num_test=num_tst, solution=__class__._exact_sol))

        self.data_neumann: List[dde.data.PDE] = []
        for bc in self.neumann_bcs:
            self.data_neumann.append(dde.data.PDE(self.geom, RectHole.pde, [bc], num_domain=0, num_boundary=num_bnd, num_test=num_tst, solution=__class__._exact_grad_n))

        self.data_test = dde.data.PDE(self.geom, RectHole.full_pde, self.dirichlet_bcs + self.neumann_bcs, num_domain=num_dom, num_boundary=num_bnd, num_test=num_tst, solution=RectHole._exact_sol)

    def _ver_boundary(r, on_boundary):
        x, _ = r
        return on_boundary and np.isclose(x**2, RectHole.b**2)

    def _int_boundary(r, on_boundary):
        x, y = r
        return on_boundary and np.isclose((x**2 + y**2), (RectHole.a / 2)**2)

    def _exact_grad_n(r):
        # just on horizontal boundary defined?
        a = RectHole.a
        b = RectHole.b
        x = r[:, 0:1]
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
        return (RectHole.a / (x**2 + y**2)**0.5) - 4

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
        self.seed = seed
        self.layers = layers
        self.model: tf.keras.Sequential = _init_model(layers=layers, seed=seed)

    @profile
    def fit(
        self, 
        X_dom: np.ndarray, phi_dom: np.ndarray, 
        dirichlet_bcs: List[Tuple[np.ndarray, np.ndarray]] = [],
        neumann_bcs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        ) -> None:
        """Fit model to PDE-Loss and given boundary conditions.

        Args:
            X_dom (np.ndarray): Input on domain.
            phi_dom (np.ndarray): not linear part of pde
            dirichlet_bcs (List[Tuple[np.ndarray, np.ndarray]], optional): Containes tuple of points on boundary and solution. Defaults to [].
            neumann_bcs (List[Tuple[np.ndarray, np.ndarray, np.ndarray]], optional): Containes tuple of points on boundary, unit normal and solution. Defaults to [].
        """
        X_dom_tf = tf.convert_to_tensor(X_dom)
        loss = _get_loss(self.model, X_dom_tf)
        # TO DO: replace zeros with coefficent of T(x) * HL in PDE
        A = np.c_[np.zeros((loss.shape[0], 1), dtype=DTYPE), loss]
        b = -phi_dom

        for X_bc, y_bc in dirichlet_bcs:
            A_bc = self.model(X_bc)
            A_bc = np.c_[np.ones((A_bc.shape[0], 1), dtype=DTYPE), A_bc]
            A = np.concatenate([A, A_bc])
            b = np.concatenate([b, y_bc])

        for X_bc, u_normal, y_bc in neumann_bcs:
            X_bc_tf = tf.convert_to_tensor(X_bc)
            T_x = get_grad(self.model, X_bc_tf, 0, 1)
            T_y = get_grad(self.model, X_bc_tf, 1, 1)
            grad = np.dstack((T_x, T_y))
            A_bc = np.zeros((T_x.shape[0], T_x.shape[1] + 1), dtype=DTYPE)
            for row in range(T_x.shape[0]):
                for col in range(T_x.shape[1]):
                    A_bc[row][col+1] = np.inner(grad[row][col], u_normal[row])
            A = np.concatenate([A, A_bc])
            b = np.concatenate([b, y_bc])

        A_inv = np.linalg.pinv(A, rcond=1e-10)
        self.w_out_ = A_inv.dot(b)

        self.full_model_: tf.keras.Sequential = _init_model(layers=self.layers, seed=None)
        self.full_model_.add(tf.keras.layers.Dense(
                    units=1,
                    ))
        for i, layer in enumerate(self.model.layers):
            self.full_model_.layers[i].set_weights(layer.get_weights())
        self.full_model_.layers[-1].set_weights([self.w_out_[1:,:], self.w_out_[0,:].flatten()])

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

    def get_accuracy(self, X: np.ndarray, phi: Callable) -> float:
        loss = _get_loss(self.full_model_, tf.convert_to_tensor(X))
        b = phi(X)
        acc = loss + b
        return (acc**2).mean()**.5

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
        self.weights.append(rng.random((self.weights[-1].shape[1] + 1, 1), dtype=DTYPE) - 0.5)

    def forward_prop(self, X: np.ndarray) -> List[np.ndarray]:
        H = []
        for weights in self.weights[:-1]:
            X = np.c_[np.ones((X.shape[0], 1)), X]
            X = np.matmul(X, weights, dtype=DTYPE)
            X = np.tanh(X)
            H.append(X)
        X = np.c_[np.ones((X.shape[0], 1)), X]
        H.append(np.matmul(X, self.weights[-1], dtype=DTYPE))
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
    u_xx = get_grad(model, X_r, 0, 2)
    u_yy = get_grad(model, X_r, 1, 2)
    return u_xx + u_yy

@profile
def get_grad(model: tf.keras.Sequential, X_r: tf.Tensor, r: int, d: int = 1) -> np.ndarray:
    """Calculate the dth derivative from the model at r.

    Args:
        model (tf.keras.Sequential): model
        X_r (tf.Tensor): Input Matrix
        r (int): index of feature
        d (int): which derivative, d > 0. Default is 1. 

    Returns:
        np.ndarray: gradient
    """
    with tf.GradientTape(persistent=True) as tape:
        cols = []
        for i in range(X_r.shape[1]):
            cols.append(X_r[:, i])
            if r == i:
                tape.watch(cols[i])

        X = tf.stack(cols, axis=1)
        u = model(X)

        G = np.empty((X_r.shape[0], u.shape[1]), dtype= DTYPE)
        for k in range(u.shape[1]):
            u_x = u[:,k]
            x = cols[r]
            for _ in range(d):
                u_x = tape.gradient(u_x, x)
            G[:, k] = u_x.numpy()
    del tape
    return G

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
    num_dom = 2048
    num_bnd = 1024
    num_tst = 4444
    layers = [32, 128, 512]

    geom = RectHole(num_dom=num_dom, num_bnd=num_bnd, num_tst=num_tst)
    X_dom, _, _ = geom.data_dom.train_next_batch()
    phi_dom = geom.phi(X_dom)

    dirichlet_bcs = []
    X_dirichlet = []
    for data_bc in geom.data_dirichlet:
        X, b_bc, _ = data_bc.train_next_batch()
        dirichlet_bcs.append((X, b_bc))
        X_dirichlet.append(X)

    neumann_bcs = []
    X_neumann = []
    for data_bc in geom.data_neumann:
        X, b_bc, _ = data_bc.train_next_batch()
        unit_normal = geom.geom.boundary_normal(X)
        neumann_bcs.append((X, unit_normal, b_bc))
        X_neumann.append(X)
    
    pde_elm = PdeELM(layers, seed=(123, 456))
    #it1 = timeit.repeat(lambda: pde_elm.fit(X_dom, phi_dom, dirichlet_bcs, neumann_bcs), number=100, repeat=1)
    s = time.perf_counter()
    pde_elm.fit(X_dom, phi_dom, dirichlet_bcs, neumann_bcs)
    print(f"\nTraining in seconds: {time.perf_counter() - s}")
    print_dev_dom(pde_elm, geom)
    X, _, _ = geom.data_test.train_next_batch()
    print(f"Accuracy pde_elm:\t{pde_elm.get_accuracy(X, geom.phi)}")

    exact_elm = ExactELM(pde_elm)
    s = time.perf_counter()
    exact_elm.fit(X_dom, X_dirichlet, geom.exact_sol)
    print(f"\nTraining in seconds: {time.perf_counter() - s}")
    print_dev_dom(exact_elm, geom)
    exact_weights = exact_elm.weights[-1]
    pde_weights = pde_elm.w_out_

if __name__=="__main__":
    if GPU_FLG:
        main()
    else:
        with tf.device('/CPU:0'):
            main()