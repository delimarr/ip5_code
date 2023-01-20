import os, sys, timeit

import deepxde as dde
from deepxde.backend import tf
if dde.backend.backend_name != 'tensorflow.compat.v1':
    raise Exception("set backend tensorflow with: python -m deepxde.backend.set_default_backend tensorflow.compat.v1")

import numpy as np

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
    raise Exception("Choose DTYPE: float32 or float64")

def grid_points(x_coord, y_coord, geom) :
    xx, yy = np.meshgrid(x_coord, y_coord)
    all_points = np.vstack((xx.flatten(), yy.flatten())).T
    selector = geom.inside(all_points)
    return all_points[selector]

# Custom class for extreme learning machine (ELM).
# Locks all training parameters except the weight and bias of the last (activationless) layer.
# This way most linear PDE become linear optimisation problems that in theory require only one O(N²) step.
# It is yet to be tested whether this actually works as expected.
class ELM_fnn(dde.nn.tensorflow_compat_v1.fnn.FNN):
    def __init__(self, *args, **kwargs):
        self.__layers_counter__ = 0
        super(ELM_fnn, self).__init__(*args, **kwargs)

    def _dense(self, inputs, units, activation=None, use_bias=True):
        if self.__layers_counter__ < len(self.layer_size) - 3:
            layer_trainable = False
            layer_activation = None
        else:
            layer_trainable = True
            layer_activation = activation
        self.__layers_counter__ = self.__layers_counter__ + 1
        return tf.layers.dense(
            inputs,
            units,
            activation=layer_activation,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.regularizer,
            kernel_constraint=self.kernel_constraint,
            trainable = layer_trainable
        )


class HiddenPrints:
    """ Class for suppressing outputs """
    hide_outputs = True
    
    def __enter__(self) :
        if self.__class__.hide_outputs :
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb) :
        if self.__class__.hide_outputs :
            sys.stdout.close()
            sys.stdout = self._original_stdout

class Dummy_With:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Runs a single simulation for a set of hyperparameters
def NN_calculation(
                   geom, pde, exact_sol, bcs, layers, lrn_rat, num_dom, num_bnd, num_tst, epochs, eval_pts, 
                   optimizer='adam', refinement=None, hide=False, ELM=False
                   ):

    HP = HiddenPrints()

    hide_object = HP if hide else Dummy_With()
    print(hide_object)

    with hide_object:
        start = timeit.default_timer()

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

        data = dde.data.PDE(geom, pde, D_bcs + N_bcs, num_domain=num_dom, num_boundary=num_bnd, num_test=num_tst, solution=None)

        if ELM:
            net = ELM_fnn([2] + layers + [1], "tanh", "Glorot uniform")
        else:
            net = dde.maps.FNN([2] + layers + [1], "tanh", "Glorot uniform")
        model = dde.Model(data, net)

        model.compile(optimizer, lr=lrn_rat, loss = "MSE")
        # model.train(epochs=10000)
        losshistory, train_state = model.train(epochs=epochs)

        if refinement is not None:
            model.compile(refinement['optimizer'], lr=refinement['lrng_rt'], loss = "MSE")

            dde.optimizers.set_LBFGS_options(**(refinement['options']))
            #     maxcor=50,
            #     ftol=1e-20,
            # #     gtol=1e-20,
            #     maxiter=1e5,
            # #     maxfun=1e5,
            # )

            # losshistory, train_state = model.train(epochs = 50000)
            losshistory, train_state = model.train(epochs = refinement['epochs'])

        end = timeit.default_timer()
        dt = end - start

        bbox = geom.bbox
        my_points = grid_points(
            np.linspace(bbox[0][0], bbox[1][0], eval_pts), 
            np.linspace(bbox[0][1], bbox[1][1], eval_pts), 
            geom)

        exct = exact_sol(my_points).flatten()
        pred = model.predict(my_points).flatten()
        loss = model.predict(my_points, operator = pde).flatten()

    RMS_dev = np.sum((pred - exct)**2 / pred.shape[0])**0.5
    RMS_pde = np.sum(loss**2 / loss.shape[0])**0.5

    print("Overall computation took:", dt, " seconds")
    print("# Eval. Points:", pred.shape[0])
    print("RMS Deviation: ", RMS_dev)
    print("RMS PDE-Loss: ", RMS_pde)
    return model, dt, RMS_dev, RMS_pde, my_points

# original problem - nn

b = 1.0
a = 0.4

def exact_sol(r):
    x = r[:, 0:1]
    y = r[:, 1:]
    return ((x**2 + y**2)**0.5 - (a / 2))**2

def pde(r, T):
    dT_xx = dde.grad.hessian(T, r, i=0, j=0)
    dT_yy = dde.grad.hessian(T, r, i=1, j=1)
    x = r[:, 0:1]
    y = r[:, 1:]
    return dT_xx + dT_yy + a / (x**2 + y**2)**0.5 - 4

def ext_boundary(r, on_boundary):
    x, y = r
    return on_boundary and (x**2 + y**2) > (a / 2)**2

def int_boundary(r, on_boundary):
    x, y = r
    return on_boundary and np.isclose((x**2 + y**2), (a / 2)**2)

def ver_boundary(r, on_boundary):
    x, y = r
    return on_boundary and np.isclose(x**2, b**2)

def hor_boundary(r, on_boundary):
    x, y = r
    return on_boundary and np.isclose(y**2, b**2)

def exact_grad_n(r):
    x = r[:, 0:1]
    y = r[:, 1:]
    return b * (2 - a / (x**2 + b**2)**0.5)

def int_boundary(r, on_boundary):
    x, y = r
    return on_boundary and np.isclose((x**2 + y**2), (a / 2)**2)

geom = dde.geometry.Rectangle([-b, -b], [b, b]) - dde.geometry.Disk([0, 0], a / 2)
bcs = {'Dirichlet': [[None, ver_boundary], [None, int_boundary]], 'Neumann': [[exact_grad_n, hor_boundary]]}

# optimum nn, paper
ref ={
      'epochs': 50000,
      'options': {'maxcor': 50, 'ftol': 1e-20, 'gtol': 1e-20, 'maxiter': 50000},
      'optimizer': 'L-BFGS-B',
      'lrng_rt': 0.015
}

layers_nn = [10] * 4
layers_elm = [8, 32, 128, 512]

lrn_rat = 0.015
num_dom = 400
num_bnd = 100
num_tst = 420
epochs = 5000
eval_pts = 101

result = NN_calculation(geom, pde, exact_sol, bcs, layers_elm, lrn_rat, num_dom, num_bnd, num_tst, epochs, eval_pts = eval_pts, refinement=None, ELM=True)

# ELM, with ref
# Deviation:  0.01712401989699179
# RMS PDE-Loss:  0.03879041685715247

# ELM
# RMS Deviation:  0.016067377924665123
# RMS PDE-Loss:  0.028863664297220155

# NN, with ref
# RMS Deviation:  0.00030713311701598996
# RMS PDE-Loss:  0.004304789312629518

# NN
# RMS Deviation:  0.041540999819840746
# RMS PDE-Loss:  0.03427840180951247