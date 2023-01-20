import os
# comment out to enable gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import timeit
import numpy as np


import deepxde as dde

dde.config.real.set_float64()
dde.config.set_default_float("float64")
from deepxde.backend import tf
#tf.function(jit_compile=True)

b = 1.0
a = 0.4

nb_grid_points = 101

def mse(y_pred, y_true) -> float:
    return (y_pred**2 + y_true**2).mean()

def exact_sol(r):
    x = r[:, 0:1]
    y = r[:, 1:]
    return ((x**2 + y**2)**0.5 - (a / 2))**2


def pde(r, T):
    dT_x = dde.grad.jacobian(T, r, i = 0)
    dT_xx = dde.grad.jacobian(dT_x, r, i = 0)
    dT_y = dde.grad.jacobian(T, r, i = 1)
    dT_yy = dde.grad.jacobian(dT_y, r, i = 1)
    return dT_xx + dT_yy

def pde_nn(r, T):
    dT_xx = dde.grad.hessian(T, r, i=0, j=0)
    dT_yy = dde.grad.hessian(T, r, i=1, j=1)
    x = r[:, 0:1]
    y = r[:, 1:]
    return dT_xx + dT_yy + a / (x**2 + y**2)**0.5 - 4


def phi(r):
    x = r[:, 0:1]
    y = r[:, 1:]
    return a / (x**2 + y**2)**0.5 - 4

geom = dde.geometry.Rectangle([-b, -b], [b, b]) - dde.geometry.Disk([0, 0], a / 2)

def grid_points(x_coord, y_coord, geom) :
    xx, yy = np.meshgrid(x_coord, y_coord)
    all_points = np.vstack((xx.flatten(), yy.flatten())).T
    selector = geom.inside(all_points)
    return all_points[selector]

def ext_boundary(r, on_boundary):
    x, y = r
    return on_boundary and (x**2 + y**2) > (a / 2)**2

def int_boundary(r, on_boundary):
    x, y = r
    return on_boundary and np.isclose((x**2 + y**2), (a / 2)**2)

bcs = {'Dirichlet': [[None, ext_boundary], [None, int_boundary]], 'Neumann': []}

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

#bcs = {'Dirichlet': [[None, ver_boundary], [None, int_boundary]], 'Neumann': [[exact_grad_n, hor_boundary]]}

# Custom class for extreme learning machine (ELM).
# Locks all training parameters except the weight and bias of the last (activationless) layer.
# This way most linear PDE become linear optimisation problems that in theory require only one O(N²) step.
# It is yet to be tested whether this actually works as expected.
"""
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
"""

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
                   optimizer='adam', refinement=None, plots=False, hide=False, ELM=False
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

        if ELM:
            data_dom = dde.data.PDE(geom, pde, [], num_domain=num_dom, num_boundary=0, num_test=num_tst)
            bc0, bc1 = D_bcs
            data_bc0 = dde.data.PDE(geom, pde, bc0, num_domain=0, num_boundary=num_bnd, num_test=num_tst)
            data_bc1 = dde.data.PDE(geom, pde, bc1, num_domain=0, num_boundary=num_bnd, num_test=num_tst)
        else:
            data = dde.data.PDE(geom, pde, D_bcs + N_bcs, num_domain=num_dom, num_boundary=num_bnd, num_test=num_tst, solution=exact_sol)

        if ELM:
            net = dde.maps.FNN([2] + layers, "tanh", "Glorot uniform")
        else:
            net = dde.maps.FNN([2] + layers + [1], "tanh", "Glorot uniform")
            model = dde.Model(data, net)
            model.compile(optimizer, lr=lrn_rat, loss = "MSE")
            model.train(epochs=10000)


        if (ELM):
            X_dom_train, _, _ = data_dom.train_next_batch(num_tst)
            X_bc0_train, _, _ = data_bc0.train_next_batch(num_tst)
            X_bc1_train, _, _ = data_bc1.train_next_batch(num_tst)
            X_train = np.concatenate([X_dom_train, X_bc0_train, X_bc1_train])

            X_dom_test = data_dom.test()
            X_bc0_test = data_bc0.test()
            X_bc1_test = data_bc1.test()
            X_test = np.concatenate([X_dom_test, X_bc0_test, X_bc1_test])

            m = dde.Model(data_dom, net)
            m.compile(optimizer, lr=lrn_rat)

            A_dom = m.outputs_losses_train(X_dom_train, None, None)

            m.data = data_bc0
            A_bc0 = m.outputs_losses_train(X_bc0_train, None, None)[0]

            m.data = data_bc1
            A_bc1 = m.outputs_losses_train(X_bc1_train, None, None)[0]

            A_dom = dde.backend.to_numpy(A_dom)
            A_bc0 = dde.backend.to_numpy(A_bc0)
            A_bc1 = dde.backend.to_numpy(A_bc1)

            b_dom = phi(X_dom_train)
            b_bc0 = exact_sol(X_bc0_train)
            b_bc1 = exact_sol(X_bc1_train)

            A = np.concatenate([A_dom, A_bc0, A_bc1])
            b_ = np.concatenate([b_dom, b_bc0, b_bc1])

            A_inv = np.linalg.pinv(A, rcond=1e-10)
            w_trained = A_inv.dot(b_)
            w_rnd = np.random.random_sample((layers[-1], 1)) - 0.5

            y_pred_rnd = m.predict(X_train).dot(w_rnd)
            y_pred = m.predict(X_train).dot(w_trained)
            #y_pred_test = m.predict(X_test).dot(w_trained)
            y_true = exact_sol(X_train)
            #y_true_test = exact_sol(X_test)
            deviation = ((y_pred - y_true)**2).mean()**0.5
            print(f"RMS ELM train:\t{deviation}")
            deviation_rnd = ((y_pred_rnd - y_true)**2).mean()**0.5
            print(f"RMS ELM random:\t{deviation_rnd}")
            #deviation_test = ((y_pred_test - y_true_test)**2).mean()**0.5
            #print(f"RMS ELM test:\t{deviation_test}")
            return deviation

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
            np.linspace(bbox[0][0], bbox[1][0], nb_grid_points), 
            np.linspace(bbox[0][1], bbox[1][1], nb_grid_points), 
            geom)

        exct = exact_sol(my_points).flatten()
        pred = model.predict(my_points).flatten()
        loss = model.predict(my_points, operator = pde).flatten()

    fig = None
    if plots:
        # plt.close('all')
        # dde.saveplot(losshistory, train_state, issave=True, isplot=True)

        # f = plt.figure()
        # plt.gca().set_aspect(1)
        ...

    RMS_dev = np.sum((pred - exct)**2 / pred.shape[0])**0.5
    #RMS_dev = np.mean((pred-exct)**2)
    RMS_pde = np.sum(loss**2 / loss.shape[0])**0.5

    print("Overall computation took:", dt, " seconds")
    print("# Eval. Points:", pred.shape[0])
    print("RMS Deviation: ", RMS_dev)
    print("RMS PDE-Loss: ", RMS_pde)
    return model, dt, RMS_dev, RMS_pde, [], fig

"""
# normal ML
layers = [8, 16, 16, 8]

lrn_rat = 0.015
num_dom = 32786
num_bnd = 8192
num_tst = 7777
epochs = 20000
eval_pts = nb_grid_points

result = NN_calculation(geom, pde, exact_sol, bcs, layers, lrn_rat, num_dom, num_bnd, num_tst, epochs, eval_pts = nb_grid_points, plots=False)

# normal ML with refinment
ref ={
      'epochs': 50000,
      'options': {'maxcor': 50, 'ftol': 1e-20, 'gtol': 1e-20, 'maxiter': 50000},
      'optimizer': 'L-BFGS-B',
      'lrng_rt': 0.015
}

layers = [10] * 4

lrn_rat = 0.015
num_dom = 250
num_bnd = 90
num_tst = 340
epochs = 5000
eval_pts = 101

result = NN_calculation(geom, pde, exact_sol, bcs, layers, lrn_rat, num_dom, num_bnd, num_tst, epochs, eval_pts = nb_grid_points, plots=True, refinement=ref)
"""
# optimum
ref ={
      'epochs': 1,
      'options': {'maxcor': 50, 'ftol': 1e-20, 'gtol': 1e-20, 'maxiter': 50000},
      'optimizer': 'L-BFGS-B',
      'lrng_rt': 0.015
}

layers_nn = [10]*4
layers_elm = [128, 10_000]

lrn_rat = 0.015
num_dom = 1_000
num_bnd = 400
num_tst = 900
epochs = 5000
eval_pts = 301

nn_res = NN_calculation(geom, pde_nn, exact_sol, bcs, layers_nn, lrn_rat, num_dom, num_bnd, num_tst, epochs, eval_pts = eval_pts, plots=False, refinement=None, ELM=False)
#elm_res = NN_calculation(geom, pde, exact_sol, bcs, layers_elm, lrn_rat, num_dom, num_bnd, num_tst, epochs, eval_pts = eval_pts, plots=False, refinement=None, ELM=True)
...
"""
# ELM with refinement
ref ={
      'epochs': 50000,
      'options': {'maxcor': 50, 'ftol': 1e-20, 'gtol': 1e-20, 'maxiter': 50000},
      'optimizer': 'L-BFGS-B',
      'lrng_rt': 0.015
}

layers = [8, 32, 128, 512]

lrn_rat = 0.015
num_dom = 250
num_bnd = 90
num_tst = 340
epochs = 5000
eval_pts = nb_grid_points

result = NN_calculation(geom, pde, exact_sol, bcs, layers, lrn_rat, num_dom, num_bnd, num_tst, epochs, 
                        eval_pts = nb_grid_points, plots=False, refinement=ref, ELM=True)
"""