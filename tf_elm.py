import tensorflow as tf
import numpy as np
from typing import List
import deepxde as dde

num_dom = 400
num_bnd = 100
num_tst = 300

def exact_sol(r):
    x = r[:, 0:1]
    y = r[:, 1:]
    return ((x**2 + y**2)**0.5 - (a / 2))**2

def ver_boundary(r, on_boundary):
    x, y = r
    return on_boundary and np.isclose(x**2, b**2)

def hor_boundary(r, on_boundary):
    x, y = r
    return on_boundary and np.isclose(y**2, b**2)

def int_boundary(r, on_boundary):
    x, y = r
    return on_boundary and np.isclose((x**2 + y**2), (a / 2)**2)

def exact_grad_n(r):
    x = r[:, 0:1]
    y = r[:, 1:]
    return b * (2 - a / (x**2 + b**2)**0.5)

def pde_bc(r, T):
    dT_x = dde.grad.jacobian(T, r, i = 0)
    dT_xx = dde.grad.jacobian(dT_x, r, i = 0)
    dT_y = dde.grad.jacobian(T, r, i = 1)
    dT_yy = dde.grad.jacobian(dT_y, r, i = 1)
    return dT_xx + dT_yy

def phi(r):
    x = r[:, 0:1]
    y = r[:, 1:]
    return a / (x**2 + y**2)**0.5 - 4

a = 0.4
b = 1.0
geom = dde.geometry.Rectangle([-b, -b], [b, b]) - dde.geometry.Disk([0, 0], a / 2)

bcs: List[dde.icbc.boundary_conditions.BC] = []
bcs.append(dde.DirichletBC(geom=geom, func=exact_sol, on_boundary=ver_boundary))
bcs.append(dde.DirichletBC(geom=geom, func=exact_sol, on_boundary=int_boundary))
data_bc = dde.data.PDE(geom, pde_bc, bcs, num_domain=0, num_boundary=num_bnd, num_test=num_tst)
data_dom = dde.data.PDE(geom, pde_bc, [], num_domain=num_dom, num_boundary=0, num_test=num_tst)

X_dom = tf.convert_to_tensor(data_dom.train_next_batch(num_tst)[0])
X_bc = tf.convert_to_tensor(data_bc.train_next_batch(num_tst)[0])

y_bc = exact_sol(X_bc)

def init_model(num_hidden_layers=1, num_neurons_per_layer=512):
    # Initialize a feedforward neural network
    model = tf.keras.Sequential()

    # Input is two-dimensional (time + one spatial dimension)
    model.add(tf.keras.Input(2))

    # Append hidden layers
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
            activation=tf.keras.activations.get('tanh'),
            kernel_initializer='glorot_normal'))
    model.add(tf.keras.layers.Dense(512))

    return model



def get_r(model, X_r):
    
    # A tf.GradientTape is used to compute derivatives in TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        # Split t and x to compute partial derivatives
        x, y = X_r[:, 0:1], X_r[:,1:2]

        # Variables t and x are watched during tape
        # to compute derivatives u_t and u_x
        tape.watch(x)
        tape.watch(y)

        # Determine residual 
        u = model(tf.stack([x[:,0], y[:,0]], axis=1))

        # Compute gradient u_x within the GradientTape
        # since we need second derivatives
        A = np.empty((X_r.shape[0], 512))
        for i in range(512):
            u_x = tape.gradient(u[:,i:i+1], x)
            u_y = tape.gradient(u[:,i:i+1], y)
            u_xx = tape.gradient(u_x, x)
            u_yy = tape.gradient(u_y, y)
            dxy = dde.backend.to_numpy(u_xx + u_yy)
            A[:,i] = dxy.flatten()

    del tape
    return A



def compute_loss(model, X_r, X_data, u_data):
    
    # Compute phi^r
    r = get_r(model, X_r)
    phi_r = tf.reduce_mean(tf.square(r))
    
    # Initialize loss
    loss = phi_r
    
    # Add phi^0 and phi^b to the loss
    for i in range(len(X_data)):
        u_pred = model(X_data[i])
        loss += tf.reduce_mean(tf.square(u_data[i] - u_pred))
    
    return loss

def rse(y_pred, y_true) -> float:
    return (y_pred**2 + y_true**2).mean()**0.5

m = init_model()
A = get_r(m, X_dom)
A_inv = np.linalg.pinv(A, rcond=1e-10)

y_true = dde.backend.to_numpy(exact_sol(X_dom))
b_ = phi(X_dom)
hl = dde.backend.to_numpy(m(X_dom))


...