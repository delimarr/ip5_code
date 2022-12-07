import tensorflow as tf
import numpy as np
import deepxde as dde
from compare_elm import RectHole
import time


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
        A = np.empty((X_r.shape[0], u.shape[1]))

        # Compute gradient u_x within the GradientTape
        # since we need second derivatives
        for k in range(u.shape[1]):
            hl = u[:,k]
            u_x = tape.gradient(hl, x)
            u_y = tape.gradient(hl, y)
            u_xx = tape.gradient(u_x, x)
            u_yy = tape.gradient(u_y, y)
            col = (u_xx + u_yy).numpy()
            A[:, k] = col.flatten()
    del tape
    return A


if __name__=="__main__":
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(2))
    initializer_w = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=123)
    initializer_b = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=321)
    layers = [32, 128]
    for layer in layers:
        model.add(tf.keras.layers.Dense(layer,
                    activation=tf.keras.activations.get('tanh'),
                    kernel_initializer=initializer_w,
                    use_bias=True,
                    bias_initializer=initializer_b))
    
    num_dom = 100
    num_bnd = 5
    num_tst = 101

    geom = RectHole(num_dom=num_dom, num_bnd=num_bnd, num_tst=num_tst)
    X_r, _, _ = geom.data_dom.train_next_batch()
    X_r = tf.convert_to_tensor(X_r)

    s = time.perf_counter()
    A = get_r(model, X_r)
    print(f"time tf in seconds: {time.perf_counter() - s}")

    ...



     
