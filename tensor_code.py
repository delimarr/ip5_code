# for tensorflow backend
loss_fn = MSE(y_true, y_pred)

# model
def outputs_losses(training, inputs, targets, auxiliary_vars, losses_fn):
    self.net.auxiliary_vars = auxiliary_vars
    # Don't call outputs() decorated by @tf.function above, otherwise the
    # gradient of outputs wrt inputs will be lost here.
    outputs_ = self.net(inputs, training=training)
    # Data losses
    losses = losses_fn(targets, outputs_, loss_fn, inputs, self)
    if not isinstance(losses, list):
        losses = [losses]
    # Regularization loss
    if self.net.regularizer is not None:
        losses += [tf.math.reduce_sum(self.net.losses)]
    losses = tf.convert_to_tensor(losses)
    # Weighted losses
    if loss_weights is not None:
        losses *= loss_weights
    return outputs_, losses        


def outputs_losses_train(inputs, targets, auxiliary_vars):
    return outputs_losses(
        True, inputs, targets, auxiliary_vars, self.data.losses_train
    )

r = result.outputs_losses_train(X_train, y_train, aux_vars)