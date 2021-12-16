""" Optimize a keras model using scipy.optimize
"""
import numpy as np
from scipy.optimize import minimize
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K  # pylint: disable=import-error

from tensorflow.python.keras.engine import data_adapter


class ScipyOptimizer():
    """ Implements a training function that uses scipy optimize in order
        to determine the weights for the model.

        The minimize function expects to be able to attempt multiple solutions
        over the model. It calls a function which collects all gradients for
        all steps and then returns the gradient information to the optimizer.
    """

    def __init__(self, model, method='cg', verbose=1, maxiter=1):
        self.model = model
        self.method = method
        self.verbose = verbose
        self.maxiter = maxiter
        if model.run_eagerly:
            self.func = model.__call__
        else:
            self.func = tf.function(
                model.__call__, experimental_relax_shapes=True)

    def _update_weights(self, x):
        x_offset = 0
        for var in self.model.trainable_variables:
            shape = var.get_shape()
            w_size = np.prod(shape)
            value = np.array(x[x_offset:x_offset+w_size]).reshape(shape)
            K.set_value(var, value)
            x_offset += w_size
        assert x_offset == len(x)

    def _fun_generator(self, x, iterator):
        """ Function optimized by scipy minimize.

            Returns function cost and gradients for all trainable variables.
        """
        model = self.model
        self._update_weights(x)
        losses = []

        dataset = iterator._dataset  # pylint:disable=protected-access
        assert dataset is not None
        iterator = iter(dataset)

        size = dataset.cardinality().numpy()
        if size > 0:
            n_steps = (size + dataset.batch_size - 1) // dataset.batch_size
        else:
            n_steps = None

        progbar = keras.utils.Progbar(n_steps, verbose=self.verbose)

        with tf.GradientTape() as tape:
            for step, data in enumerate(iterator):
                data = data_adapter.expand_1d(data)
                x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(
                    data)
                y_pred = self.func(x, training=True)
                loss = model.compiled_loss(y, y_pred, sample_weight,
                                           regularization_losses=model.losses)
                progbar.update(step, [('loss', loss.numpy())])
                losses.append(loss)
            xloss = tf.reduce_mean(tf.stack(losses))
            grads = tape.gradient(xloss, model.trainable_variables)

        cost = xloss.numpy()

        if all(isinstance(x, tf.Tensor) for x in grads):
            xgrads = np.concatenate([x.numpy().reshape(-1) for x in grads])
            return cost, xgrads

        if all(isinstance(x, tf.IndexedSlices) for x in grads):
            xgrad_list = []
            for var, grad in zip(model.trainable_variables, grads):
                value = tf.Variable(np.zeros(var.shape), dtype=var.dtype)
                value.assign_add(grad)
                xgrad_list.append(value.numpy())
            xgrads = np.concatenate([x.reshape(-1) for x in xgrad_list])
            return cost, xgrads

        raise NotImplementedError()
        return -1, np.array([])  # pylint:disable=unreachable

    def train_function(self, iterator):
        """ Called by model fit.
        """
        min_options = {
            'maxiter': self.maxiter,
            'disp': bool(self.verbose),
        }

        var_list = self.model.trainable_variables
        x0 = np.concatenate([x.numpy().reshape(-1) for x in var_list])

        result = minimize(
            self._fun_generator, x0, method=self.method, jac=True,
            options=min_options, args=(iterator,))

        self._update_weights(result['x'])
        return {'loss': result['fun']}


def make_train_function(model, **kwargs):
    """ Returns a function that will be called to train the model.

        model._steps_per_execution must be set in order for train function to
        be called once per epoch.
    """
    model._assert_compile_was_called()  # pylint:disable=protected-access
    model._configure_steps_per_execution(tf.int64.max)  # pylint:disable=protected-access
    opt = ScipyOptimizer(model, **kwargs)
    return opt.train_function
