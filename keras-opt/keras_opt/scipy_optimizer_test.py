""" Unit tests for scipy_optimizer
"""
from __future__ import print_function

import random
import unittest

import numpy as np
import numpy.testing

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Concatenate, Embedding, Dense, Dot, Input, InputLayer, Lambda  # pylint: disable=import-error
from tensorflow.keras.models import Sequential, Model  # pylint: disable=import-error

from scipy.sparse import dok_matrix
from sklearn.model_selection import train_test_split

import keras_opt.scipy_optimizer as scipy_optimizer


class MatrixDataGenerator(keras.utils.Sequence):
    """ Generate test data.
    """

    def __init__(self, matrix, batch_size=32):
        self._matrix = matrix
        self._batch_size = batch_size
        self._block_size = (matrix.size - 1) / batch_size + 1

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self._matrix.size == 0:
            return 0
        return int((self._matrix.size - 1) / self._batch_size) + 1

    def __getitem__(self, index):
        'Generate one batch of data'
        start = index * self._batch_size
        end = min(start + self._batch_size, self._matrix.size)
        X = np.empty((end - start, 2))  # pylint: disable=invalid-name
        X[:, 0] = self._matrix.row[start:end]
        X[:, 1] = self._matrix.col[start:end]
        y = self._matrix.data[start:end]
        return X, y


def make_test_matrix(shape, datapoints):
    """ Generate a sparse matrix with a specified number of datapoints.
    """
    matrix = dok_matrix(shape)
    for _ in range(datapoints):
        while True:
            row = random.randint(0, shape[0] - 1)
            col = random.randint(0, shape[1] - 1)
            if matrix.get((row, col)):
                continue
            value = 1
            if row >= shape[0] / 2:
                value += 2
            if col >= shape[1] / 2:
                value += 1
            matrix[row, col] = value
            break
    return matrix


def make_embedding_model(shape, embedding_size):
    """ matrix factorization model
    """
    coordinates = Input(shape=(2,), dtype=tf.int32)
    row_embedding = Embedding(shape[0], embedding_size, input_length=1)
    col_embedding = Embedding(shape[1], embedding_size, input_length=1)
    row = Lambda(lambda x: x[:, 0])(coordinates)
    col = Lambda(lambda x: x[:, 1])(coordinates)
    row_vecs = row_embedding(row)
    col_vecs = col_embedding(col)
    y_r = Dot(1)([row_vecs, col_vecs])
    model = Model(inputs=coordinates, outputs=y_r)
    model.compile(loss='mean_squared_error')
    return model


class ScipyOptimizerTest(unittest.TestCase):
    """ Unit tests for the scipy_optimizer module.
    """

    def setUp(self):
        random.seed(0)
        np.random.seed(0)
        tf.random.set_seed(0)

    def test_lr(self):
        """ Logistic regression.
        """
        model = Sequential()
        model.add(Dense(1, use_bias=False, input_dim=4))
        model.compile(loss='mse')

        def fn(vec):
            a, b, c, d = vec
            return 4*a + 2*b + 3*c + d

        inputs = np.random.rand(10, 4)
        outputs = np.zeros(inputs.shape[0])
        for i in range(inputs.shape[0]):
            outputs[i] = fn(inputs[i, :])

        model.train_function = scipy_optimizer.make_train_function(
            model, verbose=0, maxiter=30)
        hist = model.fit(inputs, outputs, epochs=1, verbose=False)
        self.assertTrue('loss' in hist.history)

        layers = [layer for layer in model.layers if layer.weights]
        w = layers[0].get_weights()[0].reshape(-1)
        numpy.testing.assert_almost_equal(w, [4.0, 2.0, 3.0, 1.0], decimal=4)

    def test_dataset_size(self):
        """ Logistic regression using a dataset with multiple batches.
        """
        model = Sequential()
        model.add(Dense(1, use_bias=False, input_dim=4))
        model.compile(loss='mse')

        def fn(vec):
            a, b, c, d = vec
            return 4*a + 2*b + 3*c + d

        inputs = np.random.rand(200, 4)
        outputs = np.zeros(inputs.shape[0])
        for i in range(inputs.shape[0]):
            outputs[i] = fn(inputs[i, :])

        model.train_function = scipy_optimizer.make_train_function(
            model, verbose=0, maxiter=20)
        hist = model.fit(inputs, outputs, epochs=1, verbose=False)
        self.assertTrue('loss' in hist.history)

        layers = [layer for layer in model.layers if layer.weights]
        w = layers[0].get_weights()[0].reshape(-1)
        numpy.testing.assert_almost_equal(w, [4.0, 2.0, 3.0, 1.0], decimal=4)

    def test_graph_mode(self):
        """ Ensure that the model is executed in graph mode.
        """
        def custom_layer(x):
            assert not tf.executing_eagerly()
            return tf.reduce_sum(x, axis=-1)

        model = Sequential()
        model.add(InputLayer(input_shape=(4,)))
        model.add(Dense(2))
        model.add(Lambda(custom_layer))
        model.compile(loss="mse")

        def objective_fn(x):
            a = np.dot(x, np.array([1, 2, 3, 4])[:, np.newaxis])
            b = np.dot(x, np.array([5, 6, 7, 8])[:, np.newaxis])
            return a + b

        x_data = np.random.uniform(size=40).reshape(10, 4)
        y = np.apply_along_axis(objective_fn, -1, x_data)

        model.train_function = scipy_optimizer.make_train_function(
            model, verbose=0, maxiter=25)
        hist = model.fit(x_data, y, epochs=1, verbose=False)
        self.assertLess(hist.history['loss'][-1], 1.0e-3)

    def test_2layer(self):
        """ logistic regression using an hidden layer
        """
        model = Sequential()
        model.add(Dense(3, use_bias=False, input_dim=4))
        model.add(Dense(1, use_bias=False))
        model.compile(loss='mse')

        def fn(vec):
            a, b, c, d = vec
            return a*b + 2*b + 3*c + d

        inputs = np.random.rand(10, 4)
        outputs = np.zeros(inputs.shape[0])
        for i in range(inputs.shape[0]):
            outputs[i] = fn(inputs[i, :])

        model.train_function = scipy_optimizer.make_train_function(
            model, verbose=0, maxiter=15)
        model.fit(inputs, outputs, verbose=False)

        pred = model.predict(inputs)
        delta = outputs - pred.reshape(-1)
        self.assertEqual(delta.shape, outputs.shape)
        self.assertLess(delta.sum(), 0.01)

    def test_fit_generator(self):
        """ Embedding generation using generators.
        """
        matrix = make_test_matrix((10, 10), 50)
        generator = MatrixDataGenerator(matrix.tocoo())
        model = make_embedding_model(matrix.shape, 3)
        model.train_function = scipy_optimizer.make_train_function(
            model, verbose=0, maxiter=200)
        hist = model.fit(generator, verbose=False)
        self.assertLess(hist.history['loss'][-1], 1.0e-3)

    def test_bfgs(self):
        """ Embedding generation using method bfgs.
        """
        matrix = make_test_matrix((10, 10), 50)
        generator = MatrixDataGenerator(matrix.tocoo())
        model = make_embedding_model(matrix.shape, 3)
        model.train_function = scipy_optimizer.make_train_function(
            model, method='bfgs', verbose=0, maxiter=200)
        hist = model.fit(generator, verbose=False)
        self.assertLess(hist.history['loss'][-1], 1.0e-3)

    def test_1dim(self):
        """ Input data with rank 1.
        """
        def test_fn(x):
            if x > 0.5:
                return 1
            return 0

        def make_model():
            inp = Input(shape=(1,))
            kinit = keras.initializers.RandomUniform(0.0, 1.0)
            h_layer = Dense(1, kernel_initializer=kinit,
                            activation='relu')(inp)
            outp = Dense(1, activation='sigmoid')(h_layer)
            return Model(inp, outp)

        model = make_model()
        model.compile(loss='mse')
        model.train_function = scipy_optimizer.make_train_function(
            model, verbose=0, maxiter=15)

        X = np.random.rand(100)  # pylint: disable=invalid-name
        y = np.vectorize(test_fn)(X)
        X_train, X_test, y_train, y_test = train_test_split(  # pylint: disable=invalid-name
            X, y, test_size=0.20, random_state=42)

        model.fit(X_train, y_train, verbose=False)
        self.assertLessEqual(model.evaluate(X_test, y_test), 1.0e-5)

    def test_val_data(self):
        """ Validation metrics
        """
        def test_fn(x):
            if x > 0.8:
                return 2
            return 0

        def make_model():
            inp = Input(shape=(1,))
            h_layer = Dense(1,
                            kernel_initializer=keras.initializers.RandomUniform(
                                0.0, 1.0),
                            activation='relu')(inp)
            outp = Dense(1, activation='sigmoid')(h_layer)
            return Model(inp, outp)

        model = make_model()
        model.compile(loss='mse', metrics=['mae'])
        X = np.random.rand(200)  # pylint: disable=invalid-name
        y = np.vectorize(test_fn)(X)
        X_train, X_test, y_train, y_test = train_test_split(  # pylint: disable=invalid-name
            X, y, test_size=0.20, random_state=42)

        model.train_function = scipy_optimizer.make_train_function(
            model, verbose=0, maxiter=50)
        hist = model.fit(X_train, y_train,
                         validation_data=(X_test, y_test), verbose=False)
        self.assertLessEqual(hist.history['loss'][-1], 0.2)
        self.assertTrue('val_loss' in hist.history)
        self.assertTrue('val_mean_absolute_error' in hist.history or
                        'val_mae' in hist.history)

    def test_mult_inputs(self):
        """ Multiple input model
        """
        def test_fn(x, y):
            return 2.0 * x + 4.0 * y + 1.0

        def make_model():
            x = Input(shape=(1, ))
            y = Input(shape=(1, ))
            join = Concatenate()([x, y])
            z = Dense(1)(join)
            return Model([x, y], z)

        model = make_model()
        model.compile(loss='mse')
        X = np.random.rand(10)  # pylint: disable=invalid-name
        Y = np.random.rand(10)  # pylint: disable=invalid-name
        Z = np.vectorize(test_fn)(X, Y)  # pylint: disable=invalid-name

        model.train_function = scipy_optimizer.make_train_function(
            model, verbose=0, maxiter=100)

        hist = model.fit([X, Y], Z, verbose=False)
        self.assertLess(hist.history['loss'][-1], 0.1)

    def test_non_trainable(self):
        """BatchNormalization uses non-trainable weights.
        """
        model = Sequential()
        model.add(Dense(3, use_bias=False, input_dim=4))
        model.add(BatchNormalization())
        model.add(Dense(1, use_bias=False))
        model.compile(loss='mse')

        def fn(vec):
            a, b, c, d = vec
            return a*b + 2*b + 3*c + d

        inputs = np.random.rand(10, 4)
        outputs = np.zeros(inputs.shape[0])
        for i in range(inputs.shape[0]):
            outputs[i] = fn(inputs[i, :])

        model.train_function = scipy_optimizer.make_train_function(
            model, verbose=0, maxiter=50)
        hist = model.fit(inputs, outputs, verbose=False)
        self.assertLessEqual(hist.history['loss'][-1], 1.0e3)


if __name__ == '__main__':
    unittest.main()
