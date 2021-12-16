# keras-opt

Keras Scipy optimize interface

Interface between keras optimizers and scipy.optimize. It is used to run full
batch optimization rather than mini-batch stochastic gradient descent. It
is applicable to factorization of very sparse matrices where stochastic
gradient descent is not able to converge.

Example usage:

```python
#%%
# Model definition (linear regression)
from keras_opt import scipy_optimizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

model = Sequential()
model.add(InputLayer(input_shape=(4,)))
model.add(Dense(1, use_bias=False))
model.compile(loss='mse')

#%%
# Generate test data

import numpy as np

np.random.seed(42)
X = np.random.uniform(size=40).reshape(10, 4)
y = np.dot(X, np.array([1, 2, 3, 4])[:, np.newaxis])

#%%
# Use scipy.optimize to minimize the cost
model.train_function = scipy_optimizer.make_train_function(
            model, maxiter=20)
history = model.fit(X, y)

#%%
# Show weights.
model.trainable_weights
```
