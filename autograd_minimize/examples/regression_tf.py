import numpy as np
from tensorflow import keras
from autograd_minimize.tf_wrapper import tf_function_factory
from autograd_minimize import minimize
import tensorflow as tf


 
tf.keras.backend.set_floatx('float32')   

class test:
    def __init__(self):
        # Prepares data
        self.X = np.random.random((500, 2))
        self.y = self.X[:, :1]*2+self.X[:, 1:]*0.4-1

        # Creates model
        self.model = keras.Sequential([keras.Input(shape=2),
                                  keras.layers.Dense(1)])

   
    def minimize(self):
        # Transforms model into a function of its parameter
        func, params, names = tf_function_factory(self.model, tf.keras.losses.MSE, self.X, self.y)
        # Minimization
        res = minimize(func,
                          params, 
                          method='L-BFGS-B', 
                          options={'disp':None,
                                   'maxiter': 100,
                                   'maxcor': 50,
                                   'maxls': 50,
                                   'gtol':1e-8,
                                   'eps':1e-8,
                                   'ftol': 1*np.finfo(float).eps},
                          callback= self.callback)
        return res

    
    def callback(self,x):
        # model 已经被更新了
        res =  np.mean(np.abs(self.model.predict(self.X)-self.y))
        print(f'callback: {res}')




mini = test();
mini.minimize();

print('Fitted parameters:')
print([var.numpy() for var in mini.model.trainable_variables])

print(f'mae: {np.mean(np.abs(mini.model(mini.X)-mini.y))}')
