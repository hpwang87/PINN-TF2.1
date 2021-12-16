# -*- coding: utf-8 -*-
"""
Created on Jun14 2021

@author: H.P. Wang
github:  https://github.com/hpwang87
"""

from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.layers import Layer, Activation, Add, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.layers import LeakyReLU, ReLU
import tensorflow as tf


def swish(x, beta=0.1):
    """
    activation swish
    """
    return x * K.sigmoid(10 * beta * x)



class Swish(Layer):
    """
    define the swish layer, beta is a trainable variable
    """
    def __init__(self, beta=0.1, trainable=False, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta
        self.trainable = trainable

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.beta_factor = self.add_weight(name='beta_factor', 
                                      shape=(1, 1),
                                      initializer= initializers.Constant(self.beta),
                                      trainable=self.trainable)

        super(Swish, self).build(input_shape)

    def call(self, inputs, mask=None):
        return swish(inputs, self.beta_factor)

    def get_config(self):
        config = {'beta': self.get_weights()[0] if self.trainable else self.beta,
                  'trainable': self.trainable}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
    
    
    
    


def batchnorm_activation(x):
    """
    return batchnorm and  activation
    batch normalize the data before activation
    """
    # x = BatchNormalization()(x)
    # return Activation(swish)(x)
    return Swish(beta=0.1,trainable=True)(x)
    # return Activation(K.tanh)(x)
    # return Activation(K.sigmoid)(x)
    # return LeakyReLU()(x)
    # return ReLU()(x)







def res_block(input_tensor, input_units, gamma=0.01, scale=0.2):
    x = Dense(units=input_units, activation=None, 
              kernel_regularizer=None)(input_tensor)
    x = batchnorm_activation(x)

    x = Dense(units=input_units, activation=None,
              kernel_regularizer=None)(x)
    
    if scale:
        """
        lambda匿名函数的格式：冒号前是参数，可以有多个，用逗号隔开，冒号右边的为表达式。
        其实lambda返回值是一个函数的地址，也就是函数对象。
        
        Lambda:仅仅对数据进行变换，不学习
        x = x*scale
        """
        x = Lambda(lambda t: t * scale)(x)
    # equivalent to `added = tf.keras.layers.add([x1, x2])`
    x = Add()([x, input_tensor])
    x = batchnorm_activation(x)

    return x



def generator(layers, norm_paras, map_name='rnn'):
    # regularizer
    gamma = 0.01
    # input data
    # # Now the model will take as input arrays of shape (None, layers[0])
    inputs = Input(shape=(layers[0],))
    # normalized to [-1 1]
    lb = norm_paras[0, 0:layers[0]]
    ub = norm_paras[1, 0:layers[0]]
    input_scale = 2.0*(inputs - lb)/(ub - lb) - 1.0
    
    if map_name.lower() == 'fnn':
        # Densely Connected Networks 
        x = input_scale
        for width in layers[1:-1]:
            x = Dense(units=width, activation=None,
                      kernel_regularizer=None)(x)
            x = batchnorm_activation(x)
        width = layers[-1]     
        x = Dense(units=width, activation=None,
                  kernel_regularizer=None)(x) 
        # return the model
        return Model(inputs=inputs, outputs=x)
        
    elif map_name.lower() == 'rnn':
        num_layers = len(layers)
        block_size = int((num_layers-1-2)/2)
        # Residual Densely Connected Networks  
        width = layers[1]     
        x = Dense(units=width, activation=None,
                  kernel_regularizer=None)(input_scale)
        x = batchnorm_activation(x)
        
        for b in range(0,block_size):
            width = layers[2*b+2] 
            x = res_block(x, width, gamma=gamma, scale=1.0)
        width = layers[-1]     
        x = Dense(units=width, activation=None,
                  kernel_regularizer=None)(x)  
        # return the model
        return Model(inputs=inputs, outputs=x)        