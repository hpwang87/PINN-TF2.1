"""
The function is taken from deepxde
which can be downloaded from https://github.com/lululxvi/deepxde
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

_BACKEND = "tensorflow"
_VERSION = tf.__version__
_IS_TF_1 = _VERSION.startswith("1.")
_GPU_NUM = 1
_CPU_NUM = 1


if _IS_TF_1:
    print("Using TensorFlow V1 backend.")
    # tf = tf
else:
    print("Using TensorFlow V2 backend.")
    # tf = tf.compat.v1
    # tf.disable_v2_behavior()
    

gpus = tf.config.list_physical_devices("GPU")
_GPU_NUM = len(gpus)
cpus = tf.config.list_physical_devices("CPU")
_CPU_NUM = len(cpus)
if gpus:
    """multi gpus
    """
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        # 和MirroredStrategy冲突
        #tf.config.set_visible_devices(gpu,"GPU") 
        print('Device info:')
        print(gpu)
else:
    """multi cpus
    """
    for cpu in cpus:
        tf.config.experimental.set_memory_growth(cpu, True)
        # 可能和MirroredStrategy冲突
        #tf.config.set_visible_devices(cpu,"CPU") 
        print('Device info:\n')
        print(cpu)

def backend():
    """Returns the name and version of the current backend, e.g., ("tensorflow", 1.14.0).

    Returns:
        tuple: A ``tuple`` of the name and version of the backend DeepXDE is currently using.

    Examples:

    .. code-block:: python

        >>> dde.backend.backend()
        ("tensorflow", 1.14.0)
    """
    return _BACKEND, _VERSION


def is_tf_1():
    return _IS_TF_1
