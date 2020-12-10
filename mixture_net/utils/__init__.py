import tensorflow as tf
from tensorflow.keras.layers import Activation


def nnelu(input):
    """ Computes the Non-Negative Exponential Linear Unit
    """
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))


def slice_parameter_vectors(parameter_vector, components, no_parameters):
    """ Returns an unpacked list of paramter vectors.
    """
    return [parameter_vector[:, i*components:(i+1)*components] for i in range(no_parameters)]


def register_custom_activation(activation_name, fn):
    """
    Register a custom activation function with Keras APIs (e.g. nnelu)
    """
    tf.keras.utils.get_custom_objects().update({activation_name: Activation(fn)})
    return True
