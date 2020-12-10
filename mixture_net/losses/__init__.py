from tensorflow_probability import distributions as tfd
import tensorflow as tf
from mixture_net.utils import slice_parameter_vectors


def gnll_loss(no_parameters, components):
    def custom_loss(y, parameter_vector):
        """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
        """
        alpha, mu, sigma = slice_parameter_vectors(parameter_vector, components, no_parameters)

        gm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=alpha),
            components_distribution=tfd.Normal(
                loc=mu,
                scale=sigma))

        log_likelihood = gm.log_prob(tf.transpose(y))

        return -tf.reduce_mean(log_likelihood, axis=-1)
    return custom_loss
