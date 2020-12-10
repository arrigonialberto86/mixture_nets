import unittest
from mixture_net.utils import slice_parameter_vectors
import numpy as np


class TestUtils(unittest.TestCase):
    def test_slice_fn(self):
        parameters = np.array([[1, 2, 3]])
        components, no_parameters = 1, 3
        alpha, mu, gamma = slice_parameter_vectors(parameters, components, no_parameters)
        self.assertEqual(alpha.shape[0], 1)
