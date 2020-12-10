import unittest
from mixture_net.losses import gnll_loss
import tensorflow as tf
import numpy as np


class TestLosses(unittest.TestCase):

    def test_negative_mix_likelihood(self):
        with tf.Session() as sess:
            evaluation_1 = gnll_loss(3, 1)(18.0, np.array([[1.0, 18.0, 1]]).astype(np.float32)).eval()
            evaluation_2 = gnll_loss(3, 1)(16.0, np.array([[1.0, 18.0, 1]]).astype(np.float32)).eval()
        self.assertTrue(evaluation_1 < evaluation_2)
