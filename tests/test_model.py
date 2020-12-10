import unittest
from mixture_net.utils import nnelu, register_custom_activation
from mixture_net.model import MDN
import numpy as np
from sklearn.model_selection import train_test_split
from mixture_net.losses import gnll_loss


class TestModel(unittest.TestCase):
    def setUp(self):
        samples = int(100)

        x_data = np.random.sample(samples)[:, np.newaxis].astype(np.float32)
        y_data = np.add(5 * x_data, np.multiply((x_data) ** 2, np.random.standard_normal(x_data.shape)))

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_data, y_data,
                                                                                test_size=0.5, random_state=42)

    def test_param_instantiation(self):
        register_custom_activation('nnelu', nnelu)
        tiny_net = MDN(neurons=2, components=3)
        self.assertEqual(tiny_net.neurons, 2)
        self.assertEqual(tiny_net.components, 3)

    def test_model_build(self):
        register_custom_activation('nnelu', nnelu)
        net = MDN(neurons=2, components=1)
        net.compile(loss=gnll_loss(3, 1), optimizer='adam')
        net.fit(x=self.x_train, y=self.y_train, epochs=10, batch_size=128)
        self.assertTrue(net.built)
