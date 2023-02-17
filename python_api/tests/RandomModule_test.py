import unittest
import evspikesim as sim
import numpy as np


class TestRandomModule(unittest.TestCase):
    def get_weights(self):
        network = sim.SpikingNetwork()
        init = sim.initializers.UniformInitializer()

        return network.add_fc_layer(10, 10, 0.020, 0.1, init).weights

    def test_seed(self):
        sim.random.set_seed(42)

        weights = self.get_weights()
        weights_2 = self.get_weights()

        sim.random.set_seed(42)

        weights_3 = self.get_weights()

        self.assertFalse(np.allclose(weights, weights_2))
        self.assertTrue(np.allclose(weights, weights_3))