import unittest
import evspikesim as sim
import numpy as np


class TestRandomModule(unittest.TestCase):
    def get_weights(self):
        desc = sim.layers.FCLayerDescriptor(2, 3, 0.1, 1.0)
        network = sim.SpikingNetwork()
        init = sim.initializers.UniformInitializer()

        return network.add_layer(desc, init).weights

    def test_seed(self):
        sim.random.set_seed(42)

        weights = self.get_weights()
        weights_2 = self.get_weights()

        sim.random.set_seed(42)

        weights_3 = self.get_weights()

        self.assertFalse(np.allclose(weights, weights_2))
        self.assertTrue(np.allclose(weights, weights_3))