import unittest
import evspikesim as sim
import numpy as np


class TestUniformInitializer(unittest.TestCase):
    def test_layer_initialization(self):
        network = sim.SpikingNetwork()
        init = sim.initializers.UniformInitializer()

        layer = network.add_fc_layer(10, 10, 0.020, 0.1, init)

        self.assertEqual(layer.weights.size, np.unique(layer.weights).size)
        self.assertTrue((layer.weights >= -1.0).all())
        self.assertTrue((layer.weights <= 1.0).all())


    def test_layer_initialization_bounds(self):
        network = sim.SpikingNetwork()
        init = sim.initializers.UniformInitializer(0, 42.21)

        layer = network.add_fc_layer(10, 10, 0.020, 0.1, init)

        self.assertEqual(layer.weights.size, np.unique(layer.weights).size)
        self.assertTrue((layer.weights >= 0.0).all())
        self.assertTrue((layer.weights <= 42.21).all())
