import unittest
import evspikesim as sim
import numpy as np


class TestUniformInitializer(unittest.TestCase):
    def test_layer_initialization(self):
        desc = sim.layers.FCLayerDescriptor(42, 21, 0.1, 1.0)
        network = sim.SpikingNetwork()
        init = sim.initializers.UniformInitializer()

        layer = network.add_layer(desc, init)

        self.assertEqual(layer.weights.size, np.unique(layer.weights).size)
        self.assertTrue((layer.weights >= -1.0).all())
        self.assertTrue((layer.weights <= 1.0).all())


    def test_layer_initialization_bounds(self):
        desc = sim.layers.FCLayerDescriptor(42, 21, 0.1, 1.0)
        network = sim.SpikingNetwork()
        init = sim.initializers.UniformInitializer(0, 42.21)

        layer = network.add_layer(desc, init)

        self.assertEqual(layer.weights.size, np.unique(layer.weights).size)
        self.assertTrue((layer.weights >= 0.0).all())
        self.assertTrue((layer.weights <= 42.21).all())
