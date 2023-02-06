import unittest
import evspikesim as sim
import numpy as np


class TestNormalInitializer(unittest.TestCase):
    def test_layer_initialization(self):
        desc = sim.layers.FCLayerDescriptor(42, 21, 0.1, 1.0)
        network = sim.SpikingNetwork()
        init = sim.initializers.NormalInitializer()

        layer = network.add_layer(desc, init)

        self.assertEqual(layer.weights.size, np.unique(layer.weights).size)
