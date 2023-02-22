import unittest
import evspikesim as sim
import numpy as np


class TestNormalInitializer(unittest.TestCase):
    def test_layer_initialization(self):
        network = sim.SpikingNetwork()
        init = sim.initializers.NormalInitializer()

        layer = network.add_fc_layer(10, 10, 0.020, 0.1, init)

        self.assertEqual(layer.weights.size, np.unique(layer.weights).size)
