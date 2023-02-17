import unittest
import evspikesim as sim


class TestConstantInitializer(unittest.TestCase):
    def test_layer_initialization(self):
        network = sim.SpikingNetwork()
        init = sim.initializers.ConstantInitializer(42.21)

        layer = network.add_fc_layer(10, 10, 0.020, 0.1, init)

        self.assertTrue((layer.weights == 42.21).all())