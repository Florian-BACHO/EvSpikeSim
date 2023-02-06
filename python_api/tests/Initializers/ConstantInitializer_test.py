import unittest
import evspikesim as sim


class TestConstantInitializer(unittest.TestCase):
    def test_layer_initialization(self):
        desc = sim.layers.FCLayerDescriptor(42, 21, 0.1, 1.0)
        network = sim.SpikingNetwork()
        init = sim.initializers.ConstantInitializer(42.21)

        layer = network.add_layer(desc, init)

        self.assertTrue((layer.weights == 42.21).all())