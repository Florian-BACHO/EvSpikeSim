import unittest
import evspikesim as sim


class TestLayerDescriptor(unittest.TestCase):
    def test_values(self):
        desc = sim.layers.LayerDescriptor(42, 21, 0.1, 1.0)

        self.assertEqual(desc.n_inputs, 42)
        self.assertEqual(desc.n_neurons, 21)
        self.assertAlmostEqual(desc.tau_s, 0.1, 6)
        self.assertAlmostEqual(desc.tau, 0.2, 6)
        self.assertAlmostEqual(desc.threshold, 1.0, 6)
