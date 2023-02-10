import unittest
import evspikesim as sim
from evspikesim.layers import FCLayerDescriptor
import numpy as np


class TestFCLayer(unittest.TestCase):
    def test_add_fc_layers(self):
        net = sim.SpikingNetwork()

        layer = net.add_layer(FCLayerDescriptor(2, 3, 0.1, 1.0))

        self.assertEqual(layer.descriptor.n_inputs, 2)
        self.assertEqual(layer.descriptor.n_neurons, 3)
        self.assertAlmostEqual(layer.descriptor.tau_s, 0.1)
        self.assertAlmostEqual(layer.descriptor.tau, 0.2)
        self.assertAlmostEqual(layer.descriptor.threshold, 1.0)

    def test_get_set_weights(self):
        weights = np.array([[1.0, 0.2],
                            [-0.1, 0.8],
                            [0.5, 0.4]], dtype=np.float32)

        net = sim.SpikingNetwork()
        layer = net.add_layer(FCLayerDescriptor(2, 3, 0.020, 0.020 * 0.2))

        self.assertTrue((layer.weights != weights).all())
        self.assertTrue((layer.weights == 0.0).all())

        layer.weights = weights

        self.assertTrue((layer.weights != 0.0).all())
        self.assertTrue((layer.weights == weights).all())

    def test_weights_mutability(self):
        weights = np.array([[0.0, 0.0],
                            [0.0, 0.8],
                            [0.5, 0.0]], dtype=np.float32)

        net = sim.SpikingNetwork()
        layer = net.add_layer(FCLayerDescriptor(2, 3, 0.020, 0.020 * 0.2))

        self.assertFalse((layer.weights != 0.0).any())
        self.assertFalse((layer.weights == weights).all())

        layer.weights[1, 1] = 0.8
        layer.weights[2, 0] = 0.5

        self.assertTrue((layer.weights != 0.0).any())
        self.assertTrue((layer.weights == weights).all())

        layer.weights *= 2.42

        self.assertTrue((layer.weights == (weights * 2.42)).all())

    def test_inference(self):
        weights = np.array([[1.0, 0.2],
                            [-0.1, 0.8],
                            [0.5, 0.4]], dtype=np.float32)

        indices = np.array([0, 1, 1], dtype=np.uint32)
        times = np.array([1.0, 1.5, 1.2], dtype=np.float32)
        inputs = sim.SpikeArray(indices, times)
        inputs.sort()


        targets_n_spikes = np.array([3, 4, 3])
        targets_indices = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=np.uint32)
        targets_times = np.array([1.0047829, 1.0112512, 1.0215546, 1.2063813, 1.2163547, 1.506313, 1.5162327,
                                  1.0129402, 1.2233235, 1.5267321], dtype=np.float32)
        targets = sim.SpikeArray(targets_indices, targets_times)
        targets.sort()

        net = sim.SpikingNetwork()
        layer = net.add_layer(FCLayerDescriptor(2, 3, 0.020, 0.020 * 0.2))
        layer.weights = weights

        net.infer(inputs)
        output_spikes = layer.post_spikes
        output_n_spikes = layer.n_spikes

        self.assertTrue(isinstance(output_spikes, sim.SpikeArray))
        self.assertEqual(output_spikes, targets)

        self.assertTrue(isinstance(output_n_spikes, np.ndarray))
        self.assertTrue((output_n_spikes == targets_n_spikes).all())

    def test_inference_undersized_buffer(self):
        weights = np.array([[1.0, 0.2],
                            [-0.1, 0.8],
                            [0.5, 0.4]], dtype=np.float32)

        indices = np.array([0, 1, 1], dtype=np.uint32)
        times = np.array([1.0, 1.5, 1.2], dtype=np.float32)
        inputs = sim.SpikeArray(indices, times)
        inputs.sort()


        targets_n_spikes = np.array([3, 4, 3])
        targets_indices = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=np.uint32)
        targets_times = np.array([1.0047829, 1.0112512, 1.0215546, 1.2063813, 1.2163547, 1.506313, 1.5162327,
                                  1.0129402, 1.2233235, 1.5267321], dtype=np.float32)
        targets = sim.SpikeArray(targets_indices, targets_times)
        targets.sort()

        net = sim.SpikingNetwork()
        layer = net.add_layer(FCLayerDescriptor(2, 3, 0.020, 0.020 * 0.2), buffer_size=1)
        layer.weights = weights

        net.infer(inputs)
        output_spikes = layer.post_spikes
        output_n_spikes = layer.n_spikes

        self.assertTrue(isinstance(output_spikes, sim.SpikeArray))
        self.assertEqual(output_spikes, targets)

        self.assertTrue(isinstance(output_n_spikes, np.ndarray))
        self.assertTrue((output_n_spikes == targets_n_spikes).all())