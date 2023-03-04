import unittest
import evspikesim as sim
from evspikesim.initializers import ConstantInitializer
import numpy as np


class TestSpikingNetwork(unittest.TestCase):
    def test_add_fc_layers(self):
        net = sim.SpikingNetwork()

        self.assertEqual(len(net), 0)

        net.add_fc_layer(2, 3, 0.020, 0.1, ConstantInitializer())

        self.assertEqual(len(net), 1)

        net.add_fc_layer(2, 3, 0.020, 0.1, ConstantInitializer())
        net.add_fc_layer(2, 3, 0.020, 0.1, ConstantInitializer())

        self.assertEqual(len(net), 3)

    def test_layer_iterator(self):
        params = [[2, 3, 0.1, 1.1],
                  [3, 4, 0.2, 2.2],
                  [4, 5, 0.3, 3.3]]

        net = sim.SpikingNetwork()

        for n_inputs, n_neurons, tau_s, threshold in params:
            net.add_fc_layer(n_inputs, n_neurons, tau_s, threshold, ConstantInitializer())

        for (n_inputs, n_neurons, tau_s, threshold), layer in zip(params, net):
            self.assertEqual(layer.n_inputs, n_inputs)
            self.assertEqual(layer.n_neurons, n_neurons)
            self.assertAlmostEqual(layer.tau_s, tau_s)
            self.assertAlmostEqual(layer.tau, 2 * tau_s)
            self.assertAlmostEqual(layer.threshold, threshold)

    def test_get_output_layer(self):
        params = [[2, 3, 0.1, 1.1],
                  [3, 4, 0.2, 2.2],
                  [4, 5, 0.3, 3.3]]

        net = sim.SpikingNetwork()

        for n_inputs, n_neurons, tau_s, threshold in params:
            net.add_fc_layer(n_inputs, n_neurons, tau_s, threshold, ConstantInitializer())

        n_inputs, n_neurons, tau_s, threshold = params[2]
        layer = net.output_layer
        self.assertEqual(layer.n_inputs, n_inputs)
        self.assertEqual(layer.n_neurons, n_neurons)
        self.assertAlmostEqual(layer.tau_s, tau_s)
        self.assertAlmostEqual(layer.tau, 2 * tau_s)
        self.assertAlmostEqual(layer.threshold, threshold)

    def test_get_item(self):
        params = [[2, 3, 0.1, 1.1],
                  [3, 4, 0.2, 2.2],
                  [4, 5, 0.3, 3.3]]

        net = sim.SpikingNetwork()

        for n_inputs, n_neurons, tau_s, threshold in params:
            net.add_fc_layer(n_inputs, n_neurons, tau_s, threshold, ConstantInitializer())

        for (n_inputs, n_neurons, tau_s, threshold), idx in zip(params, range(len(net))):
            layer = net[idx]
            self.assertEqual(layer.n_inputs, n_inputs)
            self.assertEqual(layer.n_neurons, n_neurons)
            self.assertAlmostEqual(layer.tau_s, tau_s)
            self.assertAlmostEqual(layer.tau, 2 * tau_s)
            self.assertAlmostEqual(layer.threshold, threshold)

    def test_infer_spike_array(self):
        weights = np.array([[1.0, 0.2],
                            [-0.1, 0.8],
                            [0.5, 0.4]], dtype=np.float32)

        indices = np.array([0, 1, 1], dtype=np.uint32)
        times = np.array([1.0, 1.5, 1.2], dtype=np.float32)
        inputs = sim.SpikeArray(indices, times)
        inputs.sort()

        targets_indices = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=np.uint32)
        targets_times = np.array([1.0047829, 1.0112512, 1.0215546, 1.2063813, 1.2163547, 1.506313, 1.5162327,
                                  1.0129402, 1.2233235, 1.5267321], dtype=np.float32)
        targets = sim.SpikeArray(targets_indices, targets_times)
        targets.sort()

        net = sim.SpikingNetwork()
        layer = net.add_fc_layer(2, 3, 0.020, 0.1, ConstantInitializer())
        layer.weights = weights

        output_spikes = net.infer(inputs)
        self.assertEqual(output_spikes, targets)

    def test_infer_spike_array_unsorted_exception(self):
        weights = np.array([[1.0, 0.2],
                            [-0.1, 0.8],
                            [0.5, 0.4]], dtype=np.float32)

        indices = np.array([0, 1, 1], dtype=np.uint32)
        times = np.array([1.0, 1.5, 1.2], dtype=np.float32)
        inputs = sim.SpikeArray(indices, times)

        net = sim.SpikingNetwork()
        layer = net.add_fc_layer(2, 3, 0.020, 0.1, ConstantInitializer())
        layer.weights = weights

        self.assertRaises(RuntimeError, net.infer, inputs)


    def test_infer_no_spike_array(self):
        weights = np.array([[1.0, 0.2],
                            [-0.1, 0.8],
                            [0.5, 0.4]], dtype=np.float32)

        input_indices = np.array([0, 1, 1], dtype=np.uint32)
        input_times = np.array([1.0, 1.5, 1.2], dtype=np.float32)

        targets_indices = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=np.uint32)
        targets_times = np.array([1.0047829, 1.0112512, 1.0215546, 1.2063813, 1.2163547, 1.506313, 1.5162327,
                                  1.0129402, 1.2233235, 1.5267321], dtype=np.float32)
        targets = sim.SpikeArray(targets_indices, targets_times)
        targets.sort()

        net = sim.SpikingNetwork()
        layer = net.add_fc_layer(2, 3, 0.020, 0.1, ConstantInitializer())
        layer.weights = weights

        output_spikes = net.infer(indices=input_indices, times=input_times)
        self.assertEqual(output_spikes, targets)

    def test_infer_reset(self):
        weights = np.array([[1.0, 0.2],
                            [-0.1, 0.8],
                            [0.5, 0.4]], dtype=np.float32)

        indices = np.array([0, 1, 1], dtype=np.uint32)
        times = np.array([1.0, 1.5, 1.2], dtype=np.float32)
        inputs = sim.SpikeArray(indices, times)
        inputs.sort()

        targets_indices = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=np.uint32)
        targets_times = np.array([1.0047829, 1.0112512, 1.0215546, 1.2063813, 1.2163547, 1.506313, 1.5162327,
                                  1.0129402, 1.2233235, 1.5267321], dtype=np.float32)
        targets = sim.SpikeArray(targets_indices, targets_times)
        targets.sort()

        net = sim.SpikingNetwork()
        layer = net.add_fc_layer(2, 3, 0.020, 0.1, ConstantInitializer())
        layer.weights = weights

        net.infer(inputs) # Run a first time
        output_spikes = net.infer(inputs)
        self.assertEqual(output_spikes, targets)