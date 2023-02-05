import unittest
import evspikesim as sim
import numpy as np


class TestSpikeArray(unittest.TestCase):
    def test_constructor_numpy(self):
        indices = np.array([21, 12, 12], dtype=np.uint32)
        times = np.array([42.42, 1.42, 21.84], dtype=np.float32)

        arr = sim.SpikeArray(indices=indices, times=times)

        for spike, true_idx, true_time in zip(arr, indices, times):
            self.assertEqual(spike.index, true_idx)
            self.assertAlmostEqual(spike.time, true_time, delta=1e-5)

    def test_constructor_lists(self):
        indices = [21, 12, 12]
        times = [42.42, 1.42, 21.84]

        arr = sim.SpikeArray(indices=indices, times=times)

        for spike, true_idx, true_time in zip(arr, indices, times):
            self.assertEqual(spike.index, true_idx)
            self.assertAlmostEqual(spike.time, true_time, delta=1e-5)

    def test_len(self):
        arr = sim.SpikeArray()

        self.assertEqual(len(arr), 0)

        arr.add(21, 42.42)
        arr.add(12, 1.42)
        arr.add(12, 21.84)

        self.assertEqual(len(arr), 3)

    def test_add(self):
        arr = sim.SpikeArray()

        spikes = [(21, 42.42), (12, 1.42), (12, 21.84)]

        for idx, time in spikes:
            arr.add(idx, time)

        for spike, (true_idx, true_time) in zip(arr, spikes):
            self.assertEqual(spike.index, true_idx)
            self.assertAlmostEqual(spike.time, true_time, delta=1e-5)

    def test_add_numpy(self):
        indices = np.array([21, 12, 12], dtype=np.uint32)
        times = np.array([42.42, 1.42, 21.84], dtype=np.float32)

        arr = sim.SpikeArray()
        arr.add(indices, times)

        for spike, true_idx, true_time in zip(arr, indices, times):
            self.assertEqual(spike.index, true_idx)
            self.assertAlmostEqual(spike.time, true_time, delta=1e-5)

    def test_add_lists(self):
        indices = [21, 12, 12]
        times = [42.42, 1.42, 21.84]

        arr = sim.SpikeArray()
        arr.add(indices, times)

        for spike, true_idx, true_time in zip(arr, indices, times):
            self.assertEqual(spike.index, true_idx)
            self.assertAlmostEqual(spike.time, true_time, delta=1e-5)

    def test_sort(self):
        arr = sim.SpikeArray()

        spikes = [(21, 42.42), (12, 1.42), (12, 21.84)]

        for idx, time in spikes:
            arr.add(idx, time)
        arr.sort()

        spikes = sorted(spikes, key=lambda x: x[1])

        for spike, (true_idx, true_time) in zip(arr, spikes):
            self.assertEqual(spike.index, true_idx)
            self.assertAlmostEqual(spike.time, true_time, delta=1e-5)

    def test_clear(self):
        arr = sim.SpikeArray()

        self.assertEqual(len(arr), 0)

        it = arr.__iter__()
        self.assertRaises(StopIteration, it.__next__)

        arr.add(21, 42.42)
        arr.add(12, 1.42)
        arr.add(12, 21.84)

        arr.clear()

        self.assertEqual(len(arr), 0)

        it = arr.__iter__()
        self.assertRaises(StopIteration, it.__next__)

    def test_comparators(self):
        arr1 = sim.SpikeArray()
        arr2 = sim.SpikeArray()

        self.assertTrue(arr1 == arr2)
        self.assertFalse(arr1 != arr2)

        arr1.add(21, 42.42)
        arr1.add(12, 1.42)
        arr1.add(12, 21.84)
        arr1.sort()

        self.assertFalse(arr1 == arr2)
        self.assertTrue(arr1 != arr2)

        arr2.add(21, 42.42)

        self.assertFalse(arr1 == arr2)
        self.assertTrue(arr1 != arr2)

        arr2.add(12, 1.42)
        arr2.add(12, 21.84)

        self.assertFalse(arr1 == arr2)
        self.assertTrue(arr1 != arr2)

        arr2.sort()

        self.assertTrue(arr1 == arr2)
        self.assertFalse(arr1 != arr2)

    def test_str(self):
        arr = sim.SpikeArray()

        arr.add(21, 42.42)
        arr.add(12, 1.42)
        arr.add(12, 21.84)
        arr.sort()

        self.assertEqual(str(arr), "Index: 12, Time: 1.42\n"
                                   "Index: 12, Time: 21.84\n"
                                   "Index: 21, Time: 42.42\n")
