import unittest
import evspikesim as sim


class TestSpike(unittest.TestCase):
    def test_values(self):
        s = sim.Spike(1, 21.42)

        self.assertEqual(s.index, 1)
        self.assertAlmostEqual(s.time, 21.42, 6)


    def test_comparators(self):
        spike_1 = sim.Spike(42, 21.42)
        spike_2 = sim.Spike(1, 10.84)
        spike_3 = sim.Spike(1, 21.42)
    
        # >
        self.assertTrue(spike_1 > spike_2)
        self.assertFalse(spike_2 > spike_1)
        self.assertFalse(spike_1 > spike_3)
        self.assertFalse(spike_3 > spike_1)
    
        # >=
        self.assertTrue(spike_1 >= spike_2)
        self.assertFalse(spike_2 >= spike_1)
        self.assertTrue(spike_1 >= spike_3)
        self.assertTrue(spike_3 >= spike_1)
    
        # <
        self.assertFalse(spike_1 < spike_2)
        self.assertTrue(spike_2 < spike_1)
        self.assertFalse(spike_1 < spike_3)
        self.assertFalse(spike_3 < spike_1)
    
        # <=
        self.assertFalse(spike_1 <= spike_2)
        self.assertTrue(spike_2 <= spike_1)
        self.assertTrue(spike_1 <= spike_3)
        self.assertTrue(spike_3 <= spike_1)
    
        # ==
        self.assertFalse(spike_1 == spike_2)
        self.assertFalse(spike_2 == spike_1)
        self.assertTrue(spike_1 == spike_3)
        self.assertTrue(spike_3 == spike_1)
    
        # !=
        self.assertTrue(spike_1 != spike_2)
        self.assertTrue(spike_2 != spike_1)
        self.assertFalse(spike_1 != spike_3)
        self.assertFalse(spike_3 != spike_1)

    def test_str(self):
        spike = sim.Spike(42, 21.42)

        self.assertEqual(str(spike), "Index: 42, Time: 21.42")