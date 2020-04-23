import unittest
import numpy as np
import sys
sys.path.append('../src')
from memory import BootstrappingMemory


class Tester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)
        self.limit = 1000
        self.adding_prob = 0.5
        self.nb_nets = 10
        self.memory = BootstrappingMemory(self.nb_nets, self.limit, adding_prob=self.adding_prob, window_length=1)

    def test_init(self):
        config = self.memory.get_config()
        self.assertEqual(config['limit'], self.limit)
        self.assertTrue(self.memory.nb_entries == 0)

    def append(self, nb_samples):
        for i in range(nb_samples):
            observation, action, reward, terminal = np.random.rand(), np.random.rand(), np.random.rand(), \
                                                    np.random.rand() < 0.1
            self.memory.append(observation, action, reward, terminal)

    def test_append(self):
        self.memory = BootstrappingMemory(self.nb_nets, self.limit, adding_prob=self.adding_prob, window_length=1)
        nb_samples = int(self.limit/2)
        self.append(nb_samples)
        self.assertEqual(self.memory.nb_entries, nb_samples)
        # This test should fail with a low probability
        for i in range(self.nb_nets):
            self.assertAlmostEqual(nb_samples*self.adding_prob/(nb_samples/10),
                                   len(self.memory.index_refs[i])/(nb_samples/10), 0)

    def test_append_full_memory(self):
        self.memory = BootstrappingMemory(self.nb_nets, self.limit, adding_prob=self.adding_prob, window_length=1)
        nb_samples = int(self.limit*9.5)
        self.append(nb_samples)
        self.assertEqual(self.memory.nb_entries, self.limit)
        for i in range(self.nb_nets):
            self.assertAlmostEqual(self.limit * self.adding_prob / (self.limit / 10),
                                   len(self.memory.index_refs[i]) / (self.limit / 10), 0)

    def test_get_recent_state(self):
        state_in = 5
        state_out = self.memory.get_recent_state(state_in)
        self.assertEqual([state_in], state_out)

    def test_sample(self):
        nb_samples = int(self.limit*1.5)
        self.append(nb_samples)
        sample = self.memory.sample(net=5, batch_size=32)
        self.assertEqual(len(sample), 32)

    def test_window_length(self):
        window_length = 5
        self.memory = BootstrappingMemory(self.nb_nets, self.limit, adding_prob=self.adding_prob,
                                          window_length=window_length)
        nb_samples = int(self.limit * 1.5)
        self.append(nb_samples)
        sample = self.memory.sample(net=5, batch_size=32)
        self.assertEqual(len(sample), 32)


if __name__ == '__main__':
    unittest.main()
