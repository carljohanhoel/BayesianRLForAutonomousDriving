import unittest
import numpy as np

import sys
sys.path.append('../src')

from policy import EnsembleTestPolicy


class Tester(unittest.TestCase):

    def test_mean(self):
        policy = EnsembleTestPolicy(policy_type='mean')
        q_values_all_nets = np.random.rand(10, 8)   # 10 nets and 8 actions in this example
        action, policy_info = policy.select_action(q_values_all_nets)
        self.assertLess(action, 8)
        self.assertGreaterEqual(action, 0)

        q_values_all_nets = np.zeros([10, 8])
        q_values_all_nets[:, 3] = np.ones([10])
        action, policy_info = policy.select_action(q_values_all_nets)
        self.assertEqual(action, 3)

    def test_voting(self):
        policy = EnsembleTestPolicy(policy_type='voting')
        q_values_all_nets = np.random.rand(10, 8)
        action, policy_info = policy.select_action(q_values_all_nets)
        self.assertLess(action, 8)
        self.assertGreaterEqual(action, 0)

        q_values_all_nets = np.zeros([10, 8])
        q_values_all_nets[:, 3] = np.ones([10])
        action, policy_info = policy.select_action(q_values_all_nets)
        self.assertEqual(action, 3)

        q_values_all_nets = np.zeros([10, 8])
        q_values_all_nets[0:5, 3] = np.ones([5])
        q_values_all_nets[5:10, 4] = np.ones([5])
        actions = []
        for i in range(100):
            actions.append(policy.select_action(q_values_all_nets)[0])
        self.assertFalse((actions[0] == np.array(actions)).all())

    def test_safe_policy(self):
        safety_threshold = 0.1
        safe_action = 4

        policy = EnsembleTestPolicy(safety_threshold=safety_threshold, safe_action=safe_action)
        q_values_all_nets = np.ones([10, 8])
        q_values_all_nets[:, 4] *= 1.001
        action, policy_info = policy.select_action(q_values_all_nets)
        self.assertEqual(action, 4)
        self.assertFalse(policy_info['safe_action'])

        q_values_all_nets = np.ones([10, 8])
        q_values_all_nets[0, 0:2] *= 100
        q_values_all_nets[:, 6] *= 1.001
        action, policy_info = policy.select_action(q_values_all_nets)
        self.assertEqual(action, 6)
        self.assertTrue(policy_info['safe_action'])

        q_values_all_nets = np.ones([10, 8])
        q_values_all_nets[0, :] *= 100
        action, policy_info = policy.select_action(q_values_all_nets)
        self.assertEqual(action, safe_action)
        self.assertTrue(policy_info['safe_action'])


if __name__ == '__main__':
    unittest.main()
