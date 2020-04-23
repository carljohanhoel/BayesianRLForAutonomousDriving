from rl.policy import Policy
import numpy as np


class EnsembleTestPolicy(Policy):
    """
    Policy used by the ensemble method during testing episodes.

    During testing episodes, the policy chooses the action with either the highest mean Q-value or the actions that has
    the highest Q-value in most of the ensemble members.
    If safety_threshold is set, only actions with a coefficient of variation below the set value are considered.
    If no action is considered safe, the fallback action safe_action is used.

    Args:
        policy_type (str): 'mean' or 'voting'
        safety_threshold (float): Maximum coefficient of variation that is considered safe.
        safe_action (int): Fallback action if all actions are considered unsafe.
    """

    def __init__(self, policy_type='mean', safety_threshold=None, safe_action=None):
        self.policy_type = policy_type
        self.safety_threshold = safety_threshold
        self.safe_action = safe_action
        if self.safety_threshold is not None:
            assert(safe_action is not None)

    def select_action(self, q_values_all_nets):
        """
        Selects action by highest mean or voting, possibly subject to safety threshold.

        Args:
            q_values_all_nets (ndarray): Array with Q-values of all the actions for all the ensemble members.

        Returns:
            tuple: containing:
                int: chosen action
                dict: if the safety threshold is active: info if the fallback action was used or not,
                      otherwise: empty
        """
        if self.policy_type == 'mean':
            mean_q_values = np.mean(q_values_all_nets, axis=0)
            if self.safety_threshold is None:
                return np.argmax(mean_q_values), {}
            else:
                std_q_values = np.std(q_values_all_nets, axis=0)
                coef_of_var = std_q_values / np.abs(mean_q_values)
                sorted_q_indexes = mean_q_values.argsort()[::-1]
                i = 0
                while i < len(coef_of_var) and coef_of_var[sorted_q_indexes[i]] > self.safety_threshold:
                    i += 1
                if i == len(coef_of_var):  # No action is considered safe - use fallback action
                    return self.safe_action, {'safe_action': True}
                else:
                    return sorted_q_indexes[i], {'safe_action': not i == 0}
        elif self.policy_type == 'voting':
            action_votes = np.argmax(q_values_all_nets, axis=1)
            actions, counts = np.unique(action_votes, return_counts=True)
            max_actions = np.flatnonzero(counts == max(counts))
            action = actions[np.random.choice(max_actions)]
            if self.safety_threshold is None:
                return action, {}
            else:
                raise Exception('Voting policy for safe actions is not yet implemented.')
        else:
            raise Exception('Unvalid policy type defined.')

    def get_config(self):
        config = super(EnsembleTestPolicy, self).get_config()
        config['type'] = self.policy_type
        return config
