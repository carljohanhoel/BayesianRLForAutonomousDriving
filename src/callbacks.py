import numpy as np
from rl.callbacks import Callback


class SaveWeights(Callback):
    """
    Callback to regularly save the weights of the neural network.

    The weights are only saved after an episode has ended, so not exactly at the specified saving frequency.

    Args:
        save_freq (int): Training steps between saves
        save_path (str): Path where the weights are saved.
    """
    def __init__(self, save_freq=10000, save_path=None):
        super(SaveWeights, self).__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.nb_saves = 0

    def on_episode_end(self, episode_step, logs=None):
        if (self.nb_saves == 0 or self.model.step - (self.nb_saves - 1) * self.save_freq >= self.save_freq) \
                and self.save_path is not None:
            print("Number of steps: ", self.model.step)
            self.model.save_weights(self.save_path + "/"+str(self.model.step))
            self.nb_saves += 1


class EvaluateAgent(Callback):
    """
    Callback to evaluate agent on testing episodes.

    Args:
        eval_freq (int): Training steps between evaluation runs.
        nb_eval_eps (int): Number of evaluation episodes.
        save_path (int): Path where the result is saved.
    """
    def __init__(self, eval_freq=10000, nb_eval_eps=5, save_path=None):
        super(EvaluateAgent, self).__init__()
        self.eval_freq = eval_freq
        self.nb_eval_eps = nb_eval_eps
        self.save_path = save_path
        self.nb_evaluation_runs = 0
        self.store_data_callback = StoreTestEpisodeData(save_path)
        self.env = None

    def on_episode_end(self, episode_step, logs=None):   # Necessary to run testing at the end of an episode
        if (self.nb_evaluation_runs == 0 or
            self.model.step - (self.nb_evaluation_runs-1) * self.eval_freq >= self.eval_freq) \
                and self.save_path is not None:
            test_result = self.model.test(self.env, nb_episodes=self.nb_eval_eps, callbacks=[self.store_data_callback],
                                          visualize=False)
            with open(self.save_path + '/test_rewards.csv', 'ab') as f:
                np.savetxt(f, test_result.history['episode_reward'], newline=' ')
                f.write(b'\n')
            with open(self.save_path + '/test_steps.csv', 'ab') as f:
                np.savetxt(f, test_result.history['nb_steps'], newline=' ')
                f.write(b'\n')
            self.model.training = True   # training is set to False in test function, so needs to be reset here
            self.nb_evaluation_runs += 1


class StoreTestEpisodeData(Callback):
    """
    Callback to log statistics on the test episodes.

    Args:
        save_path (int): Path where the result is saved.
    """
    def __init__(self, save_path=None):
        super(StoreTestEpisodeData, self).__init__()
        self.save_path = save_path
        self.episode = -1
        self.action_data = []
        self.reward_data = []
        self.q_values_data = None

    def on_step_end(self, episode_step, logs=None):
        assert(self.model.training is False)   # This should only be done in testing mode
        if logs is None:
            logs = {}

        if self.save_path is not None:
            if not logs['episode'] == self.episode:
                if not self.episode == -1:
                    with open(self.save_path + '/test_individual_reward_data.csv', 'ab') as f:
                        np.savetxt(f, self.reward_data, newline=' ')
                        f.write(b'\n')
                    with open(self.save_path + '/test_individual_action_data.csv', 'ab') as f:
                        np.savetxt(f, self.action_data, newline=' ')
                        f.write(b'\n')
                    if 'q_values_of_chosen_action' in logs:
                        with open(self.save_path + '/test_individual_qvalues_data.csv', 'ab') as f:
                            np.savetxt(f, self.q_values_data, newline='\n')
                            f.write(b'\n')
                self.episode = logs['episode']
                self.action_data = []
                self.reward_data = []
                self.action_data.append(logs['action'])
                self.reward_data.append(logs['reward'])
                if 'q_values_of_chosen_action' in logs:
                    self.q_values_data = []
                    self.q_values_data.append(logs['q_values_of_chosen_action'])
            else:
                self.action_data.append(logs['action'])
                self.reward_data.append(logs['reward'])
                if 'q_values_of_chosen_action' in logs:
                    self.q_values_data.append(logs['q_values_of_chosen_action'])
