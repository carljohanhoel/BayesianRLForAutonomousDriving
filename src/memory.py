"""
Based on the original memory.py from keras-rl (https://github.com/keras-rl/keras-rl).
Extended to handle an ensemble of agents with separate neural networks, which are trained on separate replay memories.
"""

from rl.memory import Memory, RingBuffer, zeroed_observation
import numpy as np
import warnings
from collections import namedtuple

# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')


class BootstrappingMemory(Memory):
    """
    Replay memory for ensemble, with separate memories for different ensemble members.

    The replay memory consists of a circular buffer that stores experiences.
    To limit memory usage, only one replay memory is maintained, and the separate replay memories of each
    ensemble member are created by referencing different subsets of the joint replay memory.

    Args:
        nb_nets (int): Number of ensemble members.
        limit(int): Replay memory size.
        adding_prob (float): Probability of adding an experience to the replay memory of each ensemble member.
        **kwargs: contains:
            window_length (int): Number of states in the network input.
    """
    def __init__(self, nb_nets, limit, adding_prob=0.5, **kwargs):
        super(BootstrappingMemory, self).__init__(**kwargs)
        self.nb_nets = nb_nets
        self.adding_prob = adding_prob
        self.limit = limit
        self.index_refs = [[] for i in range(self.nb_nets)]

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

    def sample(self, net, batch_size):
        """
        Returns a randomized batch of experiences for an ensemble member
        Args:
            net (int): Index of ensemble member
            batch_size (int): Size of the batch

        Returns:
            A list of random experiences
        """
        # It is not possible to tell whether the first state in the memory is terminal, because it
        # would require access to the "terminal" flag associated to the previous state. As a result
        # we will never return this first state (only using `self.terminals[0]` to know whether the
        # second state is terminal).
        # In addition we need enough entries to fill the desired window length.
        assert self.nb_entries >= self.window_length + 2, 'not enough entries in the memory'

        # Sample random indexes for the specified ensemble member
        batch_idxs = self.sample_batch_idxs(net, batch_size)

        assert np.min(batch_idxs) >= self.window_length + 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2]
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                # idx = sample_batch_indexes(self.window_length + 1, self.nb_entries, size=1)[0]
                idx = self.sample_batch_idxs(net, 1)[0]
                terminal0 = self.terminals[idx - 2]
            assert self.window_length + 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                assert current_idx >= 1
                current_terminal = self.terminals[current_idx - 1]
                if current_terminal and not self.ignore_episode_boundaries:
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size
        return experiences

    def sample_batch_idxs(self, net, batch_size):
        """
        Sample random replay memory indexes from the list of replay memory indexes of the specified ensemble member.

        Args:
            net (int): Index of ensemble member
            batch_size (int): Size of the batch

        Returns:
            A list of replay memory indexes.
        """
        memory_size = len(self.index_refs[net])
        assert memory_size > self.window_length + 1
        if batch_size > memory_size:
            warnings.warn("Less samples in memory than batch size.")
        ref_idxs = np.random.randint(0, memory_size, batch_size)
        batch_idxs = [self.index_refs[net][idx] for idx in ref_idxs]
        assert len(batch_idxs) == batch_size
        return batch_idxs

    def append(self, observation, action, reward, terminal, training=True):
        """Append an observation to the replay memory.

        With probability self.adding_prob, the index of the added experience will be added to the index list of each
        ensemble replay memory.

        Args:
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
            training (boolean): True during training episodes, false during testing episodes.
        """
        super(BootstrappingMemory, self).append(observation, action, reward, terminal, training=training)

        if training:
            if self.nb_entries < self.limit:   # One more entry will be added after this loop
                # There should be enough experiences before the chosen sample to fill the window length + 1
                if self.nb_entries > self.window_length + 1:
                    for i in range(self.nb_nets):
                        if np.random.rand() < self.adding_prob:
                            self.index_refs[i].append(self.nb_entries)

        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    @property
    def nb_entries(self):
        """Return number of observations

        Returns:
            Number of observations
        """
        return len(self.observations)

    def get_config(self):
        """Return configurations of SequentialMemory

        Returns:
            Dict of config
        """
        config = super(BootstrappingMemory, self).get_config()
        config['limit'] = self.limit
        return config
