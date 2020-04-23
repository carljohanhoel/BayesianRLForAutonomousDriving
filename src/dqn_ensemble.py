"""
Based on the original dqn.py from keras-rl (https://github.com/keras-rl/keras-rl).
Extended to handle an ensemble of agents with separate neural networks, which are trained on separate replay memories.
"""

from __future__ import division
import warnings
from keras.layers import Lambda, Input
from keras.optimizers import Adam
from core import Agent
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl.util import *
from keras.callbacks import Callback
import multiprocessing as mp
from network_architecture import NetworkMLP, NetworkCNN


class AbstractDQNAgent(Agent):
    """
    Abstract DQN agent class, inheriting from Agent class in core.py.

    Args:
        nb_actions (int): Number of possible actions
        memory: Replay memory.
        gamma (float): MDP discount factor.
        batch_size (int): Batch size for stochastic gradient descent.
        nb_steps_warmup (int): Steps before training starts.
        train_interval (int): Steps between backpropagation calls.
        memory_interval (int): Steps between samples stored to memory.
        target_model_update (int): Steps between copying the paramters of the trained network to the target network.
        delta_clip (float): Huber loss parameter.
        custom_model_objects (dict): Not currently used.
        **kwargs:
    """

    def __init__(self, nb_actions, memory, gamma=.99, batch_size=32, nb_steps_warmup=1000,
                 train_interval=1, memory_interval=1, target_model_update=10000,
                 delta_clip=np.inf, custom_model_objects=None, **kwargs):
        super(AbstractDQNAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        # Parameters.
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.target_model_update = target_model_update
        self.delta_clip = delta_clip
        self.custom_model_objects = {} if custom_model_objects is None else custom_model_objects

        # Related objects.
        self.memory = memory

        # State.
        self.compiled = False

    def process_state_batch(self, batch):
        """ Heritage from keras-rl, not used here. """
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def compute_batch_q_values(self, state_batch, net):
        batch = self.process_state_batch(state_batch)
        if self.parallel:
            self.input_queues[net].put(['predict', batch])
            q_values = self.output_queues[net].get()
        else:
            q_values = self.models[net].predict_on_batch(batch)
        assert q_values.shape == (len(state_batch), self.nb_actions)
        return q_values

    def compute_q_values(self, state, net):
        """
        Compute Q-values for a particular state for a single ensemble member.

        Args:
            state (list): Input to the neural networks.
            net (int): Ensemble member index.

        Returns:
            q_values (list): Q-values for specified state and ensemble member
        """
        q_values = self.compute_batch_q_values([state], net).flatten()
        assert q_values.shape == (self.nb_actions,)
        return q_values

    def compute_q_values_all_nets(self, state):
        """
        Compute Q-values for a particular state, for all ensemble members.

        Args:
            state (list): Input to the neural networks.

        Returns:
            q_values_all_nets (ndarray): Matrix that describe the Q-value for all actions for all ensemble members.
        """
        q_values_all_nets = []
        if self.parallel:
            for net in range(self.nb_models):
                q_values_all_nets.append(self.compute_q_values(state, net))
        else:
            for net in range(len(self.models)):
                q_values_all_nets.append(self.compute_q_values(state, net))
        q_values_all_nets = np.array(q_values_all_nets)
        if self.parallel:
            assert q_values_all_nets.shape == (self.nb_models, self.nb_actions)
        else:
            assert q_values_all_nets.shape == (len(self.models), self.nb_actions)
        return q_values_all_nets

    def get_config(self):
        return {
            'nb_actions': self.nb_actions,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'nb_steps_warmup': self.nb_steps_warmup,
            'train_interval': self.train_interval,
            'memory_interval': self.memory_interval,
            'target_model_update': self.target_model_update,
            'delta_clip': self.delta_clip,
            'memory': get_object_config(self.memory),
        }


# An implementation of the DQN agent as described in Mnih (2013) and Mnih (2015).
# http://arxiv.org/pdf/1312.5602.pdf
# http://arxiv.org/abs/1509.06461
class DQNAgentEnsemble(AbstractDQNAgent):
    """
    Ensemble DQN agent, with sequential update of all ensemble members.

    # Arguments
        model__: A Keras model.
        policy__: A Keras-rl policy that are defined in
           [policy](https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py).
        test_policy__: A Keras-rl policy.
        enable_double_dqn__: A boolean which enable target network as a second network proposed by van Hasselt et al.
           to decrease overfitting.
        enable_dueling_dqn__: A boolean which enable dueling architecture proposed by Mnih et al.
        dueling_type__: If `enable_dueling_dqn` is set to `True`, a type of dueling architecture must be chosen which
           calculate Q(s,a) from V(s) and A(s,a) differently. Note that `avg` is recommanded in the
           [paper](https://arxiv.org/abs/1511.06581).
                `avg`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
                `max`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
                `naive`: Q(s,a;theta) = V(s;theta) + A(s,a;theta)

    """

    def __init__(self, models, policy=None, test_policy=None, enable_double_dqn=True, enable_dueling_network=False,
                 dueling_type='avg', *args, **kwargs):
        super(DQNAgentEnsemble, self).__init__(*args, **kwargs)

        # Validate (important) input.
        if hasattr(models[0].output, '__len__') and len(models[0].output) > 1:
            raise ValueError(
                'Model "{}" has more than one output. DQN expects a model that has a single output.'.format(models[0]))
        if models[0].output._keras_shape != (None, self.nb_actions):
            raise ValueError('Model output "{}" has invalid shape. DQN expects a model that has one dimension for each '
                             'action, in this case {}.'.format(models[0].output, self.nb_actions))

        # Parameters.
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type

        # Related objects.
        self.models = models
        self.active_model = np.random.randint(len(self.models))
        if policy is None:
            policy = EpsGreedyQPolicy()
        if test_policy is None:
            test_policy = GreedyQPolicy()
        self.policy = policy
        self.test_policy = test_policy
        self.parallel = False

        # State.
        self.recent_action = None
        self.recent_observation = None
        self.reset_states()

        # Models
        self.trainable_models = []
        self.target_models = None

    def change_active_model(self):
        """ Change which ensemble member that chooses the actions for each training episode."""
        self.active_model = np.random.randint(len(self.models))

    def get_config(self):
        config = super(DQNAgentEnsemble, self).get_config()
        config['enable_double_dqn'] = self.enable_double_dqn
        config['dueling_type'] = self.dueling_type
        config['enable_dueling_network'] = self.enable_dueling_network
        config['model'] = [get_object_config(model) for model in self.models]
        config['policy'] = get_object_config(self.policy)
        config['test_policy'] = get_object_config(self.test_policy)
        if self.compiled:
            config['target_model'] = [get_object_config(target_model) for target_model in self.target_models]
        return config

    def compile(self, optimizer, metrics=None):
        """ Set up the training of the neural network."""
        if metrics is None:
            metrics = []
        metrics += [mean_q]  # register default metrics
        metrics += [max_q]

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_models = [clone_model(model, self.custom_model_objects) for model in self.models]
        for i in range(len(self.models)):
            self.target_models[i].compile(optimizer='sgd', loss='mse')
            self.models[i].compile(optimizer='sgd', loss='mse')

        # Compile model.
        if self.target_model_update < 1.:
            raise Exception("Soft target model updates not implemented yet")
            # # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            # updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
            # optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def clipped_masked_error(args):
            y_true, y_pred, mask = args
            loss = huber_loss(y_true, y_pred, self.delta_clip)
            loss *= mask  # apply element-wise mask
            return K.sum(loss, axis=-1)

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        for model in self.models:
            y_pred = model.output
            y_true = Input(name='y_true', shape=(self.nb_actions,))
            mask = Input(name='mask', shape=(self.nb_actions,))
            loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_true, y_pred, mask])
            ins = [model.input] if type(model.input) is not list else model.input
            trainable_model = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])
            assert len(trainable_model.output_names) == 2
            combined_metrics = {trainable_model.output_names[1]: metrics}
            losses = [
                lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
                lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
            ]
            trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
            self.trainable_models.append(trainable_model)

        self.compiled = True

    def load_weights(self, filepath):
        for i, model in enumerate(self.models):
            model.load_weights(filepath+"_"+str(i))
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        for i, model in enumerate(self.models):
            model.save_weights(filepath+"_"+str(i), overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            for i in range(len(self.models)):
                self.models[i].reset_states()
                self.target_models[i].reset_states()

    def update_target_model_hard(self):
        """ Copy current network parameters to the target network. """
        for i, target_model in enumerate(self.target_models):
            target_model.set_weights(self.models[i].get_weights())

    def forward(self, observation):
        """
        Ask the agent to choose an action based on the current observation.
        Args:
            observation (ndarray): Current observation.

        Returns:
            action (int): Index of chosen action
            info (dict): Information about the Q-values of the chosen action.
        """
        info = {}
        # Select an action.
        state = self.memory.get_recent_state(observation)
        if self.training:
            q_values = self.compute_q_values(state, self.active_model)
            action = self.policy.select_action(q_values=q_values)
            info['q_values'] = q_values
        else:
            q_values_all_nets = self.compute_q_values_all_nets(state)
            action, policy_info = self.test_policy.select_action(q_values_all_nets=q_values_all_nets)
            info['q_values_all_nets'] = q_values_all_nets
            info['mean'] = np.mean(q_values_all_nets[:, :], axis=0)
            info['standard_deviation'] = np.std(q_values_all_nets[:, :], axis=0)
            info['coefficient_of_variation'] = np.std(q_values_all_nets[:, :], axis=0) / \
                                               np.mean(q_values_all_nets[:, :], axis=0)
            info.update(policy_info)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action, info

    def backward(self, reward, terminal):
        """ Store the most recent experience in the replay memory and update all ensemble networks. """
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = None
        for active_net in range(len(self.models)):
            metrics = self.train_single_net(active_net)

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics   # This is only the metrics of the last agent.

    def train_single_net(self, active_net):
        """ Retrieve a batch of experiences from the replay memory of the specified ensemble member and update
        the network weights. """

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(active_net, self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                # According to the paper "Deep Reinforcement Learning with Double Q-learning"
                # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
                # while the target network is used to estimate the Q value.
                q_values = self.models[active_net].predict_on_batch(state1_batch)
                assert q_values.shape == (self.batch_size, self.nb_actions)
                actions = np.argmax(q_values, axis=1)
                assert actions.shape == (self.batch_size,)

                # Now, estimate Q values using the target network but select the values with the
                # highest Q value wrt to the online model (as computed above).
                target_q_values = self.target_models[active_net].predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = target_q_values[range(self.batch_size), actions]
            else:
                # Compute the q_values given state1, and extract the maximum for each sample in the batch.
                # We perform this prediction on the target_model instead of the model for reasons
                # outlined in Mnih (2015). In short: it makes the algorithm more stable.
                target_q_values = self.target_models[active_net].predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = np.max(target_q_values, axis=1).flatten()
            assert q_batch.shape == (self.batch_size,)

            targets = np.zeros((self.batch_size, self.nb_actions))
            dummy_targets = np.zeros((self.batch_size,))
            masks = np.zeros((self.batch_size, self.nb_actions))

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = self.gamma * q_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch
            for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                target[action] = R  # update action with estimated accumulated reward
                dummy_targets[idx] = R
                mask[action] = 1.  # enable loss for this specific action
            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')

            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            ins = [state0_batch] if type(self.models[active_net].input) is not list else state0_batch
            metrics = self.trainable_models[active_net].train_on_batch(ins + [targets, masks], [dummy_targets, targets])
            metrics = [metric for idx, metric in enumerate(metrics) if
                       idx not in (1, 2)]  # throw away individual losses
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

            metrics += [self.active_model]

        return metrics

    @property
    def layers(self):
        warnings.warn("Using layers in dqn, which has not been updated to ensemble.")
        return self.model.layers[:]    # Unsure how this function is used and therefore which model to use

    @property
    def metrics_names(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_models[0].output_names) == 2
        dummy_output_name = self.trainable_models[0].output_names[1]
        model_metrics = [name for idx, name in enumerate(self.trainable_models[0].metrics_names) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        names += ["active_model"]
        return names

    @property
    def policy(self):
        return self.__policy

    @policy.setter
    def policy(self, policy):
        self.__policy = policy
        self.__policy._set_agent(self)

    @property
    def test_policy(self):
        return self.__test_policy

    @test_policy.setter
    def test_policy(self, policy):
        self.__test_policy = policy
        self.__test_policy._set_agent(self)


class DQNAgentEnsembleParallel(AbstractDQNAgent):
    """
    Ensemble DQN agent, with parallel update of all ensemble members.

    The speed of the backwards pass of the ensemble members is significantly increased if all members are updated
    in parallel.

    # Arguments
        model__: A Keras model.
        policy__: A Keras-rl policy that are defined in
           [policy](https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py).
        test_policy__: A Keras-rl policy.
        enable_double_dqn__: A boolean which enable target network as a second network proposed by van Hasselt et al.
           to decrease overfitting.
        enable_dueling_dqn__: A boolean which enable dueling architecture proposed by Mnih et al.
        dueling_type__: If `enable_dueling_dqn` is set to `True`, a type of dueling architecture must be chosen which
           calculate Q(s,a) from V(s) and A(s,a) differently. Note that `avg` is recommanded in the
           [paper](https://arxiv.org/abs/1511.06581).
                `avg`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
                `max`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
                `naive`: Q(s,a;theta) = V(s;theta) + A(s,a;theta)

    """

    def __init__(self, nb_models, learning_rate, nb_ego_states, nb_states_per_vehicle, nb_vehicles,  nb_conv_layers,
                 nb_conv_filters, nb_hidden_fc_layers, nb_hidden_neurons, network_seed, prior_scale_factor,
                 window_length, policy=None, test_policy=None, enable_double_dqn=True, enable_dueling_network=False,
                 dueling_type='avg', *args, **kwargs):
        super(DQNAgentEnsembleParallel, self).__init__(*args, **kwargs)

        # Parameters.
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type
        # Related objects.
        self.nb_models = nb_models
        self.active_model = np.random.randint(nb_models)
        if policy is None:
            policy = EpsGreedyQPolicy()
        if test_policy is None:
            test_policy = GreedyQPolicy()
        self.policy = policy
        self.test_policy = test_policy
        self.lr = learning_rate

        # Network parameters
        self.nb_ego_states = nb_ego_states
        self.nb_states_per_vehicle = nb_states_per_vehicle
        self.nb_vehicles = nb_vehicles
        self.nb_conv_layers = nb_conv_layers
        self.nb_conv_filters = nb_conv_filters
        self.nb_hidden_fc_layers = nb_hidden_fc_layers
        self.nb_hidden_neurons = nb_hidden_neurons
        self.network_seed = network_seed
        self.prior_scale_factor = prior_scale_factor
        self.window_length = window_length

        # State.
        self.recent_action = None
        self.recent_observation = None
        self.reset_states()

        self.parallel = True
        self.input_queues = None
        self.output_queues = None

        self.init_parallel_execution()
        self.compiled = True

    def init_parallel_execution(self):
        """ Initalize one worker for each ensemble member and set up corresponding queues. """
        self.input_queues = [mp.Queue() for _ in range(self.nb_models)]
        self.output_queues = [mp.Queue() for _ in range(self.nb_models)]
        workers = []
        for i in range(self.nb_models):
            worker = Worker(self.network_seed + i, self.input_queues[i], self.output_queues[i],
                            nb_ego_states=self.nb_ego_states, nb_states_per_vehicle=self.nb_states_per_vehicle,
                            nb_vehicles=self.nb_vehicles, nb_actions=self.nb_actions,
                            nb_conv_layers=self.nb_conv_layers, nb_conv_filters=self.nb_conv_filters,
                            nb_hidden_fc_layers=self.nb_hidden_fc_layers, nb_hidden_neurons=self.nb_hidden_neurons,
                            duel=True, prior_scale_factor=self.prior_scale_factor, window_length=self.window_length,
                            processor=self.processor, batch_size=self.batch_size,
                            enable_double_dqn=self.enable_double_dqn, gamma=self.gamma, lr=self.lr,
                            delta_clip=self.delta_clip, target_model_update=self.target_model_update,
                            policy=self.policy)
            workers.append(worker)
        for worker in workers:
            worker.start()

    def change_active_model(self):
        """ Change which ensemble member that chooses the actions for each training episode."""
        self.active_model = np.random.randint(self.nb_models)

    def get_config(self):
        config = super(DQNAgentEnsemble, self).get_config()
        config['enable_double_dqn'] = self.enable_double_dqn
        config['dueling_type'] = self.dueling_type
        config['enable_dueling_network'] = self.enable_dueling_network
        config['model'] = [get_object_config(model) for model in self.nb_models]
        config['policy'] = get_object_config(self.policy)
        config['test_policy'] = get_object_config(self.test_policy)
        if self.compiled:
            config['target_model'] = [get_object_config(target_model) for target_model in self.nb_models]
        return config

    def get_model_as_string(self):
        self.input_queues[0].put(['model_as_string'])   # All models are the same, so enough to get one of them
        return self.output_queues[0].get()

    def load_weights(self, filepath):
        for i in range(self.nb_models):
            self.input_queues[i].put(['load_weights', filepath+"_"+str(i)])
            output = self.output_queues[i].get()
            assert(output == 'weights_loaded_' + str(i))
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        for i in range(self.nb_models):
            self.input_queues[i].put(['save_weights', filepath+"_"+str(i), overwrite])
            output = self.output_queues[i].get()
            assert(output == 'weights_saved_' + str(i))

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            for i in range(self.nb_models):
                self.input_queues[i].put(['reset_states'])
                out = self.output_queues[i].get()
                assert(out == 'reset_states_done_' + str(i))

    def update_target_model_hard(self):
        for i in range(self.nb_models):
            self.input_queues[i].put(['update_target_model'])
            output = self.output_queues[i].get()
            assert(output == 'target_model_updated_' + str(i))

    def forward(self, observation):
        info = {}
        # Select an action.
        state = self.memory.get_recent_state(observation)
        if self.training:
            q_values = self.compute_q_values(state, self.active_model)
            action = self.policy.select_action(q_values=q_values)
            info['q_values'] = q_values
        else:
            q_values_all_nets = self.compute_q_values_all_nets(state)
            action, policy_info = self.test_policy.select_action(q_values_all_nets=q_values_all_nets)
            info['q_values_all_nets'] = q_values_all_nets
            info['mean'] = np.mean(q_values_all_nets[:, :], axis=0)
            info['standard_deviation'] = np.std(q_values_all_nets[:, :], axis=0)
            info['coefficient_of_variation'] = np.std(q_values_all_nets[:, :], axis=0) / \
                                               np.abs(np.mean(q_values_all_nets[:, :], axis=0))
            info.update(policy_info)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action, info

    def backward(self, reward, terminal):
        """ Store the most recent experience in the replay memory and update all ensemble networks. """
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if self.training:
            if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
                for net in range(self.nb_models):
                    experiences = self.memory.sample(net, self.batch_size)
                    assert len(experiences) == self.batch_size
                    self.input_queues[net].put(['train', experiences])

                for net in range(self.nb_models):   # Wait for all workers to finish
                    output = self.output_queues[net].get()
                    if net == self.nb_models - 1:   # Store the metrics of the last agent
                        metrics = output[1]
                    assert(output[0] == 'training_done_' + str(net))

                    metrics += [self.active_model]

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics   # This is only the metrics of the last agent.

    @property
    def layers(self):
        warnings.warn("Using layers in dqn, which has not been updated to ensemble.")
        return self.model.layers[:]    # Unsure how this function is used and therefore which model to use

    @property
    def metrics_names(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        self.input_queues[0].put(['output_names'])
        output_names = self.output_queues[0].get()
        assert len(output_names) == 2
        dummy_output_name = output_names[1]
        self.input_queues[0].put(['metrics_names'])
        metrics_names_ = self.output_queues[0].get()
        model_metrics = [name for idx, name in enumerate(metrics_names_) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        names += ["active_model"]
        return names

    @property
    def policy(self):
        return self.__policy

    @policy.setter
    def policy(self, policy):
        self.__policy = policy
        self.__policy._set_agent(self)

    @property
    def test_policy(self):
        return self.__test_policy

    @test_policy.setter
    def test_policy(self, policy):
        self.__test_policy = policy
        self.__test_policy._set_agent(self)


class Worker(mp.Process):
    """
    Creates a set of workers that maintains each ensemble member.

    Args:
        seed (int): Seed of worker. Needs to be unique, otherwise all network parameters will be initialized equally.
        input_queue (multiprocessing.queues.Queue): Input queue for worker tasks.
        output_queue (multiprocessing.queues.Queue): Output queue for worker tasks.
        nb_ego_states (int): Number of states that describe the ego vehicle.
        nb_ego_states (int): Number of states that describe the ego vehicle.
        nb_states_per_vehicle (int): Number of states that describe each of the surrounding vehicles.
        nb_vehicles (int): Maximum number of surrounding vehicles.
        nb_actions: (int): Number of outputs from the network.
        nb_conv_layers (int): Number of convolutional layers.
        nb_conv_filters (int): Number of convolutional filters.
        nb_hidden_fc_layers (int): Number of hidden layers.
        nb_hidden_neurons (int): Number of neurons in the hidden layers.
        duel (bool): Use dueling architecture.
        prior_scale_factor (float): Scale factor that balances trainable/untrainable contribution to the output.
        window_length (int): How many historic states that are used as input. Set to 1 in this work.
        processor: Not used
        batch_size (int): Batch size for stochastic gradient descent.
        enable_double_dqn (bool): True if double DQN is used, otherwise false
        gamma (float): MDP discount factor.
        lr (float): Learning rate-
        delta_clip (float): Huber loss parameter.
        target_model_update (int): Steps between copying the paramters of the trained network to the target network.
        policy: Policy of the agent
    """
    def __init__(self, seed, input_queue, output_queue, nb_ego_states, nb_states_per_vehicle, nb_vehicles, nb_actions,
                 nb_conv_layers, nb_conv_filters, nb_hidden_fc_layers, nb_hidden_neurons, duel, prior_scale_factor,
                 window_length, processor, batch_size, enable_double_dqn, gamma, lr, delta_clip, target_model_update,
                 policy, verbose=0):
        mp.Process.__init__(self)
        self.seed = seed
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.nb_ego_states = nb_ego_states
        self.nb_states_per_vehicle = nb_states_per_vehicle
        self.nb_vehicles = nb_vehicles
        self.nb_actions = nb_actions
        self.nb_conv_layers = nb_conv_layers
        self.nb_conv_filters = nb_conv_filters
        self.nb_hidden_fc_layers = nb_hidden_fc_layers
        self.nb_hidden_neurons = nb_hidden_neurons
        self.duel = duel
        self.prior_scale_factor = prior_scale_factor
        self.window_length = window_length

        self.processor = processor
        self.batch_size = batch_size
        self.enable_double_dqn = enable_double_dqn
        self.gamma = gamma
        self.target_model_update = target_model_update
        self.delta_clip = delta_clip
        self.lr = lr
        self.policy = policy

        self.verbose = verbose
        self.model = None
        self.target_model = None
        self.trainable_model = None

    def run(self):
        """ Initializes individual networks and starts the workers for each ensemble member. """
        np.random.seed(self.seed)
        proc_name = str(int(self.name[self.name.find('-')+1:]) - 1)
        n = NetworkCNN(nb_ego_states=self.nb_ego_states, nb_states_per_vehicle=self.nb_states_per_vehicle,
                       nb_vehicles=self.nb_vehicles, nb_actions=self.nb_actions, nb_conv_layers=self.nb_conv_layers,
                       nb_conv_filters=self.nb_conv_filters, nb_hidden_fc_layers=self.nb_hidden_fc_layers,
                       nb_hidden_neurons=self.nb_hidden_neurons, duel=self.duel, prior=True,
                       prior_scale_factor=self.prior_scale_factor, window_length=self.window_length,
                       activation='relu', duel_type='avg')
        self.model = n.model
        self.compile()

        while True:
            input_ = self.input_queue.get()
            if self.verbose:
                print("Read input proc " + str(proc_name) + ' ' + input_[0])
            if input_ is None:  # If sending None, the process is killed
                break

            if input_[0] == 'predict':
                output = self.model.predict_on_batch(input_[1])
            elif input_[0] == 'train':
                metrics = self.train_single_net(experiences=input_[1])
                output = ['training_done_' + proc_name, metrics]
            elif input_[0] == 'reset_states':
                self.model.reset_states()
                self.target_model.reset_states()
                output = 'reset_states_done_' + proc_name
            elif input_[0] == 'update_target_model':
                self.target_model.set_weights(self.model.get_weights())
                output = 'target_model_updated_' + proc_name
            elif input_[0] == 'save_weights':
                self.model.save_weights(input_[1], overwrite=input_[2])
                output = 'weights_saved_' + proc_name
            elif input_[0] == 'load_weights':
                self.model.load_weights(input_[1])
                output = 'weights_loaded_' + proc_name
            elif input_[0] == 'output_names':
                output = self.trainable_model.output_names
            elif input_[0] == 'metrics_names':
                output = self.trainable_model.metrics_names
            elif input_[0] == 'model_as_string':
                output = self.model.to_json()

            else:
                raise Exception('input command not defined')

            self.output_queue.put(output)
        return

    def compile(self, metrics=None):
        """ Set up the training of the neural network."""
        if metrics is None:
            metrics = []
        metrics += [mean_q]  # register default metrics
        metrics += [max_q]

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.model)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')

        # Compile model.
        if self.target_model_update < 1.:
            raise Exception("Soft target model updates not implemented yet")
            # # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            # updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
            # optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def clipped_masked_error(args):
            y_true, y_pred, mask = args
            loss = huber_loss(y_true, y_pred, self.delta_clip)
            loss *= mask  # apply element-wise mask
            return K.sum(loss, axis=-1)

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        y_pred = self.model.output
        y_true = Input(name='y_true', shape=(self.nb_actions,))
        mask = Input(name='mask', shape=(self.nb_actions,))
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_true, y_pred, mask])
        ins = [self.model.input] if type(self.model.input) is not list else self.model.input
        self.trainable_model = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])
        assert len(self.trainable_model.output_names) == 2
        combined_metrics = {self.trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        self.trainable_model.compile(optimizer=Adam(lr=self.lr), loss=losses, metrics=combined_metrics)

    def train_single_net(self, experiences):
        """ Retrieve a batch of experiences from the replay memory of the ensemble member and update
        the network weights. """
        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in experiences:
            state0_batch.append(e.state0)
            state1_batch.append(e.state1)
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            terminal1_batch.append(0. if e.terminal1 else 1.)

        # Prepare and validate parameters.
        state0_batch = self.process_state_batch(state0_batch)
        state1_batch = self.process_state_batch(state1_batch)
        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)
        assert reward_batch.shape == (self.batch_size,)
        assert terminal1_batch.shape == reward_batch.shape
        assert len(action_batch) == len(reward_batch)

        # Compute Q values for mini-batch update.
        if self.enable_double_dqn:
            # According to the paper "Deep Reinforcement Learning with Double Q-learning"
            # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
            # while the target network is used to estimate the Q value.
            q_values = self.model.predict_on_batch(state1_batch)
            assert q_values.shape == (self.batch_size, self.nb_actions)
            actions = np.argmax(q_values, axis=1)
            assert actions.shape == (self.batch_size,)

            # Now, estimate Q values using the target network but select the values with the
            # highest Q value wrt to the online model (as computed above).
            target_q_values = self.target_model.predict_on_batch(state1_batch)
            assert target_q_values.shape == (self.batch_size, self.nb_actions)
            q_batch = target_q_values[range(self.batch_size), actions]
        else:
            # Compute the q_values given state1, and extract the maximum for each sample in the batch.
            # We perform this prediction on the target_model instead of the model for reasons
            # outlined in Mnih (2015). In short: it makes the algorithm more stable.
            target_q_values = self.target_model.predict_on_batch(state1_batch)
            assert target_q_values.shape == (self.batch_size, self.nb_actions)
            q_batch = np.max(target_q_values, axis=1).flatten()
        assert q_batch.shape == (self.batch_size,)

        targets = np.zeros((self.batch_size, self.nb_actions))
        dummy_targets = np.zeros((self.batch_size,))
        masks = np.zeros((self.batch_size, self.nb_actions))

        # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
        # but only for the affected output units (as given by action_batch).
        discounted_reward_batch = self.gamma * q_batch
        # Set discounted reward to zero for all states that were terminal.
        discounted_reward_batch *= terminal1_batch
        assert discounted_reward_batch.shape == reward_batch.shape
        Rs = reward_batch + discounted_reward_batch
        for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
            target[action] = R  # update action with estimated accumulated reward
            dummy_targets[idx] = R
            mask[action] = 1.  # enable loss for this specific action
        targets = np.array(targets).astype('float32')
        masks = np.array(masks).astype('float32')

        # Finally, perform a single update on the entire batch. We use a dummy target since
        # the actual loss is computed in a Lambda layer that needs more complex input. However,
        # it is still useful to know the actual target to compute metrics properly.
        ins = [state0_batch] if type(self.model.input) is not list else state0_batch
        metrics = self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets])
        metrics = [metric for idx, metric in enumerate(metrics) if
                   idx not in (1, 2)]  # throw away individual losses
        metrics += self.policy.metrics
        if self.processor is not None:
            metrics += self.processor.metrics

        return metrics

    def process_state_batch(self, batch):
        """ Heritage from keras-rl, not used here. """
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)


class UpdateActiveModelCallback(Callback):
    """
    Callback that decides which ensemble neural network that is active for each training episode.

    For exploration during training, one of the ensemble networks decides which action to take by greedily maximizing
    the Q-value. This callback sets which ensemble neural network that should be active for the episode that is about
    to start.

    Args:
        dqn: The ensemble DQN agent.
    """
    def __init__(self, dqn):
        super(UpdateActiveModelCallback, self).__init__()
        self.dqn = dqn

    def on_episode_begin(self, episode, logs={}):
        """ Change which ensemble member that is active. """
        self.dqn.change_active_model()


def max_q(y_true, y_pred):
    """ Returns average maximum Q-value of training batch. """
    return K.mean(K.max(y_pred, axis=-1))


def mean_q(y_true, y_pred):
    """ Returns average Q-value of training batch. """
    return K.mean(K.mean(y_pred, axis=-1))
