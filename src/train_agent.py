"""
Train agent with DQN method or ensemble with randomized prior functions method.

This script should be called from the folder src, where the script is located.

The parameters of the training are set in "parameters.py".
The parameters of the highway driving environment are set in "parameters_simulation.py".

Logfiles are stored in ../logs/train_agent_DATE_TIME
The script can be called with an optional argument NAME, which sets the name of the log. The log is then stored in
../logs/train_agent_DATE_TIME_NAME

In the log folder, the following is stored:
- A copy of the parameters and the code that was used for the run.
- The weights of the neural networks at different times during the training process. Named as the training step,
  with additional _N if using an ensemble, where N is the index of the ensemble member.
- A tensorboard log.
- csv-files which stores:
   - test_rewards.csv: the total reward for each test episode
   - test_steps.csv: number of steps for each test episode
   - test_individual_action_data.csv: the actions that were taken during the test episodes
   - test_individual_reward_data.csv: the individual rewards that were obtained during the test episodes
   - test_individual_qvalues_data.cvs: the estimated Q-values for all actions during the test episodes
"""


import numpy as np
import random   # Required to set random seed for replay memory

import os
import datetime
import sys
from shutil import copytree, copyfile

from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import SequentialMemory

from dqn_standard import DQNAgent
from dqn_ensemble import DQNAgentEnsemble, DQNAgentEnsembleParallel, UpdateActiveModelCallback
from memory import BootstrappingMemory
from policy import EnsembleTestPolicy
from driving_env import Highway
from network_architecture import NetworkMLP, NetworkCNN
from callbacks import SaveWeights, EvaluateAgent

import parameters as p
import parameters_simulation as ps


# Set log path and name
start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_name = os.path.basename(__file__)[0:-3]+"_"+start_time+("_"+sys.argv[1] if len(sys.argv) > 1 else "")
save_path = "../logs/"+log_name

# Save parameters and code
if not os.path.isdir(save_path):
    if not os.path.isdir('../logs'):
        os.mkdir('../logs')
    os.mkdir(save_path)
    os.mkdir(save_path + '/src')
for file in os.listdir('.'):
    if file[-3:] == '.py':
        copyfile('./' + file, save_path + '/src/' + file[:-3] + '_stored.py')

env = Highway(sim_params=ps.sim_params, road_params=ps.road_params, use_gui=False)
nb_actions = env.nb_actions
nb_observations = env.nb_observations

np.random.seed(p.random_seed)
random.seed(p.random_seed)   # memory.py uses random module

save_weights_callback = SaveWeights(p.save_freq, save_path)
evaluate_agent_callback = EvaluateAgent(eval_freq=p.eval_freq, nb_eval_eps=p.nb_eval_eps, save_path=save_path)
tensorboard_callback = TensorBoard(log_dir=save_path, histogram_freq=0, write_graph=True, write_images=False)
callbacks = [tensorboard_callback, save_weights_callback, evaluate_agent_callback]

# This structure initializes the agent. The different options allows the choice of using a
# convolutional or fully connected neural network architecture,
# and to run the backpropagation of the ensemble members in parallel or sequential.
if p.agent_par["parallel"]:
    if not p.agent_par['ensemble']:
        raise Exception('Parallel mode only works with ensemble DQN.')
    nb_models = p.agent_par['number_of_networks']
    policy = GreedyQPolicy()
    test_policy = EnsembleTestPolicy('mean')
    memory = BootstrappingMemory(nb_nets=p.agent_par['number_of_networks'], limit=p.agent_par['buffer_size'],
                                 adding_prob=p.agent_par["adding_prob"], window_length=p.agent_par["window_length"])
    dqn = DQNAgentEnsembleParallel(nb_models=nb_models, learning_rate=p.agent_par['learning_rate'],
                                   nb_ego_states=env.nb_ego_states, nb_states_per_vehicle=env.nb_states_per_vehicle,
                                   nb_vehicles=ps.sim_params['sensor_nb_vehicles'],
                                   nb_conv_layers=p.agent_par['nb_conv_layers'],
                                   nb_conv_filters=p.agent_par['nb_conv_filters'],
                                   nb_hidden_fc_layers=p.agent_par['nb_hidden_fc_layers'],
                                   nb_hidden_neurons=p.agent_par['nb_hidden_neurons'], policy=policy,
                                   test_policy=test_policy, enable_double_dqn=p.agent_par['double_q'],
                                   enable_dueling_network=False, nb_actions=nb_actions,
                                   prior_scale_factor=p.agent_par['prior_scale_factor'],
                                   window_length=p.agent_par['window_length'], memory=memory,
                                   gamma=p.agent_par['gamma'], batch_size=p.agent_par['batch_size'],
                                   nb_steps_warmup=p.agent_par['learning_starts'],
                                   train_interval=p.agent_par['train_freq'],
                                   target_model_update=p.agent_par['target_network_update_freq'],
                                   delta_clip=p.agent_par['delta_clip'], network_seed=p.random_seed)
    callbacks.append(UpdateActiveModelCallback(dqn))
    model_as_string = dqn.get_model_as_string()

else:
    if p.agent_par["ensemble"]:
        models = []
        for i in range(p.agent_par["number_of_networks"]):
            if p.agent_par['cnn']:
                models.append(NetworkCNN(env.nb_ego_states, env.nb_states_per_vehicle,
                                         ps.sim_params['sensor_nb_vehicles'], nb_actions,
                                         nb_conv_layers=p.agent_par['nb_conv_layers'],
                                         nb_conv_filters=p.agent_par['nb_conv_filters'],
                                         nb_hidden_fc_layers=p.agent_par['nb_hidden_fc_layers'],
                                         nb_hidden_neurons=p.agent_par['nb_hidden_neurons'],
                                         duel=p.agent_par['duel_q'], prior=True,  activation='relu',
                                         window_length=p.agent_par["window_length"], duel_type='avg',
                                         prior_scale_factor=p.agent_par["prior_scale_factor"]).model)
            else:
                models.append(NetworkMLP(nb_observations, nb_actions,
                                         nb_hidden_layers=p.agent_par['nb_hidden_fc_layers'],
                                         nb_hidden_neurons=p.agent_par['nb_hidden_neurons'], duel=p.agent_par['duel_q'],
                                         prior=True, activation='relu',
                                         prior_scale_factor=p.agent_par["prior_scale_factor"], duel_type='avg',
                                         window_length=p.agent_par["window_length"]).model)
        print(models[0].summary())
        model_as_string = models[0].to_json()
        policy = GreedyQPolicy()
        test_policy = EnsembleTestPolicy('mean')
        memory = BootstrappingMemory(nb_nets=p.agent_par['number_of_networks'], limit=p.agent_par['buffer_size'],
                                     adding_prob=p.agent_par["adding_prob"], window_length=p.agent_par["window_length"])
        dqn = DQNAgentEnsemble(models=models, policy=policy, test_policy=test_policy,
                               enable_double_dqn=p.agent_par['double_q'],
                               enable_dueling_network=False, nb_actions=nb_actions, memory=memory,
                               gamma=p.agent_par['gamma'], batch_size=p.agent_par['batch_size'],
                               nb_steps_warmup=p.agent_par['learning_starts'], train_interval=p.agent_par['train_freq'],
                               target_model_update=p.agent_par['target_network_update_freq'],
                               delta_clip=p.agent_par['delta_clip'])
        callbacks.append(UpdateActiveModelCallback(dqn))

    else:
        if p.agent_par['cnn']:
            model = NetworkCNN(env.nb_ego_states, env.nb_states_per_vehicle, ps.sim_params['sensor_nb_vehicles'],
                               nb_actions, nb_conv_layers=p.agent_par['nb_conv_layers'],
                               nb_conv_filters=p.agent_par['nb_conv_filters'],
                               nb_hidden_fc_layers=p.agent_par['nb_hidden_fc_layers'],
                               nb_hidden_neurons=p.agent_par['nb_hidden_neurons'], duel=p.agent_par['duel_q'],
                               prior=False, activation='relu', window_length=p.agent_par["window_length"],
                               duel_type='avg').model
        else:
            model = NetworkMLP(nb_observations, nb_actions, nb_hidden_layers=p.agent_par['nb_hidden_fc_layers'],
                               nb_hidden_neurons=p.agent_par['nb_hidden_neurons'], duel=p.agent_par['duel_q'],
                               prior=False, activation='relu', duel_type='avg',
                               window_length=p.agent_par["window_length"]).model
        print(model.summary())
        model_as_string = model.to_json()
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                                      value_min=p.agent_par['exploration_final_eps'], value_test=.0,
                                      nb_steps=p.agent_par['exploration_steps'])
        test_policy = GreedyQPolicy()
        memory = SequentialMemory(limit=p.agent_par['buffer_size'], window_length=p.agent_par["window_length"])
        dqn = DQNAgent(model=model, policy=policy, test_policy=test_policy, enable_double_dqn=p.agent_par['double_q'],
                       enable_dueling_network=False, nb_actions=nb_actions, memory=memory,
                       gamma=p.agent_par['gamma'], batch_size=p.agent_par['batch_size'],
                       nb_steps_warmup=p.agent_par['learning_starts'], train_interval=p.agent_par['train_freq'],
                       target_model_update=p.agent_par['target_network_update_freq'],
                       delta_clip=p.agent_par['delta_clip'])

    dqn.compile(Adam(lr=p.agent_par['learning_rate']))

with open(save_path+"/"+'model.txt', 'w') as text_file:
    text_file.write(model_as_string)

# Run training
dqn.fit(env, nb_steps=p.nb_training_steps, visualize=False, verbose=2, callbacks=callbacks)
