"""
Run trained agent on test episodes or special cases.

This script should be called from the folder src, where the script is located.

The options for what to run are set below, after the import statements. The options are:
- Which agent to run, defined by:
   - filepath = '../logs/train_agent_DATE_TIME_NAME/', which is the folder name of log file for a specific run
   - agent name = 'STEP', which chooses the agent that was saved after STEP training steps.
                          In case of ensemble, only use STEP and do not include _N, where N is the ensemble index.
- Which case to run:
   - case = 'CASE_NAME', options 'rerun_test_scenarios', 'fast_overtaking', 'standstill'
- Ensemble test policy:
   - use_ensemble_test_policy = True/False, which specifies it the ensemble safety policy should be used or not
   - safety_threshold = e.g. 0.02, which defines the maximum allowed coefficient of variation for an action
- Save run as video:
   - save_video = True/False, if True, a set of images are stored in '../Videos/', which can be used to create a video

The different cases are:
- rerun_test_scenarios: This option will run the trained agent on the same test episodes that are used to evaluate
                        the performance of the agent during the training process.
- fast_overtaking: This option demonstrates how the safety criterion of the ensemble RPF agent can be used,
                   and is the same as was included in the paper.
- standstill: This option demonstrates how the safety criterion of the ensemble RPF agent can be used,
              and is the same as was included in the paper.
"""

import os
import sys
import numpy as np
from keras.optimizers import Adam
from keras.models import model_from_json, clone_model
from rl.policy import GreedyQPolicy
from rl.memory import Memory
from dqn_standard import DQNAgent
from dqn_ensemble import DQNAgentEnsemble, UpdateActiveModelCallback
from policy import EnsembleTestPolicy
from driving_env import Highway
import traci
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42   # To avoid Type 3 fonts in figures
rcParams['ps.fonttype'] = 42

""" Options: """
filepath = '../logs/train_agent_DATE_TIME_NAME/'
agent_name = 'STEP'
case = 'rerun_test_scenarios'   # 'rerun_test_scenarios', 'fast_overtaking', 'standstill'
use_ensemble_test_policy = False
safety_threshold = 0.02   # Only used if ensemble test policy is chosen
save_video = False
""" End options """

# These import statements need to come after the choice of which agent that should be used.
sys.path.insert(0, filepath + 'src/')
import parameters_stored as p
import parameters_simulation_stored as ps


ps.sim_params['remove_sumo_warnings'] = False
nb_actions = len(ps.sim_params['action_interp'])

with open(filepath + 'model.txt') as text_file:
    saved_model = model_from_json(text_file.read())
if p.agent_par["ensemble"]:
    models = []
    for i in range(p.agent_par["number_of_networks"]):
        models.append(clone_model(saved_model))
    print(models[0].summary())
else:
    model = saved_model
    print(model.summary())

memory = Memory(window_length=p.agent_par['window_length'])   # Not used, simply needed to create the agent

if p.agent_par["ensemble"]:
    policy = GreedyQPolicy()
    test_policy = EnsembleTestPolicy('mean')
    dqn = DQNAgentEnsemble(models=models, policy=policy, test_policy=test_policy,
                           enable_double_dqn=p.agent_par['double_q'], enable_dueling_network=False,
                           nb_actions=nb_actions, memory=memory, gamma=p.agent_par['gamma'],
                           batch_size=p.agent_par['batch_size'], nb_steps_warmup=p.agent_par['learning_starts'],
                           train_interval=p.agent_par['train_freq'],
                           target_model_update=p.agent_par['target_network_update_freq'],
                           delta_clip=p.agent_par['delta_clip'])
else:
    policy = GreedyQPolicy()
    test_policy = GreedyQPolicy()
    dqn = DQNAgent(model=model, policy=policy, enable_double_dqn=p.agent_par['double_q'],
                   enable_dueling_network=False, nb_actions=nb_actions, memory=memory,
                   gamma=p.agent_par['gamma'], batch_size=p.agent_par['batch_size'],
                   nb_steps_warmup=p.agent_par['learning_starts'], train_interval=p.agent_par['train_freq'],
                   target_model_update=p.agent_par['target_network_update_freq'], delta_clip=p.agent_par['delta_clip'])
dqn.compile(Adam(lr=p.agent_par['learning_rate']))

dqn.load_weights(filepath+agent_name)
dqn.training = False

if use_ensemble_test_policy:
    test_policy_safe = EnsembleTestPolicy(safety_threshold=safety_threshold, safe_action=3)
    dqn.test_policy = test_policy_safe

if save_video and not os.path.isdir("../videos"):
    os.mkdir('../videos')

if case == 'rerun_test_scenarios':
    env = Highway(sim_params=ps.sim_params, road_params=ps.road_params, use_gui=True)
    env.reset()
    if save_video:
        try:
            traci.gui.setSchema('View #0', 'scheme_for_videos')
        except:
            print("SUMO scheme not defined.")
        if not os.path.isdir("../videos"):
            os.mkdir('../videos')
    episode_rewards = []
    episode_steps = []
    nb_safe_actions_per_episode = []
    for i in range(0, 100):
        np.random.seed(i)
        obs = env.reset()
        if save_video:
            video_folder = "../videos/rerun_" + str(i) + "___" + filepath[8:-1]
            if not os.path.isdir(video_folder):
                os.mkdir(video_folder)
                os.mkdir(video_folder + "/images")
            traci.gui.setZoom('View #0', 10500)
            traci.gui.screenshot("View #0", video_folder + "/images/" + str(0) + ".png")
        done = False
        episode_reward = 0
        step = 0
        nb_safe_actions = 0
        max_coef_of_var = 0
        while done is False:
            action, action_info = dqn.forward(obs)
            obs, rewards, done, info = env.step(action, action_info)
            episode_reward += rewards
            step += 1
            if 'safe_action' in action_info:
                if action_info['safe_action']:
                    nb_safe_actions += 1
                if action_info['coefficient_of_variation'][action] > max_coef_of_var:
                    max_coef_of_var = action_info['coefficient_of_variation'][action]
            if save_video:
                traci.gui.screenshot("View #0", video_folder + "/images/" + str(step) + ".png")

        episode_rewards.append(episode_reward)
        episode_steps.append(step)
        nb_safe_actions_per_episode.append(nb_safe_actions)
        print("Episode: " + str(i))
        print("Episode steps: " + str(step))
        print("Episode reward: " + str(episode_reward))
        print("Number of safety actions: " + str(nb_safe_actions))
        print('Max coef of var: ' + str(max_coef_of_var))

    print(episode_rewards)
    print(episode_steps)
    print(nb_safe_actions_per_episode)

elif case == 'fast_overtaking':
    ps.sim_params['nb_vehicles'] = 5
    env = Highway(sim_params=ps.sim_params, road_params=ps.road_params, use_gui=True)
    env.reset()
    if save_video:
        try:
            traci.gui.setSchema('View #0', 'scheme_for_videos')
        except:
            print("SUMO scheme not defined.")
    for _ in range(5):   # Display the case 5 times
        # Make sure that the vehicles are not affected by previous state
        np.random.seed(57)
        env.reset()
        if env.use_gui:
            traci.vehicle.setColor('veh2', (178, 102, 255))
        s0 = 1000.
        traci.vehicle.moveTo('veh0', 'highway_0', s0 - 300)
        traci.vehicle.moveTo('veh1', 'highway_1', s0 - 300)
        traci.vehicle.moveTo('veh2', 'highway_2', s0 - 300)
        traci.vehicle.moveTo('veh3', 'highway_2', s0 - 400)
        traci.vehicle.moveTo('veh4', 'highway_2', s0 - 500)
        traci.simulationStep()
        env.speeds[0, 0] = 15
        for veh in env.vehicles:
            traci.vehicle.setSpeedMode(veh, 0)
        traci.vehicle.setSpeed('veh0', 15)
        traci.vehicle.setSpeed('veh1', 15)
        traci.vehicle.setSpeed('veh2', 55)
        traci.vehicle.setSpeed('veh3', 15)
        traci.vehicle.setSpeed('veh4', 15)
        traci.vehicle.setMaxSpeed('veh0', 25)
        traci.vehicle.setMaxSpeed('veh1', 15)
        traci.vehicle.setMaxSpeed('veh2', 55)
        traci.vehicle.setMaxSpeed('veh3', 15)
        traci.vehicle.setMaxSpeed('veh4', 15)
        traci.simulationStep()
        env.step(0)
        if env.use_gui:
            traci.vehicle.setColor('veh2', (178, 102, 255))

        # Overtaking case
        traci.vehicle.moveTo('veh0', 'highway_0', s0)
        traci.vehicle.moveTo('veh1', 'highway_0', s0 + 50)
        traci.vehicle.moveTo('veh2', 'highway_1', s0 - 150)
        traci.vehicle.moveTo('veh3', 'highway_2', s0 - 50)
        traci.vehicle.moveTo('veh4', 'highway_2', s0 - 0)
        traci.vehicle.setSpeed('veh0', 15)
        traci.vehicle.setSpeed('veh1', 15)
        traci.vehicle.setSpeed('veh2', 55)
        traci.vehicle.setSpeed('veh3', 15)
        traci.vehicle.setSpeed('veh4', 15)
        if save_video:
            video_folder = "../videos/fast_overtaking___" + filepath[8:-1]
            if not os.path.isdir(video_folder):
                os.mkdir(video_folder)
                os.mkdir(video_folder + "/images")
            traci.gui.setZoom('View #0', 10500)
            traci.gui.screenshot("View #0", video_folder + "/images/" + str(0) + ".png")
        observation, reward, done, info = env.step(0)
        if env.use_gui:
            traci.vehicle.setColor('veh2', (178, 102, 255))
        for veh in env.vehicles[1:]:
            traci.vehicle.setSpeed(veh, -1)

        # Run fast overtaking case
        for i in range(8):
            if save_video:
                traci.gui.screenshot("View #0", video_folder + "/images/" + str(i+1) + ".png")
            action, action_info = dqn.forward(observation)
            observation, reward, done, info = env.step(action, action_info)
            if env.use_gui:
                traci.vehicle.setColor('veh2', (178, 102, 255))

elif case == 'standstill':
    ps.sim_params['nb_vehicles'] = 7
    env = Highway(sim_params=ps.sim_params, road_params=ps.road_params, use_gui=True)
    env.reset()
    if save_video:
        try:
            traci.gui.setSchema('View #0', 'scheme_for_videos')
        except:
            print("SUMO scheme not defined.")
    for _ in range(5):   # Display the case 5 times
        # Make sure that the vehicles are not affected by previous state
        np.random.seed(57)
        env.reset()
        if env.use_gui:
            traci.vehicle.setColor('veh1', (255, 255, 255))
        s0 = 1000.
        traci.vehicle.moveTo('veh0', 'highway_0', s0 - 300)
        traci.vehicle.moveTo('veh1', 'highway_1', s0 - 300)
        traci.vehicle.moveTo('veh2', 'highway_2', s0 - 300)
        traci.vehicle.moveTo('veh3', 'highway_2', s0 - 400)
        traci.vehicle.moveTo('veh4', 'highway_2', s0 - 500)
        traci.vehicle.moveTo('veh5', 'highway_2', s0 - 600)
        traci.vehicle.moveTo('veh6', 'highway_2', s0 - 700)
        traci.simulationStep()
        env.speeds[0, 0] = 25
        for veh in env.vehicles:
            traci.vehicle.setSpeedMode(veh, 0)
            traci.vehicle.setLaneChangeMode(veh, 0)  # Turn off all lane changes
        traci.vehicle.setSpeed('veh0', 25)
        traci.vehicle.setSpeed('veh1', 0)
        traci.vehicle.setSpeed('veh2', 15)
        traci.vehicle.setSpeed('veh3', 15)
        traci.vehicle.setSpeed('veh4', 15)
        traci.vehicle.setSpeed('veh5', 15)
        traci.vehicle.setSpeed('veh6', 15)
        traci.simulationStep()
        traci.vehicle.setMaxSpeed('veh0', 25)
        traci.vehicle.setMaxSpeed('veh1', 0.001)
        traci.vehicle.setMaxSpeed('veh2', 15)
        traci.vehicle.setMaxSpeed('veh3', 15)
        traci.vehicle.setMaxSpeed('veh4', 15)
        traci.vehicle.setMaxSpeed('veh5', 15)
        traci.vehicle.setMaxSpeed('veh6', 15)
        traci.simulationStep()
        env.step(0)
        if env.use_gui:
            traci.vehicle.setColor('veh1', (255, 255, 255))

        # Standstill case
        traci.vehicle.moveTo('veh0', 'highway_0', s0)
        traci.vehicle.moveTo('veh1', 'highway_0', s0 + 300)
        traci.vehicle.moveTo('veh2', 'highway_1', s0 + 36)
        traci.vehicle.moveTo('veh3', 'highway_1', s0 + 54)
        traci.vehicle.moveTo('veh4', 'highway_1', s0 + 72)
        traci.vehicle.moveTo('veh5', 'highway_1', s0 + 90)
        traci.vehicle.moveTo('veh6', 'highway_1', s0 + 108)
        traci.vehicle.setSpeed('veh0', 25)
        traci.vehicle.setSpeed('veh1', 0)
        traci.vehicle.setSpeed('veh2', 15)
        traci.vehicle.setSpeed('veh3', 15)
        traci.vehicle.setSpeed('veh4', 15)
        traci.vehicle.setSpeed('veh5', 15)
        traci.vehicle.setSpeed('veh6', 15)
        if save_video:
            video_folder = "../videos/standstill___" + filepath[8:-1]
            if not os.path.isdir(video_folder):
                os.mkdir(video_folder)
                os.mkdir(video_folder + "/images")
            traci.gui.setZoom('View #0', 6850)
            traci.gui.screenshot("View #0", video_folder + "/images/" + str(0) + ".png")
        observation, reward, done, info = env.step(0)
        if env.use_gui:
            traci.vehicle.setColor('veh1', (255, 255, 255))
        for veh in env.vehicles[1:]:
            traci.vehicle.setSpeed(veh, -1)

        # Run standstill case
        action_log = []
        q_log = []
        cv_log = []
        v_log = []
        nb_safe_actions = 0
        for i in range(15):
            if save_video:
                traci.gui.screenshot("View #0", video_folder + "/images/" + str(i+1) + ".png")
            action, action_info = dqn.forward(observation)
            observation, reward, done, info = env.step(action)
            if env.use_gui:
                traci.vehicle.setColor('veh1', (255, 255, 255))
            if 'safe_action' in action_info:
                if action_info['safe_action']:
                    nb_safe_actions += 1
            action_log.append(action)
            q_log.append(action_info['mean'] if p.agent_par['ensemble'] else action_info['q_values'])
            cv_log.append(action_info['coefficient_of_variation'] if p.agent_par['ensemble'] else action_info['q_values']*0)
            v_log.append(env.speeds[0, 0])

        # Plot results
        f0 = plt.figure(0)
        ax0 = plt.gca()
        q_log = np.array(q_log)
        for action in range(0, np.shape(q_log)[1]):
            ax0.plot(q_log[:, action], label=str(action))
        ax0.legend()
        f0.show()

        f1 = plt.figure(1)
        f1.set_size_inches(7.5, 4.5)
        plt.rc('font', size=14)
        ax1 = plt.gca()
        ax1_ = ax1.twinx()
        cv_log = np.array(cv_log)
        ax1.plot(cv_log[:, 0], label='$\dot{v}_{x,0} = 0$')
        ax1.plot(cv_log[:, 1], label='$\dot{v}_{x,0} = 1$')
        ax1.plot(cv_log[:, 2], label='$\dot{v}_{x,0} = -1$')
        ax1.plot(cv_log[:, 3], label='$\dot{v}_{x,0} = -4$')
        ax1.legend(loc='upper left')
        ax1.axis([0, 10, 0, 0.1])
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Uncertainty, $c_\mathrm{v}$")
        ax1.axhline(y=0.02, color='k', linestyle='--')
        ax1.text(0.5, 0.025, '$c_\mathrm{v}^\mathrm{safe}$', rotation=0)
        ax1.set_yticks([0, 0.05, 0.1])
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)

        h1_ = ax1_.plot(v_log, color='tab:gray', linestyle='-.', label='$v_{x,0}$')
        ax1_.axis([0, 10, -0.05, 25.05])
        ax1_.set_ylabel("Speed (m/s)")
        ax1_.set_yticks([0, 10, 20])
        ax1_.spines['left'].set_visible(False)
        ax1_.spines['top'].set_visible(False)
        ax1_.legend(loc='upper right')

        plt.tight_layout()
        f1.show()

        f2 = plt.figure(2)
        ax2 = plt.gca()
        ax2.plot(v_log)
        f2.show()

else:
    raise Exception('Case not defined.')
