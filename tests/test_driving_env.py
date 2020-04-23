import unittest
import numpy as np
import sys
sys.path.append('../src')
import parameters_simulation as p
from driving_env import Highway
import traci
from copy import deepcopy


class Tester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)

    def test_init(self):
        self.env = Highway(sim_params=p.sim_params, road_params=p.road_params, use_gui=False)
        self.env.reset()
        self.assertEqual(self.env.nb_vehicles, len(traci.vehicle.getIDList()))
        traci.close()

    # def test_subscription(self):
    #     self.env = Highway(sim_params=p.sim_params, road_params=p.road_params, use_gui=False)
    #     obs = self.env.reset()
    #     # To do

    def test_step(self):
        self.env = Highway(sim_params=p.sim_params, road_params=p.road_params, use_gui=False)
        self.env.reset()
        try:
            action = 0
            self.env.step(action)
            self.assertEqual(self.env.nb_vehicles, len(traci.vehicle.getIDList()))
        finally:
            traci.close()

    def test_sensor_model(self):
        self.env = Highway(sim_params=p.sim_params, road_params=p.road_params, use_gui=False)
        self.env.reset()
        try:
            self.env.nb_lanes = 3
            lane_width = self.env.lane_width
            lane_change_speed = self.env.lane_width / self.env.lane_change_duration
            positions = np.array([[34.2, lane_width], [56.3, 2 * lane_width], [3.4, lane_width],
                                 [165.6, 0], [63.2, 1.5 * lane_width]])
            speeds = np.array([[25.0, 0], [23.4, 0.], [10.1, 0.], [33.5, 0], [15.8, lane_change_speed]])
            done = False
            state = [positions, speeds, done]
            observation = self.env.sensor_model(state)
            self.assertEqual(observation[0], 0)
            self.assertEqual(observation[1], 2 * 25 / self.env.max_ego_speed - 1)
            self.assertEqual(observation[2], 0)
            self.assertEqual(observation[3], 0.0)
            if self.env.sensor_range < 165:
                self.assertAlmostEqual(observation[12], (63.2 - 34.2) / self.env.sensor_range)
                self.assertAlmostEqual(observation[13],
                                       0.5 * lane_width / ((self.env.nb_lanes - 1) * self.env.lane_width))
                self.assertAlmostEqual(observation[14], (15.8 - 25.0) / (self.env.max_speed - self.env.min_speed))
                self.assertAlmostEqual(observation[15], 1)
                self.assertListEqual(list(observation[16:20]), [-1, 0, 0, 0])
            else:
                self.assertAlmostEqual(observation[16], (63.2 - 34.2) / self.env.sensor_range)
                self.assertAlmostEqual(observation[17],
                                       0.5 * lane_width / ((self.env.nb_lanes - 1) * self.env.lane_width))
                self.assertAlmostEqual(observation[18], (15.8 - 25.0) / (self.env.max_speed - self.env.min_speed))
                self.assertAlmostEqual(observation[19], 1)
                self.assertListEqual(list(observation[-4:]), [-1, 0, 0, 0])
            self.assertEqual(len(observation), 4 + self.env.sensor_nb_vehicles * 4)

            positions = np.random.rand(self.env.sensor_nb_vehicles + 5, 2)
            speeds = np.random.rand(self.env.sensor_nb_vehicles + 5, 2)
            state = [positions, speeds, done]
            observation = self.env.sensor_model(state)
            self.assertEqual(len(observation), 4 + self.env.sensor_nb_vehicles * 4)

        finally:
            traci.close()

    def test_reward_model(self):
        self.env = Highway(sim_params=p.sim_params, road_params=p.road_params, use_gui=False)
        self.env.reset()
        try:
            positions = np.array([[0., 0.], [-50, 0]])
            speeds = np.array([[25., 0.], [16., 0.]])
            accs = np.zeros(2)
            action = [0, 0]
            done = False
            state_t0 = [positions, speeds, done]
            reward = self.env.reward_model(state_t0, state_t0, action, done, accs)
            self.assertEqual(reward, 25/self.env.max_ego_speed)
            reward = self.env.reward_model(state_t0, state_t0, action, done, accs, ego_collision=True)
            self.assertEqual(reward, 25/self.env.max_ego_speed - p.sim_params['collision_penalty'])
            reward = self.env.reward_model(state_t0, state_t0, action, done, accs, ego_near_collision=True)
            self.assertEqual(reward, 25 / self.env.max_ego_speed - p.sim_params['near_collision_penalty'])
            reward = self.env.reward_model(state_t0, state_t0, action, done, accs, outside_road=True)
            self.assertEqual(reward, 25/self.env.max_ego_speed - p.sim_params['collision_penalty'])
            reward = self.env.reward_model(state_t0, state_t0, action, True, accs)   # End of episode, no collision
            self.assertEqual(reward, 25 / self.env.max_ego_speed)
            action = [0, 1]
            reward = self.env.reward_model(state_t0, state_t0, action, done, accs)
            self.assertEqual(reward, 25 / self.env.max_ego_speed - p.sim_params['lane_change_penalty'])
            # Emergency braking, but too far away
            accs = np.array([-3., -7.])
            action = [0, -1]
            reward = self.env.reward_model(state_t0, state_t0, action, done, accs)
            self.assertEqual(reward, 25 / self.env.max_ego_speed - p.sim_params['lane_change_penalty'])
            # Emergency brake
            positions[1][0] = -20.
            reward = self.env.reward_model(state_t0, state_t0, action, done, accs)
            self.assertEqual(reward, 25 / self.env.max_ego_speed - p.sim_params['emergency_braking_penalty']
                             - p.sim_params['lane_change_penalty'])

        finally:
            traci.close()

    def test_gym_interface(self):
        self.env = Highway(sim_params=p.sim_params, road_params=p.road_params, use_gui=False)
        self.env.reset()
        try:
            self.assertEqual(self.env.nb_actions, len(p.sim_params['action_interp']))
            self.assertEqual(self.env.nb_observations, 4 + 4*p.sim_params['sensor_nb_vehicles'])
        finally:
            traci.close()

    def test_action_range(self):
        tmp = p.sim_params['nb_vehicles']
        p.sim_params['nb_vehicles'] = 1
        self.env = Highway(sim_params=p.sim_params, road_params=p.road_params, use_gui=False)
        self.env.reset()
        p.sim_params['nb_vehicles'] = tmp   # Reset value for other tests
        try:
            # test limit ego speed downwards
            action = 3   # action brake hard
            self.assertEqual(-4, p.sim_params['action_interp'][action][0])
            for i in range(20):
                observation, reward, done, info = self.env.step(action)
                self.assertGreaterEqual(observation[1], -1.)   # Ego speed larger than 0
            # test limit ego speed upwards
            action = 1   # action accelerate
            self.assertEqual(1, p.sim_params['action_interp'][action][0])
            for i in range(30):
                observation, reward, done, info = self.env.step(action)
                self.assertLessEqual(observation[1], 1.)   # Ego speed larger than 0
            self.assertEqual(observation[1], 1.)
        finally:
            traci.close()

    def test_prevent_simultaneous_lane_change_to_same_lane(self):
        tmp = p.sim_params['nb_vehicles']
        p.sim_params['nb_vehicles'] = 3
        self.env = Highway(sim_params=p.sim_params, road_params=p.road_params, use_gui=False)
        self.env.reset()
        try:
            for _ in range(3):
                # Make sure that the vehicles are not affected by previous state
                np.random.seed(57)
                self.env.reset()
                s0 = 1000.
                traci.vehicle.moveTo('veh0', 'highway_2', s0 - 50)
                traci.vehicle.moveTo('veh1', 'highway_1', s0 - 50)
                traci.vehicle.moveTo('veh2', 'highway_0', s0 - 50)
                self.env.step(0)
                traci.vehicle.setSpeed('veh0', 25)
                traci.vehicle.setSpeed('veh1', 30)
                traci.vehicle.setSpeed('veh2', 20)
                traci.vehicle.setMaxSpeed('veh0', 25)
                traci.vehicle.setMaxSpeed('veh1', 30)
                traci.vehicle.setMaxSpeed('veh2', 20)
                self.env.step(0)
                self.env.step(0)

                # This setup causes a collision without any preventive measures
                traci.vehicle.moveTo('veh0', 'highway_2', s0 + 63.8)
                traci.vehicle.moveTo('veh1', 'highway_0', s0)
                traci.vehicle.moveTo('veh2', 'highway_0', s0 + 200)
                traci.vehicle.setSpeed('veh0', 25)
                traci.vehicle.setSpeed('veh1', 30)
                traci.vehicle.setSpeed('veh2', 20)
                traci.vehicle.setMaxSpeed('veh0', 25)
                traci.vehicle.setMaxSpeed('veh1', 30)
                traci.vehicle.setMaxSpeed('veh2', 20)
                for i in range(25):
                    if i == 13:
                        action = 7
                    else:
                        action = 0
                    observation, reward, done, info = self.env.step(action)
                    self.assertFalse(done)

                # Same but mirrored
                np.random.seed(57)
                self.env.reset()
                s0 = 1000.
                traci.vehicle.moveTo('veh0', 'highway_0', s0 - 50)
                traci.vehicle.moveTo('veh1', 'highway_1', s0 - 50)
                traci.vehicle.moveTo('veh2', 'highway_2', s0 - 50)
                self.env.step(0)
                traci.vehicle.setSpeed('veh0', 25)
                traci.vehicle.setSpeed('veh1', 30)
                traci.vehicle.setSpeed('veh2', 20)
                traci.vehicle.setMaxSpeed('veh0', 25)
                traci.vehicle.setMaxSpeed('veh1', 30)
                traci.vehicle.setMaxSpeed('veh2', 20)
                self.env.step(0)
                self.env.step(0)
                traci.vehicle.moveTo('veh0', 'highway_0', s0 + 63.8)
                traci.vehicle.moveTo('veh1', 'highway_2', s0)
                traci.vehicle.moveTo('veh2', 'highway_2', s0 + 200)
                traci.vehicle.setSpeed('veh0', 25)
                traci.vehicle.setSpeed('veh1', 30)
                traci.vehicle.setSpeed('veh2', 20)
                traci.vehicle.setMaxSpeed('veh0', 25)
                traci.vehicle.setMaxSpeed('veh1', 30)
                traci.vehicle.setMaxSpeed('veh2', 20)
                for i in range(25):
                    if i == 13:
                        action = 4
                    else:
                        action = 0
                    observation, reward, done, info = self.env.step(action)
                    self.assertFalse(done)

        finally:
            p.sim_params['nb_vehicles'] = tmp  # Reset value for other tests
            traci.close()

    def test_close_collision_penalty(self):
        tmp = p.sim_params['nb_vehicles']
        p.sim_params['nb_vehicles'] = 3
        self.env = Highway(sim_params=p.sim_params, road_params=p.road_params, use_gui=False)
        self.env.reset()
        try:
            for _ in range(3):
                # Make sure that the vehicles are not affected by previous state
                np.random.seed(57)
                self.env.reset()
                s0 = 1000.
                traci.vehicle.moveTo('veh0', 'highway_0', s0 - 300)
                traci.vehicle.moveTo('veh1', 'highway_1', s0 - 300)
                traci.vehicle.moveTo('veh2', 'highway_2', s0 - 300)
                self.env.step(0)
                traci.vehicle.setSpeed('veh0', 25)
                traci.vehicle.setSpeed('veh1', 25)
                traci.vehicle.setSpeed('veh2', 25)
                traci.vehicle.setMaxSpeed('veh0', 25)
                traci.vehicle.setMaxSpeed('veh1', 25)
                traci.vehicle.setMaxSpeed('veh2', 25)
                self.env.step(0)
                self.env.step(0)
                # Change into other vehicle, should cause collision
                traci.vehicle.moveTo('veh0', 'highway_0', s0)
                traci.vehicle.moveTo('veh1', 'highway_1', s0)
                traci.vehicle.moveTo('veh2', 'highway_2', s0)
                traci.vehicle.setSpeed('veh0', 25)
                traci.vehicle.setSpeed('veh1', 25)
                traci.vehicle.setSpeed('veh2', 25)
                for i in range(5):
                    if i == 1:
                        action = 5
                    else:
                        action = 0
                    observation, reward, done, info = self.env.step(action)
                    if i == 2:
                        self.assertTrue(done)

                # Make sure that the vehicles are not affected by previous state
                np.random.seed(57)
                self.env.reset()
                s0 = 1000.
                traci.vehicle.moveTo('veh0', 'highway_0', s0 - 300)
                traci.vehicle.moveTo('veh1', 'highway_1', s0 - 300)
                traci.vehicle.moveTo('veh2', 'highway_2', s0 - 300)
                self.env.step(0)
                traci.vehicle.setSpeed('veh0', 25)
                traci.vehicle.setSpeed('veh1', 20)
                traci.vehicle.setSpeed('veh2', 20)
                traci.vehicle.setMaxSpeed('veh0', 25)
                traci.vehicle.setMaxSpeed('veh1', 20)
                traci.vehicle.setMaxSpeed('veh2', 20)
                self.env.step(0)
                self.env.step(0)
                # Rear end other vehicle, should cause collision
                traci.vehicle.moveTo('veh0', 'highway_0', s0)
                traci.vehicle.moveTo('veh1', 'highway_0', s0 + 17)
                traci.vehicle.moveTo('veh2', 'highway_1', s0 + 17)
                traci.vehicle.setSpeed('veh0', 25)
                traci.vehicle.setSpeed('veh1', 20)
                traci.vehicle.setSpeed('veh2', 20)
                traci.vehicle.setMaxSpeed('veh1', 20)
                traci.vehicle.setMaxSpeed('veh2', 20)
                for i in range(5):
                    action = 0
                    observation, reward, done, info = self.env.step(action)
                    if i == 1:
                        self.assertFalse(done)   # Should be considered a near collision
                        self.assertEqual(reward, 1 - p.sim_params['near_collision_penalty'])
                    if i == 2:
                        self.assertTrue(done)   # Now a full collision
                        self.assertEqual(reward, 1 - p.sim_params['collision_penalty'])

        finally:
            p.sim_params['nb_vehicles'] = tmp  # Reset value for other tests
            traci.close()

    def test_nb_lanes(self):
        self.assertEqual(p.road_params['nb_lanes'], 3)
        # The number of lanes is assumed to be 3 in the code that takes care of simultaneous lane changes
        # to the center lane.

    def test_fast_overtake(self):
        tmp_sim_params = deepcopy(p.sim_params)
        tmp_road_params = deepcopy(p.road_params)
        p.sim_params['nb_vehicles'] = 5
        self.env = Highway(sim_params=p.sim_params, road_params=p.road_params, use_gui=False)
        self.env.reset()
        try:
            for _ in range(3):
                # Make sure that the vehicles are not affected by previous state
                np.random.seed(57)
                self.env.reset()
                s0 = 1000.
                traci.vehicle.moveTo('veh0', 'highway_0', s0 - 300)
                traci.vehicle.moveTo('veh1', 'highway_1', s0 - 300)
                traci.vehicle.moveTo('veh2', 'highway_2', s0 - 300)
                traci.vehicle.moveTo('veh3', 'highway_2', s0 - 400)
                traci.vehicle.moveTo('veh4', 'highway_2', s0 - 500)
                traci.simulationStep()
                self.env.speeds[0, 0] = 15
                for veh in self.env.vehicles:
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
                traci.simulationStep()

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
                traci.simulationStep()
                for veh in self.env.vehicles[1:]:
                    traci.vehicle.setSpeed(veh, -1)

                for i in range(8):
                    action = 4
                    observation, reward, done, info = self.env.step(action)

        finally:
            p.sim_params = tmp_sim_params  # Reset value for other tests
            p.road_params = tmp_road_params
            traci.close()

    def test_standstill(self):
        tmp_sim_params = deepcopy(p.sim_params)
        tmp_road_params = deepcopy(p.road_params)
        p.sim_params['nb_vehicles'] = 6
        self.env = Highway(sim_params=p.sim_params, road_params=p.road_params, use_gui=False)
        self.env.reset()
        try:
            for _ in range(3):
                # Make sure that the vehicles are not affected by previous state
                np.random.seed(57)
                self.env.reset()
                s0 = 1000.
                traci.vehicle.moveTo('veh0', 'highway_0', s0 - 300)
                traci.vehicle.moveTo('veh1', 'highway_1', s0 - 300)
                traci.vehicle.moveTo('veh2', 'highway_2', s0 - 300)
                traci.vehicle.moveTo('veh3', 'highway_2', s0 - 400)
                traci.vehicle.moveTo('veh4', 'highway_2', s0 - 500)
                traci.vehicle.moveTo('veh5', 'highway_2', s0 - 600)
                traci.simulationStep()
                self.env.speeds[0, 0] = 25
                for veh in self.env.vehicles:
                    traci.vehicle.setSpeedMode(veh, 0)
                    traci.vehicle.setLaneChangeMode(veh, 0)   # Turn off all lane changes
                traci.vehicle.setSpeed('veh0', 25)
                traci.vehicle.setSpeed('veh1', 0)
                traci.vehicle.setSpeed('veh2', 15)
                traci.vehicle.setSpeed('veh3', 15)
                traci.vehicle.setSpeed('veh4', 15)
                traci.vehicle.setSpeed('veh5', 15)
                traci.simulationStep()
                traci.vehicle.setMaxSpeed('veh0', 25)
                traci.vehicle.setMaxSpeed('veh1', 0.001)
                traci.vehicle.setMaxSpeed('veh2', 15)
                traci.vehicle.setMaxSpeed('veh3', 15)
                traci.vehicle.setMaxSpeed('veh4', 15)
                traci.vehicle.setMaxSpeed('veh5', 15)
                traci.simulationStep()
                traci.simulationStep()

                # Standstill case
                traci.vehicle.moveTo('veh0', 'highway_0', s0)
                traci.vehicle.moveTo('veh1', 'highway_0', s0 + 170)
                traci.vehicle.moveTo('veh2', 'highway_1', s0 + 54)
                traci.vehicle.moveTo('veh3', 'highway_1', s0 + 36)
                traci.vehicle.moveTo('veh4', 'highway_1', s0 + 18)
                traci.vehicle.moveTo('veh5', 'highway_1', s0 + 0)
                traci.vehicle.setSpeed('veh0', 25)
                traci.vehicle.setSpeed('veh1', 0)
                traci.vehicle.setSpeed('veh2', 15)
                traci.vehicle.setSpeed('veh3', 15)
                traci.vehicle.setSpeed('veh4', 15)
                traci.vehicle.setSpeed('veh5', 15)
                traci.simulationStep()
                for veh in self.env.vehicles[1:]:
                    traci.vehicle.setSpeed(veh, -1)

                for i in range(20):
                    if i < 2:
                        action = 0
                    else:
                        action = 3
                    observation, reward, done, info = self.env.step(action)
                    self.assertFalse(done)

        finally:
            p.sim_params = tmp_sim_params  # Reset value for other tests
            p.road_params = tmp_road_params
            traci.close()


if __name__ == '__main__':
    unittest.main()
