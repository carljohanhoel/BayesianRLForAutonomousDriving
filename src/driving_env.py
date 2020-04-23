import os
import sys
import numpy as np
import copy
import warnings
warnings.simplefilter('always', UserWarning)

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
from sumolib import checkBinary
import traci

from road import Road

# Sumo subscription constants
POSITION = 66
LONG_SPEED = 64
LAT_SPEED = 50
LONG_ACC = 114


class Highway(object):
    """
    This class creates a gym-like highway driving environment.

    The parameters of the environment are defined in parameters_simulation.py.
    The environment is built in a gym-like structure, with the methods 'reset' and 'step'

    Args:
        sim_params: Parameters that describe the simulation setup, the action space, and the reward function
        road_params: Parameters that describe the road geometry, rules, and properties of different vehicles
        use_gui (bool): Run simulation w/wo GUI
        start_time (str): Optional label
    """
    def __init__(self, sim_params, road_params, use_gui=True, start_time=''):
        self.step_ = 0
        self.max_steps = sim_params['max_steps']
        self.max_dist = sim_params['max_dist']
        self.init_steps = sim_params['init_steps']
        self.nb_vehicles = sim_params['nb_vehicles']
        self.vehicles = None
        self.safety_check = sim_params['safety_check']

        self.road = Road(road_params, start_time=start_time)
        self.road.create_road()

        self.nb_lanes = self.road.road_params['nb_lanes']
        self.lane_width = self.road.road_params['lane_width']
        self.lane_change_duration = self.road.road_params['lane_change_duration']
        self.positions = np.zeros([self.nb_vehicles, 2])
        self.speeds = np.zeros([self.nb_vehicles, 2])
        self.accs = np.zeros([self.nb_vehicles, 1])
        self.init_ego_position = 0.
        self.ego_id = 'veh' + str(0).zfill(int(np.ceil(np.log10(self.nb_vehicles))))   # Add leading zeros to number

        self.max_speed = self.road.road_params['speed_range'][1]
        self.min_speed = self.road.road_params['speed_range'][0]
        self.max_ego_speed = self.road.road_params['vehicles'][0]['maxSpeed']
        self.sensor_range = sim_params['sensor_range']
        self.sensor_nb_vehicles = sim_params['sensor_nb_vehicles']
        self.action_interp = sim_params['action_interp']
        self.collision_penalty = sim_params['collision_penalty']
        self.near_collision_penalty = sim_params['near_collision_penalty']
        self.lane_change_penalty = sim_params['lane_change_penalty']
        self.emergency_braking_penalty = sim_params['emergency_braking_penalty']
        self.emergency_braking_limit = sim_params['emergency_braking_limit']
        self.emergency_braking_dist = sim_params['emergency_braking_dist']

        self.nb_ego_states = 4
        self.nb_states_per_vehicle = 4

        self.state_t0 = None
        self.state_t1 = None

        self.use_gui = use_gui
        if self.use_gui:
            sumo_binary = checkBinary('sumo-gui')
        else:
            sumo_binary = checkBinary('sumo')

        # this is the normal way of using traci. sumo is started as a
        # subprocess and then the python script connects and runs
        if sim_params['remove_sumo_warnings']:
            traci.start([sumo_binary, "-c", self.road.road_path + self.road.name + ".sumocfg", "--start",
                         "--no-warnings"])
        else:
            traci.start([sumo_binary, "-c", self.road.road_path + self.road.name + ".sumocfg", "--start"])

    def reset(self, sumo_ctrl=False):
        """
        Resets the highway driving environment to a new random initial state.

        The ego vehicle starts in a random lane. A number of surrounding vehicles are added to random positions.
        Vehicles in front of the ego vehicle are initalized with a lower speed than the ego vehicle, and vehicles behind
         the ego vehicle are initalized with a faster speed. If two vehicles vehicles are initalized too close
         to each other, one of them is moved.

        Args:
            sumo_ctrl (bool): For testing purposes, setting this True lets SUMO control the ego vehicle.

        Returns:
            observation (ndarray): The observation of the traffic situation, according to the sensor model.
        """
        # Remove all vehicles
        for veh in traci.vehicle.getIDList():
            traci.vehicle.remove(veh)
        traci.simulationStep()

        # Add vehicles
        for i in range(self.nb_vehicles):
            veh_id = 'veh' + str(i).zfill(int(np.ceil(np.log10(self.nb_vehicles))))   # Add leading zeros to number
            lane = i % self.nb_lanes
            traci.vehicle.add(veh_id, 'route0', typeID='truck' if i == 0 else 'car', depart=None, departLane=lane,
                              departPos='base', departSpeed=self.road.road_params['vehicles'][0]['maxSpeed'],
                              arrivalLane='current', arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='',
                              line='', personCapacity=0, personNumber=0)
            if (i + 1) % self.nb_lanes == 0:  # When all lanes are filled
                traci.simulationStep()  # Deploy vehicles
                for veh in traci.vehicle.getIDList():  # Move all vehicles 30 m forwards
                    traci.vehicle.moveTo(veh, traci.vehicle.getLaneID(veh), traci.vehicle.getLanePosition(veh) + 30.)
        traci.simulationStep()
        assert (len(traci.vehicle.getIDList()) == self.nb_vehicles)
        self.vehicles = traci.vehicle.getIDList()

        # Randomly distribute vehicles
        start_lane_name = self.road.road_params['edges'][1]
        start_lane_length = traci.lane.getLength(start_lane_name + '_0')
        x_pos = np.random.uniform(0., start_lane_length, self.nb_vehicles)
        x_pos[0] = start_lane_length/2
        lane = np.random.randint(0, self.nb_lanes, self.nb_vehicles)

        # Move vehicles that start too close to each other, to avoid collisions
        total_nb_moves = 0
        move_vehicle = np.zeros([self.nb_vehicles], dtype=bool)
        for l in range(self.nb_lanes):
            idx1, idx2, diff = find_min_diff(x_pos[lane == l])
            if diff < self.road.road_params['min_start_dist']:
                move_vehicle[np.argwhere(lane == l)[idx2]] = True
        while move_vehicle.any():
            assert not move_vehicle[0]
            nb_moves = len(x_pos[move_vehicle])
            total_nb_moves += nb_moves
            x_pos[move_vehicle] = np.random.uniform(0., start_lane_length, nb_moves)
            lane[move_vehicle] = np.random.randint(0, self.nb_lanes, nb_moves)
            move_vehicle = np.zeros([self.nb_vehicles], dtype=bool)
            for l in range(self.nb_lanes):
                idx1, idx2, diff = find_min_diff(x_pos[lane == l])
                if diff < self.road.road_params['min_start_dist']:
                    move_vehicle[np.argwhere(lane == l)[idx2]] = True
        if total_nb_moves > 20:
            warnings.warn("Too crowded road, hard to find initial state. " + str(total_nb_moves) + " moves required.")

        init_speed = np.zeros(self.nb_vehicles)
        leaders = x_pos > x_pos[0]
        followers = x_pos < x_pos[0]
        init_speed[0] = self.road.road_params['vehicles'][0]['maxSpeed']
        init_speed[leaders] = np.random.uniform(self.road.road_params['speed_range'][0],
                                                init_speed[0], np.sum(leaders))
        init_speed[followers] = np.random.uniform(init_speed[0], self.road.road_params['speed_range'][1],
                                                  np.sum(followers))

        for i, veh in enumerate(self.vehicles):
            if i == 0 and not sumo_ctrl:
                traci.vehicle.moveTo(veh, start_lane_name + '_' + str(lane[i]), x_pos[i])
                continue
            traci.vehicle.moveTo(veh, start_lane_name + '_' + str(lane[i]), x_pos[i])
            # SpeedMode 1 means that the speed will be reduced to a safe speed, but maximum acceleration is not
            # considered when using setSpeed
            traci.vehicle.setSpeedMode(veh, 1)
            traci.vehicle.setSpeed(veh, init_speed[i])   # Set current speed
            traci.vehicle.setMaxSpeed(veh, init_speed[i])   # Set speed of "cruise controller"

        # Init variable subscriptions from sumo
        for veh in self.vehicles:
            traci.vehicle.subscribe(veh, [POSITION, LONG_SPEED, LAT_SPEED, LONG_ACC])  # position, speed

        for i in range(self.init_steps):
            traci.simulationStep()

        # Return speed control to sumo, starting from initial random speed
        for i, veh in enumerate(self.vehicles[1:]):
            # print(traci.vehicle.getSpeed(veh) - init_speed[i+1])
            traci.vehicle.setSpeed(veh, -1)

        # Turn off all internal lane changes and all safety checks for ego vehicle
        if not sumo_ctrl:
            if not self.safety_check:
                traci.vehicle.setSpeedMode(self.ego_id, 0)
                traci.vehicle.setLaneChangeMode(self.ego_id, 0)
        else:
            traci.vehicle.setSpeed(self.ego_id, -1)

        if self.use_gui:
            traci.gui.trackVehicle('View #0', self.ego_id)

        self.step_ = 0
        self.init_ego_position = traci.vehicle.getPosition(self.ego_id)[0]

        for i, veh in enumerate(self.vehicles):
            out = traci.vehicle.getSubscriptionResults(veh)
            self.positions[i, :] = np.array(out[POSITION]) + np.array([0, self.lane_width * self.nb_lanes -
                                                                       self.lane_width/2])
            self.speeds[i, 0] = out[LONG_SPEED]
            self.speeds[i, 1] = out[LAT_SPEED] \
                if not np.isclose((out[POSITION][1] - self.lane_width / 2) % self.lane_width, 0.)\
                else 0.0  # Complicated due to bug in sumo output lateral speed
            self.accs[i] = out[LONG_ACC]
            if self.use_gui:
                if i == 0:
                    traci.vehicle.setColor(veh, (0, 200, 0))
                else:
                    speed_factor = (self.speeds[i, 0] - self.min_speed)/(self.max_speed - self.min_speed)
                    speed_factor = np.max([speed_factor, 0])
                    speed_factor = np.min([speed_factor, 1])
                    traci.vehicle.setColor(veh, (255, int(255*(1-speed_factor)), 0))
        self.state_t1 = [self.positions, self.speeds, False]
        observation = self.sensor_model(self.state_t1)

        if self.use_gui:
            self.print_info_in_gui(info='Start')

        return observation

    def step(self, action, action_info=None, sumo_ctrl=False):
        """
        Transition the environment to the next state with the specified action.

        Args:
            action (int): Specified action, which is then translated to a longitudinal and lateral action.
            action_info (dict): Only used to display information in the GUI.
            sumo_ctrl (bool): For testing purposes, setting this True lets SUMO control the ego vehicle.

        Returns:
            tuple, containing:
                observation (ndarray): Observation of the environment, given by the sensor model.
                reward (float): Reward of the current time step.
                done (bool): True if terminal state is reached, otherwise False
                info (list): List of information on what caused the terminal condition.

        """
        self.state_t0 = np.copy(self.state_t1)

        long_action, lat_action = self.action_interp[action]
        if self.speeds[0, 0] + long_action > self.max_ego_speed:   # Limit maximum speed to max of vehicle
            long_action = self.max_ego_speed - self.speeds[0, 0]
        elif self.speeds[0, 0] + long_action < 0.:
            long_action = 0. - self.speeds[0, 0]

        self.step_ += 1
        if not sumo_ctrl:
            traci.vehicle.setSpeed(self.ego_id, self.speeds[0, 0] + long_action)
            traci.vehicle.changeLaneRelative(self.ego_id, lat_action, 1e15)
        traci.simulationStep()

        # Number of digits in vehicle name. Can't just enumerate index because vehicles can be removed in the event of
        # simultaneous change to center lane.
        nb_digits = int(np.floor(np.log10(self.nb_vehicles))) + 1
        for veh in self.vehicles:
            i = int(veh[-nb_digits:])   # See comment above
            out = traci.vehicle.getSubscriptionResults(veh)
            self.positions[i, :] = np.array(out[POSITION]) + \
                np.array([0, self.lane_width * self.nb_lanes - self.lane_width/2])
            self.speeds[i, 0] = out[LONG_SPEED]
            self.speeds[i, 1] = out[LAT_SPEED] \
                if not np.isclose((out[POSITION][1] - self.lane_width / 2) % self.lane_width, 0.) \
                else 0.0  # Complicated due to bug in sumo output lateral speed
            self.accs[i] = out[LONG_ACC]
            if self.use_gui and not i == 0:
                if i == 0:
                    traci.vehicle.setColor(veh, (0, 200, 0))
                else:
                    speed_factor = (self.speeds[i, 0] - self.min_speed)/(self.max_speed - self.min_speed)
                    speed_factor = np.max([speed_factor, 0])
                    speed_factor = np.min([speed_factor, 1])
                    traci.vehicle.setColor(veh, (255, int(255*(1-speed_factor)), 0))

        # This conditions takes care of the case when both the ego vehicle and another vehicle starts to change to the
        # center lane at the same time. The other vehicle is then simply deleted cause of implementation issues.
        # Not an elegant solution, but it happen so seldom that it doesn't matter.
        if np.isclose(self.positions[0, 1], self.lane_width / self.lane_change_duration) and \
           lat_action > 0:
            changing_to_center_vehicles = np.logical_and(np.isclose(self.positions[:, 1],
                                                                    2*self.lane_width - self.lane_width /
                                                                    self.lane_change_duration),
                                                         self.speeds[:, 1] < 0.)
            changing_veh_close = np.abs(self.positions[changing_to_center_vehicles, 0] -
                                        self.positions[0, 0]) < 15.
            if np.any(changing_veh_close):
                for idx in np.argwhere(changing_to_center_vehicles)[np.argwhere(changing_veh_close)]:
                    veh_id = 'veh' + str(np.ndarray.item(idx)).zfill(int(np.ceil(np.log10(self.nb_vehicles))))
                    traci.vehicle.remove(veh_id)
                    self.vehicles = traci.vehicle.getIDList()
                    self.positions[idx, 1] = 2*self.lane_width
                warnings.warn('Collision due to simultaneous change to middle lane prevented.')
        elif np.isclose(self.positions[0, 1], 2*self.lane_width - self.lane_width / self.lane_change_duration) and \
                lat_action < 0:
            changing_to_center_vehicles = np.logical_and(np.isclose(self.positions[:, 1],
                                                                    self.lane_width / self.lane_change_duration),
                                                         self.speeds[:, 1] > 0.)
            changing_veh_close = np.abs(self.positions[changing_to_center_vehicles, 0] -
                                        self.positions[0, 0]) < 15.
            if np.any(changing_veh_close):
                for idx in np.argwhere(changing_to_center_vehicles)[np.argwhere(changing_veh_close)]:
                    veh_id = 'veh' + str(np.ndarray.item(idx)).zfill(int(np.ceil(np.log10(self.nb_vehicles))))
                    traci.vehicle.remove(veh_id)
                    self.vehicles = traci.vehicle.getIDList()
                    self.positions[idx, 1] = 0.
                warnings.warn('Collision due to simultaneous change to middle lane prevented.')

        collision = traci.simulation.getCollidingVehiclesNumber() > 0
        info = []
        ego_collision = False
        ego_near_collision = False
        done = False
        if collision:
            colliding_ids = traci.simulation.getCollidingVehiclesIDList()
            colliding_positions = [traci.vehicle.getPosition(veh) for veh in colliding_ids]

            # If "collision" because violating minGap distance. Don't consider this as a collision,
            # but a near collision and give negative reward.
            if self.ego_id in colliding_ids:
                long_dist = colliding_positions[1][0] - colliding_positions[0][0]
                if colliding_ids[0] == 'veh00':
                    other_veh_idx = 1
                else:
                    other_veh_idx = 0
                if other_veh_idx == 0:
                    long_dist = -long_dist
                if long_dist > 0:   # Only consider collisions when the other vehicle is in front of the ego vehicle
                    front_veh_length = traci.vehicle.getLength(colliding_ids[other_veh_idx])
                    if long_dist - front_veh_length > 0:
                        collision = False
                        ego_near_collision = True
        # Second if statement because a situation that is considered a collision by SUMO can be reclassified to a
        # near collision
        if collision:
            info.append(colliding_ids)
            info.append(colliding_positions)
            info.append([traci.vehicle.getSpeed(veh) for veh in info[0]])
            if self.step_ == 0:
                warnings.warn('Collision during reset phase. This should not happen.')
                print(info)
            else:
                if self.ego_id in info[0]:
                    ego_collision = True
                    done = True
                else:
                    warnings.warn('Collision not involving ego vehicle. This should normally not happen.')
                    print(self.step_, info)
                # assert self.ego_id in info[0]   # If not, there has been a collision between two other vehicles

        outside_road = False
        if not self.safety_check:
            if np.isclose(self.positions[0, 1], 0) and lat_action < 0:
                done = True
                outside_road = True
                info.append('Outside of road')
            elif np.isclose(self.positions[0, 1], (self.nb_lanes - 1) * self.lane_width) and lat_action > 0:
                done = True
                outside_road = True
                info.append('Outside of road')

        if self.step_ == self.max_steps:
            done = True
            info.append('Max steps')

        if self.positions[0, 0] - self.init_ego_position >= self.max_dist:
            done = True
            info.append('Max dist')

        self.state_t1 = copy.deepcopy([self.positions, self.speeds, done])
        observation = self.sensor_model(self.state_t1)
        reward = self.reward_model(s0=self.state_t0, s1=self.state_t1, a=[long_action, lat_action], term=done,
                                   accs=self.accs, ego_collision=ego_collision, ego_near_collision=ego_near_collision,
                                   outside_road=outside_road)

        if self.use_gui:
            self.print_info_in_gui(reward=reward, action=[long_action, lat_action], info=info, action_info=action_info)

        return observation, reward, done, info

    def reward_model(self, s0, s1, a, term, accs, ego_collision=False, ego_near_collision=False, outside_road=False):
        """
        Reward model of the highway environment.

        Args:
            s0 (ndarray): Old state, currently not used
            s1 (list): New state
            a (list): Longitudinal and lateral action.
            term (bool): True if terminal state, otherwise false, currently not used
            accs (ndarray): Acceleration vehicles
            ego_collision (bool): True if ego vehicle collides.
            ego_near_collision (bool): True if ego vehicle is close to a collision.
            outside_road (bool): True if ego vehicle drives off the road.

        Returns:
            reward (float): Reward for the current environment step.
        """
        reward = s1[1][0, 0] / self.max_ego_speed   # ego speed
        if np.abs(a[1]) > 0:
            reward -= self.lane_change_penalty
        if ego_collision or outside_road:
            reward -= self.collision_penalty
        if ego_near_collision:
            reward -= self.near_collision_penalty
        if (accs < self.emergency_braking_limit - 0.1).any():   # Small error margin added
            braking_veh_idx = np.where(accs < self.emergency_braking_limit - 0.1)[0][0]
            if 0 < s1[0][0, 0] - s1[0][braking_veh_idx, 0] < self.emergency_braking_dist:
                reward -= self.emergency_braking_penalty
                # warnings.warn('Emergency brake')
        return reward

    def sensor_model(self, state):
        """
        Sensor model of the ego vehicle.

        Creates an observation vector from the current state of the environment. All observations are normalized.
        Only surrounding vehicles within the sensor range are included.

        Args:
            state (list): Current state of the environment.

        Returns:
            observation( (ndarray): Current observation of the highway environment.
        """
        if self.road.road_params['oncoming_traffic']:   # Special hack to make sensor output consistent in oncoming case
            state[0][:, 1] -= 3.2
            state[1][3] = -state[1][3]
        vehicles_in_range = np.abs(state[0][1:, 0] - state[0][0, 0]) <= self.sensor_range
        if np.sum(vehicles_in_range) > self.sensor_nb_vehicles:
            warnings.warn('More vehicles within range than sensor can represent')
        observation = np.zeros(self.nb_ego_states + self.nb_states_per_vehicle * self.sensor_nb_vehicles)
        observation[0] = 2 * state[0][0, 1] / ((self.nb_lanes - 1) * self.lane_width) - 1   # lat pos
        observation[1] = 2 * state[1][0, 0] / self.max_ego_speed - 1   # long speed
        observation[2] = np.sign(state[1][0, 1])   # Lat speed
        observation[3] = float(state[2])   # term
        assert(self.nb_ego_states == 4)
        idx = 0
        for i, in_range in enumerate(vehicles_in_range):
            if not in_range:
                continue
            observation[4 + idx*4] = (state[0][i + 1, 0] - state[0][0, 0]) / self.sensor_range
            observation[5 + idx*4] = (state[0][i + 1, 1] - state[0][0, 1]) / ((self.nb_lanes - 1) * self.lane_width)
            observation[6 + idx*4] = (state[1][i + 1, 0] - state[1][0, 0]) / (self.max_speed - self.min_speed)
            observation[7 + idx*4] = (state[1][i + 1, 1] - state[1][0, 1]) / \
                                     (self.lane_width / self.lane_change_duration)
            idx += 1
            if idx >= self.sensor_nb_vehicles:
                break
        for i in range(idx, self.sensor_nb_vehicles):
            observation[4 + idx * 4] = -1
            observation[5 + idx * 4] = 0
            observation[6 + idx * 4] = 0
            observation[7 + idx * 4] = 0
            idx += 1
        assert(self.nb_states_per_vehicle == 4)
        return observation

    def print_info_in_gui(self, reward=None, action=None, info=None, action_info=None):
        """
        Prints information in the GUI.
        """
        polygons = traci.polygon.getIDList()
        for polygon in polygons:
            traci.polygon.remove(polygon)
        dy = 10
        traci.polygon.add('Position: {0:.1f}, {1:.1f}'.format(self.positions[0, 0] - self.init_ego_position,
                                                              self.positions[0, 1]),
                          [self.positions[0] + self.road.road_params['info_pos'], self.positions[0] +
                           self.road.road_params['info_pos'] + [1, 0]], [0, 0, 0, 0])
        traci.polygon.add('Speed: {0:.1f}, {1:.1f}'.format(*self.speeds[0, :]),
                          [self.positions[0] + self.road.road_params['info_pos'], self.positions[0] +
                           self.road.road_params['info_pos'] + [1, -dy]], [0, 0, 0, 0])
        traci.polygon.add('Action previous step: ' + str(action),
                          [self.positions[0] + self.road.road_params['info_pos'], self.positions[0] +
                           self.road.road_params['info_pos'] + [1, -2*dy]], [0, 0, 0, 0])
        traci.polygon.add('Reward: ' + str(reward),
                          [self.positions[0] + self.road.road_params['info_pos'], self.positions[0] +
                           self.road.road_params['info_pos'] + [1, -3 * dy]], [0, 0, 0, 0])
        traci.polygon.add(str(info),
                          [self.positions[0] + self.road.road_params['info_pos'], self.positions[0] +
                           self.road.road_params['info_pos'] + [1, -4*dy]], [0, 0, 0, 0])
        traci.polygon.add('Step: ' + str(self.step_),
                          [self.positions[0] + self.road.road_params['info_pos'], self.positions[0] +
                           self.road.road_params['info_pos'] + [1, -5 * dy]], [0, 0, 0, 0])
        if action_info is not None:
            if 'q_values' in action_info:
                traci.polygon.add('  | '.join(['{:6.1f}'.format(element) for element in action_info['q_values']]),
                                  [self.positions[0] + self.road.road_params['action_info_pos'], self.positions[0] +
                                   self.road.road_params['action_info_pos'] + [1, 0]], [0, 0, 0, 0])
            if 'q_values_all_nets' in action_info:
                for i, row in enumerate(action_info['q_values_all_nets']):
                    traci.polygon.add('  | '.join(['{:6.1f}'.format(element) for element in row]),
                                      [self.positions[0] + self.road.road_params['action_info_pos'], self.positions[0] +
                                       self.road.road_params['action_info_pos'] + [1, -i*dy]], [0, 0, 0, 0])
            if 'mean' in action_info:
                traci.polygon.add('  | '.join(['{:6.1f}'.format(element) for element in action_info['mean']]),
                                  [self.positions[0] + self.road.road_params['action_info_pos'], self.positions[0] +
                                   self.road.road_params['action_info_pos'] + [1, -10.5*dy]], [0, 0, 0, 0])
            if 'coefficient_of_variation' in action_info:
                traci.polygon.add('  | '.join(['{:5.3f}'.format(element) for element in
                                               action_info['coefficient_of_variation']]),
                                  [self.positions[0] + self.road.road_params['action_info_pos'], self.positions[0] +
                                   self.road.road_params['action_info_pos'] + [1, -11.5*dy]], [0, 0, 0, 0])

    @property
    def nb_actions(self):
        return len(self.action_interp)

    @property
    def nb_observations(self):
        return self.nb_ego_states + self.nb_states_per_vehicle * self.sensor_nb_vehicles


def find_min_diff(array):
    """
    Returns minimum difference between any pair in array, and the corresponding indices

    Args:
        array (ndarray):

    Returns:
        Indexes of elements with minimum distance, and the minimum distance

    """
    length_array = len(array)
    diff = sys.maxsize
    idx1 = -1
    idx2 = -1
    for i in range(length_array-1):
        for j in range(i+1, length_array):
            if np.abs(array[i] - array[j]) < diff:
                diff = np.abs(array[i] - array[j])
                idx1 = i
                idx2 = j
    if length_array > 2:
        assert idx1 >= 0
        assert idx2 >= 0
    return idx1, idx2, diff
