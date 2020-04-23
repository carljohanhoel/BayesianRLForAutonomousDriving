"""
Parameters of the highway driving environment.

The meaning of the different parameters are described below.
"""

import numpy as np

# Simulation parameters
sim_params = {}
sim_params['max_steps'] = 100   # Episode number of steps
sim_params['max_dist'] = 2000000   # Episode maximum length in meters
sim_params['init_steps'] = 4   # Initial steps before episode starts. At least one to get sumo subscriptions right.
sim_params['nb_vehicles'] = 25   # Number of inserted vehicles.
sim_params['remove_sumo_warnings'] = True
sim_params['safety_check'] = False   # Should be False. Otherwise, SUMO checks if agent's decisions are safe.
sim_params['sensor_range'] = 200.
sim_params['sensor_nb_vehicles'] = 25   # Maximum number of vehicles the sensor can represent.
# Action index translated to longitudinal and lateral actions. The longitudinal number corresponds to
# acceleration in m/s^2, and the lateral number corresponds to stay in lane/change left/change right.
sim_params['action_interp'] = [[0, 0], [1, 0], [-1, 0], [-4, 0], [0, 1], [1, 1], [-1, 1], [0, -1], [1, -1], [-1, -1]]

# Reward parameters
sim_params['collision_penalty'] = 10
sim_params['near_collision_penalty'] = 10
sim_params['emergency_braking_penalty'] = 10   # If ego vehicle action forces another vehicle to emergency brake.
sim_params['emergency_braking_limit'] = -4.5   # Condition for when braking is considered emergency braking.
sim_params['emergency_braking_dist'] = 25   # Only vehicles within this distance are considered when checking for emergency braking.
sim_params['lane_change_penalty'] = 1

# Vehicle type parameters
# Vehicle 0 is the ego vehicle
vehicles = []
vehicles.append({})
vehicles[0]['id'] = 'truck'
vehicles[0]['vClass'] = 'trailer'
vehicles[0]['length'] = 16.0   # default 16.5
vehicles[0]['width'] = 2.55   # default 2.55
vehicles[0]['maxSpeed'] = 25.0
vehicles[0]['speedFactor'] = 1.
vehicles[0]['speedDev'] = 0
vehicles[0]['carFollowModel'] = 'Krauss'
vehicles[0]['minGap'] = 2.5   # default 2.5. Minimum longitudinal gap. A closer distance will trigger a collision.
vehicles[0]['accel'] = 1.1   # default 1.1.
vehicles[0]['decel'] = 4.0   # default 4.0.
vehicles[0]['emergencyDecel'] = 9.0   # default 4.0
vehicles[0]['sigma'] = 0.0   # default 0.5. Driver imperfection (0 = perfect driver)
vehicles[0]['tau'] = 1.0   # default 1.0. Time headway to leading vehicle.
vehicles[0]['color'] = '1,0,0'
vehicles[0]['laneChangModel'] = 'LC2013'
vehicles[0]['lcStrategic'] = 0
vehicles[0]['lcCooperative'] = 0   # default 1.0. 0 - no cooperation
vehicles[0]['lcSpeedGain'] = 1.0   # default 1.0. Eagerness for tactical lane changes.
vehicles[0]['lcKeepRight'] = 0   # default 1.0. 0 - no incentive to move to the rightmost lane
vehicles[0]['lcOvertakeRight'] = 0   # default 0. Obsolete since overtaking on the right is allowed.
vehicles[0]['lcOpposite'] = 1.0   # default 1.0. Obsolete for freeway.
vehicles[0]['lcLookaheadLeft'] = 2.0   # default 2.0. Probably obsolete.
vehicles[0]['lcSpeedGainRight'] = 1.0   # default 0.1. 1.0 - symmetric desire to change left/right
vehicles[0]['lcAssertive'] = 1.0   # default 1.0. 1.0 - no effect
vehicles[0]['lcMaxSpeedLatFactor'] = 1.0   # default 1.0. Obsolete.
vehicles[0]['lcSigma'] = 0.0   # default 0.0. Lateral imperfection.

# Vehicle 1 is the type of the surrounding vehicles
vehicles.append({})
vehicles[1]['id'] = 'car'
vehicles[1]['vClass'] = 'passenger'
vehicles[1]['length'] = 4.8   # default 5.0. 4.8 used in previous paper.
vehicles[1]['width'] = 1.8   # default 1.8.
vehicles[1]['maxSpeed'] = 100.0   # Obsolete, since will be randomly set later
vehicles[1]['speedFactor'] = 1.   # Factor times the speed limit. Obsolete, since the speed is set.
vehicles[1]['speedDev'] = 0   # Randomness in speed factor. Obsolete, since speed is set.
vehicles[1]['carFollowModel'] = 'Krauss'
vehicles[1]['minGap'] = 2.5   # default 2.5. Minimum longitudinal gap.
vehicles[1]['accel'] = 2.6   # default 2.6
vehicles[1]['decel'] = 4.5   # default 4.6
vehicles[1]['emergencyDecel'] = 9.0   # default 9.0
vehicles[1]['sigma'] = 0.0   # default 0.5. Driver imperfection.
vehicles[1]['tau'] = 1.0   # default 1.0. Time headway to leading vehicle.
vehicles[1]['laneChangModel'] = 'LC2013'
vehicles[1]['lcStrategic'] = 0
vehicles[1]['lcCooperative'] = 0   # default 1.0. 0 - no cooperation
vehicles[1]['lcSpeedGain'] = 1.0   # default 1.0. Eagerness for tactical lane changes.
vehicles[1]['lcKeepRight'] = 0   # default 1.0. 0 - no incentive to move to the rightmost lane
vehicles[1]['lcOvertakeRight'] = 0   # default 0. Obsolete since overtaking on the right is allowed.
vehicles[1]['lcOpposite'] = 1.0   # default 1.0. Obsolete for freeway.
vehicles[1]['lcLookaheadLeft'] = 2.0   # default 2.0. Probably obsolete.
vehicles[1]['lcSpeedGainRight'] = 1.0   # default 0.1. 1.0 - symmetric desire to change left/right
vehicles[1]['lcAssertive'] = 1.0   # default 1.0. 1.0 - no effect
# vehicles[1]['lcMaxSpeedLatStanding']   # default maxSpeedLat
vehicles[1]['lcMaxSpeedLatFactor'] = 1.0   # default 1.0. Obsolete.
vehicles[1]['lcSigma'] = 0.0   # default 0.0. Lateral imperfection.

# Road parameters
road_params = {}
road_params['name'] = 'highway'
road_params['nb_lanes'] = 3
road_params['lane_width'] = 3.2   # default 3.2
road_params['max_road_speed'] = 100.   # Set very high, the actual max speed is set by the vehicle type parameters.
road_params['lane_change_duration'] = 4   # Number of time steps for a lane change
road_params['speed_range'] = np.array([15, 35])   # Speed range of surrounging vehicles.
road_params['min_start_dist'] = 30   # Minimum vehicle separation when the surrounding vehicles are added.
road_params['overtake_right'] = 'true'   # Allow overtaking on the right side.
road_params['nodes'] = np.array([[-3000., 0.], [-1000., 0.], [0., 0.], [10000., 0.], [20000., 0.]])   # Road nodes
road_params['edges'] = ['add', 'start', 'highway', 'exit']
road_params['vehicles'] = vehicles
road_params['collision_action'] = 'warn'   # 'none', 'warn' (if none, sumo totally ignores collisions)
road_params['oncoming_traffic'] = False   # Only used for allowing test case with oncoming traffic

# Terminal output
road_params['emergency_decel_warn_threshold'] = 10   # A high value disables the warnings
road_params['no_display_step'] = 'true'

# Gui settings
road_params['view_position'] = np.array([750, 0])
road_params['zoom'] = 6000
road_params['view_delay'] = 200
road_params['info_pos'] = [0, 50]
road_params['action_info_pos'] = [0, -30]
