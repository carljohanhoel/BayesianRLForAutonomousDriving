import unittest
import numpy as np
import sys
sys.path.append('../src')

from road import Road


class Tester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)

        vehicles = []
        vehicles.append({})
        vehicles[0]['id'] = 'truck'
        vehicles[0]['vClass'] = 'truck'
        vehicles[0]['length'] = 15.0
        vehicles[0]['maxSpeed'] = 90.0
        vehicles[0]['accel'] = 1.0
        vehicles[0]['decel'] = 5.0
        vehicles[0]['sigma'] = 0.0
        vehicles.append({})
        vehicles[1]['id'] = 'car'
        vehicles[1]['vClass'] = 'passenger'
        vehicles[1]['length'] = 4.0
        vehicles[1]['maxSpeed'] = 120.0
        vehicles[1]['accel'] = 1.0
        vehicles[1]['decel'] = 5.0
        vehicles[1]['sigma'] = 0.0

        road_params = {}
        road_params['name'] = 'highway_test'
        road_params['nb_lanes'] = 3
        road_params['nodes'] = np.array([[-200., 0.], [-100., 0.], [0., 0.], [1000., 0.], [10000., 0.]])
        road_params['edges'] = ['add', 'start', 'highway', 'exit']
        road_params['vehicles'] = vehicles
        road_params['lane_change_duration'] = 4
        road_params['max_road_speed'] = 35
        road_params['overtake_right'] = True
        road_params['lane_width'] = 3.2
        road_params['emergency_decel_warn_threshold'] = 10

        road_params['collision_action'] = 'warn'
        road_params['no_display_step'] = 'true'
        road_params['oncoming_traffic'] = False

        road_params['view_position'] = np.array([750, 0])
        road_params['view_delay'] = 100
        road_params['zoom'] = 2000

        self.road = Road(road_params)

    def test_nodes(self):
        self.road.nodes()

    def test_edges(self):
        self.road.edges()

    def test_routes(self):
        self.road.routes()

    def test_config(self):
        self.road.config()

    def test_gui_settings(self):
        self.road.gui_settings()

    def test_create_road_files(self):
        self.road.create_road()


if __name__ == '__main__':
    unittest.main()
