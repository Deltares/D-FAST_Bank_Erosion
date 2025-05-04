from shapely.geometry import Point
from geopandas.geoseries import GeoSeries
from dfastbe.io import write_shp, write_csv, ConfigFile


class Debugger:
    def __init__(self, config_file: ConfigFile, river_data):
        self.config_file = config_file
        self.river_data = river_data

    def debug_process_discharge_levels_1(
            self, ib, bank_data, fairway_data, erosion_inputs, pars, water_depth_fairway, dn_eq1, dv_eq1, bcrds, bank_height,
            segment_length
    ):
        bcrds_mid = (bcrds[:-1] + bcrds[1:]) / 2
        bank_coords_points = [Point(xy) for xy in bcrds_mid]
        bank_coords_geo = GeoSeries(
            bank_coords_points, crs=self.config_file.crs
        )
        params = {
            "chainage": bank_data.bank_chainage_midpoints[ib],
            "x": bcrds_mid[:, 0],
            "y": bcrds_mid[:, 1],
            "iface_fw": bank_data.fairway_face_indices[ib],
            "iface_bank": bank_data.bank_face_indices[ib],  # bank_index
            "bank_height": bank_height[ib],
            "segment_length": segment_length[ib],
            "zw0": fairway_data.fairway_initial_water_levels[ib],
            "ship_velocity": pars["v_ship"][ib],
            "ship_type": pars["ship_type"][ib],
            "draught": pars["t_ship"][ib],
            "mu_slp": pars["mu_slope"][ib],
            "bank_fairway_dist": bank_data.fairway_distances[ib],
            "fairway_wave_reduction_distance": erosion_inputs.wave_fairway_distance_0[ib],
            "fairway_wave_disappear_distance": erosion_inputs.wave_fairway_distance_1[ib],
            "water_depth_fairway": water_depth_fairway,
            "dike_height": erosion_inputs.bank_protection_level[ib],
            "erosion_distance": dn_eq1,
            "erosion_volume": dv_eq1,
        }

        write_shp(bank_coords_geo, params, f"{str(self.river_data.output_dir)}/debug.EQ.B{ib + 1}.shp")
        write_csv(params, f"{str(self.river_data.output_dir)}/debug.EQ.B{ib + 1}.csv")

    def debug_process_discharge_levels_2(
        self, ib, iq, bank_data, fairway_data, erosion_inputs, pars, water_depth_fairway, bcrds, velocity, bank_height, segment_length,
        water_level, chezy, dniqib, dviqib, erosion_distance_shipping, erosion_distance_flow
    ):
        bcrds_mid = (bcrds[:-1] + bcrds[1:]) / 2

        bank_coords_points = [Point(xy1) for xy1 in bcrds_mid]
        bank_coords_geo = GeoSeries(bank_coords_points, crs=self.config_file.crs)
        params = {
            "chainage": bank_data.bank_chainage_midpoints[ib],
            "x": bcrds_mid[:, 0],
            "y": bcrds_mid[:, 1],
            "iface_fw": bank_data.fairway_face_indices[ib],
            "iface_bank": bank_data.bank_face_indices[ib],  # bank_index
            "velocity": velocity[iq][ib],
            "bank_height": bank_height[ib],
            "segment_length": segment_length[ib],
            "zw": water_level[iq][ib],
            "zw0": fairway_data.fairway_initial_water_levels[ib],
            "tauc": erosion_inputs.tauc[ib],
            "num_ship": pars["n_ship"][ib],
            "ship_velocity": pars["v_ship"][ib],
            "num_waves_per_ship": pars["n_wave"][ib],
            "ship_type": pars["ship_type"][ib],
            "draught": pars["t_ship"][ib],
            "mu_slp": pars["mu_slope"][ib],
            "mu_reed": pars["mu_reed"][ib],
            "dist_fw": bank_data.fairway_distances[ib],
            "fairway_wave_reduction_distance": erosion_inputs.wave_fairway_distance_0[ib],
            "fairway_wave_disappear_distance": erosion_inputs.wave_fairway_distance_1[ib],
            "water_depth_fairway": water_depth_fairway,
            "chez": chezy[iq][ib],
            "dike_height": erosion_inputs.bank_protection_level[ib],
            "erosion_distance": dniqib,
            "erosion_volume": dviqib,
            "erosion_distance_shipping": erosion_distance_shipping,
            "erosion_distance_flow": erosion_distance_flow,
        }
        write_shp(bank_coords_geo, params, f"{str(self.river_data.output_dir)}/debug.Q{iq + 1}.B{ib + 1}.shp")
        write_csv(params, f"{str(self.river_data.output_dir)}/debug.Q{iq + 1}.B{ib + 1}.csv")