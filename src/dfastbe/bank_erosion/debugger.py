from shapely.geometry import Point
from geopandas.geoseries import GeoSeries
from dfastbe.io import write_shp, write_csv, ConfigFile


class Debugger:
    def __init__(self, config_file: ConfigFile, river_data):
        self.config_file = config_file
        self.river_data = river_data

    def debug_process_discharge_levels_1(
            self, ib, bank_data, fairway_data, erosion_inputs, pars, hfw, dn_eq1, dv_eq1, bcrds, bank_height,
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
            "zb": bank_height[ib],
            "len": segment_length[ib],
            "zw0": fairway_data.fairway_initial_water_levels[ib],
            "vship": pars["v_ship"][ib],
            "shiptype": pars["ship_type"][ib],
            "draught": pars["t_ship"][ib],
            "mu_slp": pars["mu_slope"][ib],
            "dist_fw": bank_data.fairway_distances[ib],
            "dfw0": erosion_inputs.wave_fairway_distance_0[ib],
            "dfw1": erosion_inputs.wave_fairway_distance_1[ib],
            "hfw": hfw,
            "zss": erosion_inputs.bank_protection_level[ib],
            "dn": dn_eq1,
            "dv": dv_eq1,
        }

        write_shp(bank_coords_geo, params, f"{str(self.river_data.output_dir)}/debug.EQ.B{ib + 1}.shp")
        write_csv(params, f"{str(self.river_data.output_dir)}/debug.EQ.B{ib + 1}.csv")

    def debug_process_discharge_levels_2(
        self, ib, iq, bank_data, fairway_data, erosion_inputs, pars, hfw, bcrds, velocity, bank_height, segment_length,
        water_level, chezy, dniqib, dviqib, dn_ship, dn_flow
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
            "u": velocity[iq][ib],
            "zb": bank_height[ib],
            "len": segment_length[ib],
            "zw": water_level[iq][ib],
            "zw0": fairway_data.fairway_initial_water_levels[ib],
            "tauc": erosion_inputs.tauc[ib],
            "nship": pars["n_ship"][ib],
            "vship": pars["v_ship"][ib],
            "nwave": pars["n_wave"][ib],
            "shiptype": pars["ship_type"][ib],
            "draught": pars["t_ship"][ib],
            "mu_slp": pars["mu_slope"][ib],
            "mu_reed": pars["mu_reed"][ib],
            "dist_fw": bank_data.fairway_distances[ib],
            "dfw0": erosion_inputs.wave_fairway_distance_0[ib],
            "dfw1": erosion_inputs.wave_fairway_distance_1[ib],
            "hfw": hfw,
            "chez": chezy[iq][ib],
            "zss": erosion_inputs.bank_protection_level[ib],
            "dn": dniqib,
            "dv": dviqib,
            "dnship": dn_ship,
            "dnflow": dn_flow,
        }
        write_shp(bank_coords_geo, params, f"{str(self.river_data.output_dir)}/debug.Q{iq + 1}.B{ib + 1}.shp")
        write_csv(params, f"{str(self.river_data.output_dir)}/debug.Q{iq + 1}.B{ib + 1}.csv")