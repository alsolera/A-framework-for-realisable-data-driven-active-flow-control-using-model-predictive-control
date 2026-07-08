"""AFC environment
Apache Software License 2.0

Copyright (c) 2022, Qiulei Wang

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import math
from os.path import join

import gymnasium as gym
import numpy as np
from gymprecice.core import Adapter
from gymprecice.utils.fileutils import open_file
from gymprecice.utils.openfoamutils import (
    get_interface_patches,
    get_patch_geometry,
    read_line,
)
from scipy import signal
from datetime import datetime

time_str = datetime.now().strftime("%Y%m%d_%H%M")

class JetTruck2DEnv(Adapter):
    r"""An AFC environment for jet truck control.

    ## Description
    This environment corresponds to the DRL-based control case
    We control flow rate of a synthetic jet attached to the base of a 2D truck immersed in a two-dimensional incompressible channel flow.
    The goal is to reduce drag forces acting on the truck.

    ## Action Space
    The action is a `ndarray` with shape `(1,)` which can take values in the range of `[-2.5e-4, 2.5e-4]` with the value corresponding to
    the control flow rate of the synthetic jet.
    **Note**: Each control action (flow rate of the synthetic jet) is smoothly and linearly distributed over `self.action_interval` simulation time steps.

    ## Observation Space
    The observation is a `ndarray` with shape `(25,)` which can take values in the range of `[-Inf, Inf]` with the values corresponding to
    pressure sensor probes allocated within the truck surface.

    ## Rewards
    Since the goal is to keep the drag forces acting on the cylinder as minimum as possible, a reward is defined based on the following relation:
    reward = <Cd> - 0.2 * |<Cl>|, where <Cd> and <Cl> are respectively drag and lift coefficients of the cylinder averaged over the latest 20 s simulation time.
    including the termination step, is allotted. The threshold for rewards is 500 for v1 and 200 for v0.

    ## Episode End
    The episode ends after 50 seconds of flow field simulation.

    Args:
        Adapter: gymprecice adapter super-class
    """

    def __init__(self, options: dict = None, idx: int = 0, openloop: bool = False) -> None:
        """Environment constructor.

        Args:
            options: a dictionary containing the information within gymprecice-config.json. It is a return of `gymprecice.utils.fileutils.make_result_dir` method called within the controller algorithm.
            idx: environment index.
        """
        super().__init__(options, idx)

        # Get a logger specific to this class instance
        self.logger = logging.getLogger(f"{__name__}.Env{idx}")
        # Set a default level (can be configured further if needed)
        if not self.logger.hasHandlers(): # Add handler only if none exist
             # Prevent logs from propagating to root logger if not desired
             self.logger.propagate = False
             self.logger.setLevel(logging.INFO) # Or DEBUG, WARNING etc.
             # Add a NullHandler to prevent "No handler found" warnings if no
             # other handler is configured higher up. Doesn't output anywhere.
             self.logger.addHandler(logging.NullHandler())
             # If you WANT to see env logs in the console where you run MPC_onFOM:
             # stream_handler = logging.StreamHandler()
             # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
             # stream_handler.setFormatter(formatter)
             # self.logger.addHandler(stream_handler)

        self.openloop = openloop

        self.truck_width = 1
        self.truck_length = 7.647
        self.jet_width = 0.05
        self.jet_centers_x = [self.truck_length, self.truck_length]
        self.jet_centers_y = [(self.truck_width - self.jet_width)/2, -(self.truck_width - self.jet_width)/2]

        # Limits to ~1 velocity magnitude in jets
        self._min_jet_rate = - self.jet_width * 1.5
        self._max_jet_rate = self.jet_width * 1.5

        self._n_probes = 90

        self._n_forces = 12
        self._latest_available_sim_time = 0

        self.action_interval = 10
        self.reward_average_time_window = 20

        self._previous_action = None
        self._prerun_data_required = self._latest_available_sim_time > 0.0

        # Add attributes to store the latest calculated forces
        self.latest_cd = np.nan
        self.latest_cl = np.nan

        self.action_space = gym.spaces.Box(
            low=self._min_jet_rate,
            high=self._max_jet_rate,
            shape=(1,),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._n_probes,), dtype=np.float32
        )

        # observations and rewards are obtained from post-processing files
        self._observation_info = {
            "filed_name": "p",
            "n_probes": self._n_probes,  # number of probes
            "file_path": f"/postProcessing/probes/{self._latest_available_sim_time}/p",
            "file_handler": None,
            "data": None,  # live data for the controlled period (t > self._latest_available_sim_time)
        }
        self._reward_info = {
            "filed_name": "forces",
            "n_forces": 12,  # number of data columns (excluding the time column)
            "Cd_column": 1,
            "Cl_column": 3,
            "file_path": f"/postProcessing/forceCoeffs/{self._latest_available_sim_time}/coefficient.dat",
            "file_handler": None,
            "prerun_file_path": "/postProcessing/forceCoeffs/0/coefficient.dat",  # cache data to prevent unnecessary run for the no control period
            "data": None,  # live data for the controlled period (t > self._latest_available_sim_time)
        }

        # find openfoam solver (we have only one openfoam solver)
        openfoam_case_name = ""
        for solver_name in self._solver_list:
            if solver_name.rpartition("-")[-1].lower() == "openfoam":
                openfoam_case_name = solver_name
        self._openfoam_solver_path = join(self._env_path, openfoam_case_name)

        openfoam_interface_patches = get_interface_patches(
            join(openfoam_case_name, "system", "preciceDict")
        )

        action_patch = []
        self.action_patch_geometric_data = {}
        for interface in self._controller_config["write_to"]:
            if interface in openfoam_interface_patches:
                action_patch.append(interface)
        self.action_patch_geometric_data = get_patch_geometry(
            openfoam_case_name, action_patch
        )
        action_patch_coords = {}
        for patch_name in self.action_patch_geometric_data.keys():
            action_patch_coords[patch_name] = [
                np.delete(coord, 2)
                for coord in self.action_patch_geometric_data[patch_name]["face_centre"]
            ]

        observation_patch = []
        for interface in self._controller_config["read_from"]:
            if interface in openfoam_interface_patches:
                observation_patch.append(interface)
        self.observation_patch_geometric_data = get_patch_geometry(
            openfoam_case_name, observation_patch
        )
        observation_patch_coords = {}
        for patch_name in self.observation_patch_geometric_data.keys():
            observation_patch_coords[patch_name] = [
                np.delete(coord, 2)
                for coord in self.observation_patch_geometric_data[patch_name][
                    "face_centre"
                ]
            ]

        patch_coords = {
            "read_from": observation_patch_coords,
            "write_to": action_patch_coords,
        }

        self._set_precice_vectices(patch_coords)

    @property
    def n_probes(self):
        """Get the number of pressure probes."""
        return self._n_probes

    @n_probes.setter
    def n_probes(self, value):
        """Set the number of pressure probes."""
        self._n_probes = value
        self._observation_info["n_probes"] = value
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(value,), dtype=np.float32
        )

    @property
    def n_forces(self):
        """Get the number of data columns in forces file (excluding the time column)."""
        return self._n_forces

    @n_forces.setter
    def n_forces(self, value):
        """Set the number of data columns in forces file (excluding the time column)."""
        assert value >= 3, "Number of forceCoeff columns must be greater than 2"
        self._n_forces = value
        self._reward_info["n_forces"] = value

    @property
    def min_jet_rate(self):
        """Get the minimum flow rate of the synthetic jet."""
        return self._min_jet_rate

    @min_jet_rate.setter
    def min_jet_rate(self, value):
        """Set the minimum flow rate of the synthetic jet."""
        self._min_jet_rate = value
        self.action_space = gym.spaces.Box(
            low=value, high=self._max_jet_rate, shape=(1,), dtype=np.float32
        )

    @property
    def max_jet_rate(self):
        """Get the maximum flow rate of the synthetic jet."""
        return self._max_jet_rate

    @max_jet_rate.setter
    def max_jet_rate(self, value):
        """Set the maximum flow rate of the synthetic jet."""
        self._max_jet_rate = value
        self.action_space = gym.spaces.Box(
            low=self._min_jet_rate, high=value, shape=(1,), dtype=np.float32
        )

    @property
    def latest_available_sim_time(self):
        """Get the starting time of the flow field simulation."""
        return self._latest_available_sim_time

    @latest_available_sim_time.setter
    def latest_available_sim_time(self, value):
        """Set the starting time of the flow field simulation."""
        if value == 0.0:
            value = int(value)
        self._latest_available_sim_time = value
        self._reward_info[
            "file_path"
        ] = f"/postProcessing/forceCoeffs/{value}/coefficient.dat"
        self._observation_info["file_path"] = f"/postProcessing/probes/{value}/p"
        self._prerun_data_required = value > 0.0

    def step(self, action):
        r"""Distribute the control action smoothly and linearly over `self.action_interval` simulation time steps."""
        return self._smooth_step(action)

    def _get_action(self, action, write_var_list):
        acuation_interface_field = self._action_to_patch_field(action)
        write_data = {
            var: acuation_interface_field[var.rpartition("-")[-1]]
            for var in write_var_list
        }
        return write_data

    def _get_observation(self, read_data, read_var_list):
        return self._probes_to_observation()

    def _get_reward(self):
        return self._forces_to_reward()

    def _close_external_resources(self):
        # close probes and forces files
        try:
            if self._observation_info["file_handler"] is not None:
                self._observation_info["file_handler"].close()
                self._observation_info["file_handler"] = None
            if self._reward_info["file_handler"] is not None:
                self._reward_info["file_handler"].close()
                self._reward_info["file_handler"] = None
        except Exception as err:
            self.logger.error("Can't close probes/forces file")
            raise err

    def _action_to_patch_field(self, action):
        patch_flow_rate = [-action, action]

        # velocity field of the actuation patches
        U_profile = {}
        for idx, patch_name in enumerate(self.action_patch_geometric_data.keys()):
            patch_ctr = np.array(
                [
                    self.jet_centers_x[idx],
                    self.jet_centers_y[idx],
                    0.5,
                ]
            )
            magSf = self.action_patch_geometric_data[patch_name]["face_area_mag"]
            Sf = self.action_patch_geometric_data[patch_name]["face_area_vector"]
            nf = self.action_patch_geometric_data[patch_name]["face_normal"]
            Sc = self.action_patch_geometric_data[patch_name]["face_centre"]
            w_patch = self.jet_width

            # Build a parabolic velocity profile along face normals
            avg_U = (patch_flow_rate[idx] / np.sum(magSf)).item()
            max_U = avg_U * 3/2  # from the parabolic profile

            dists_to_centre = np.linalg.norm(Sc - patch_ctr, axis=1)
            face_velocities = max_U * (1 - (2*dists_to_centre/w_patch)**2)
            face_velocities = face_velocities[:, np.newaxis] * nf

            # estimate total flow rate
            Q_calc = sum([u.dot(s) for u, s in zip(face_velocities, Sf)])

            # correct velocity profile to enforce mass conservation
            if not np.isclose(patch_flow_rate[idx], 0):
                Q_err = patch_flow_rate[idx] / Q_calc
                U_patch = face_velocities * Q_err
            else:
                U_patch = face_velocities * 0

            # return the velocity profile
            Q_final = sum([u.dot(s) for u, s in zip(U_patch, Sf)])

            if np.isclose(Q_final, patch_flow_rate[idx]):  # check if final Q is correct
                # remove z component for 2D case
                U_profile[patch_name] = np.array(
                    [np.delete(item, 2) for item in U_patch]
                )
            else:
                print(f'Q achieved: {Q_final}, Q expected: {patch_flow_rate[idx]}')
                self.logger.error("Error: Not a synthetic jet: Q_jet1 + Q_jet2 is not zero")
                raise Exception("Not a synthetic jet: Q_jet1 + Q_jet2 is not zero")

        return U_profile

    def _probes_to_observation(self):
        if self.openloop:
            return
        # If coupling ended, skip reading
        if self._interface is None or not self._interface.is_coupling_ongoing():
            return None  # caller (_smooth_step) already handles this via truncated=True
        self._read_probes_from_file()

        assert self._observation_info["data"], "probes-data is empty!"
        probes_data = self._observation_info["data"]

        latest_time_data = np.array(
            probes_data[-1][2]
        )  # only the last timestep and remove the time and size columns

        return np.stack(latest_time_data, axis=0)

    def _forces_to_reward(self):
        if self.openloop:
            return
        # If coupling ended, skip reading and return last known reward
        if self._interface is None or not self._interface.is_coupling_ongoing():
            reward = -self.latest_cd - 0.2 * np.abs(self.latest_cl)
            return reward
        self._read_forces_from_file()

        assert self._reward_info["data"], "forces-data is empty!"
        forces_data = self._reward_info["data"]

        n_lookback = int(self.reward_average_time_window // self._dt) + 1

        # get the data within a time_window for computing reward
        if self._time_window == 0:
            time_bound = [0, self.reward_average_time_window]
        else:
            time_bound = [
                (self._time_window - n_lookback) * self._dt
                + self.reward_average_time_window,
                self._time_window * self._dt + self.reward_average_time_window,
            ]

        # avoid the starting again and again from t0 by working in reverse order
        reversed_forces_data = forces_data[::-1]
        reward_data = []

        for data_line in reversed_forces_data:
            time_stamp = data_line[0]
            if time_stamp <= time_bound[0]:
                break
            reward_data.append(data_line)

        cd = np.array(
            [
                [x[0], x[2][self._reward_info["Cd_column"] - 1]]
                for x in reward_data[::-1]
            ]
        )
        cl = np.array(
            [
                [x[0], x[2][self._reward_info["Cl_column"] - 1]]
                for x in reward_data[::-1]
            ]
        )

        start_time_step = cd[0, 0]
        latest_time_step = cd[-1, 0]

        # average is not correct when using adaptive time-stepping
        cd_uniform = np.interp(
            np.linspace(start_time_step, latest_time_step, num=100, endpoint=True),
            cd[:, 0],
            cd[:, 1],
        )
        cl_uniform = np.interp(
            np.linspace(start_time_step, latest_time_step, num=100, endpoint=True),
            cl[:, 0],
            cl[:, 1],
        )
        # for constant time stepping one can filter the signals
        cd_filtered = signal.savgol_filter(cd_uniform, 49, 0)
        cl_filtered = signal.savgol_filter(cl_uniform, 49, 0)

        # Store the calculated mean values as instance attributes
        self.latest_cd = cd[-1, 1]
        self.latest_cl = cl[-1, 1]

        # Calculate reward based on the stored values
        reward = -self.latest_cd - 0.2 * np.abs(self.latest_cl)

        reward = - np.mean(cd_filtered) - 0.2 * np.abs(np.mean(cl_filtered))
        return reward

    def _read_probes_from_file(self):
        # sequential read of a single line (last line) of probes file at each RL-Gym step
        data_path = f"{self._openfoam_solver_path}{self._observation_info['file_path']}"

        self.logger.debug(f"reading pressure probes from: {data_path}")

        if self._observation_info["file_handler"] is None:
            file_object = open_file(data_path)
            self._observation_info["file_handler"] = file_object
            self._observation_info["data"] = []

        new_time_stamp = True
        latest_time_stamp = self._t + self._latest_available_sim_time
        if self._observation_info["data"]:
            new_time_stamp = self._observation_info["data"][-1][0] != latest_time_stamp

        if new_time_stamp:
            time_stamp = 0
            while not math.isclose(
                time_stamp, latest_time_stamp):
                # --- Check if coupling is still ongoing before trying to read ---
                if self._interface is None or not self._interface.is_coupling_ongoing():
                    self.logger.warning(f"Coupling ended; breaking probe read loop at time {time_stamp}.")
                    break
                # --------------------------------------------------------------------
                while True:
                    # --- Also check here in case inner loop gets stuck ---
                    if self._interface is None or not self._interface.is_coupling_ongoing():
                        break
                    # --------------------------------------------------------
                    is_comment, time_stamp, n_probes, probes_data = read_line(
                        self._observation_info["file_handler"],
                        self._observation_info["n_probes"],
                    )
                    if (
                        not is_comment
                        and n_probes == self._observation_info["n_probes"] # noqa
                    ):
                        break
                self._observation_info["data"].append(
                    [time_stamp, n_probes, probes_data]
                )
            if self._interface.is_coupling_ongoing():
                assert math.isclose(
                    time_stamp, latest_time_stamp
                ), f"Mismatched time data: {time_stamp} vs {self._t}"

        self.logger.debug(f"Did read pressure probes")

    def _read_forces_from_file(self):
        # sequential read of a single line (last line) of forces file at each RL step
        if self._prerun_data_required:
            self._reward_info["data"] = []

            data_path = (
                f"{self._openfoam_solver_path}{self._reward_info['prerun_file_path']}"
            )
            self.logger.debug(f"reading pre-run forces from: {data_path}")

            file_object = open_file(data_path)
            self._reward_info["file_handler"] = file_object

            latest_time_stamp = self._latest_available_sim_time

            time_stamp = 0
            while not math.isclose(
                time_stamp, latest_time_stamp
            ):  # read till the end of pre-run data
                while True:
                    # --- FIX: Also check here in case inner loop gets stuck ---
                    if self._interface is None or not self._interface.is_coupling_ongoing():
                        break
                    # --------------------------------------------------------
                    is_comment, time_stamp, n_forces, forces_data = read_line(
                        self._reward_info["file_handler"], self._reward_info["n_forces"]
                    )
                    if not is_comment and n_forces == self._reward_info["n_forces"]:
                        break

                # If the loop was broken by the coupling check, exit the outer loop too
                if not self._interface.is_coupling_ongoing() and not math.isclose(time_stamp, latest_time_stamp):
                    break

                self._reward_info["data"].append([time_stamp, n_forces, forces_data])
            assert math.isclose(
                time_stamp, latest_time_stamp
            ), f"Mismatched time data: {time_stamp} vs {self._t}"

            self._prerun_data_required = False

            self._reward_info["file_handler"].close()

            data_path = f"{self._openfoam_solver_path}{self._reward_info['file_path']}"
            file_object = open_file(data_path)
            self._reward_info["file_handler"] = file_object

        else:
            data_path = f"{self._openfoam_solver_path}{self._reward_info['file_path']}"

            if self._reward_info["file_handler"] is None:
                file_object = open_file(data_path)
                self._reward_info["file_handler"] = file_object
                if not self._reward_info.get("data"): # Initialize if not present
                    self._reward_info["data"] = []

        self.logger.debug(f"reading forces from: {data_path}")

        new_time_stamp = True
        latest_time_stamp = self._t + self._latest_available_sim_time
        if self._reward_info["data"]:
            new_time_stamp = self._reward_info["data"][-1][0] != latest_time_stamp

        if new_time_stamp:
            time_stamp = 0
            while not math.isclose(time_stamp, latest_time_stamp):
                # --- FIX: Check if coupling is still ongoing before trying to read ---
                if self._interface is None or not self._interface.is_coupling_ongoing():
                    self.logger.warning(f"Coupling ended; breaking forces read loop at time {time_stamp}.")
                    break
                # --------------------------------------------------------------------

                while True:
                    is_comment, time_stamp, n_forces, forces_data = read_line(
                        self._reward_info["file_handler"], self._reward_info["n_forces"]
                    )
                    if not is_comment and n_forces == self._reward_info["n_forces"]:
                        break
                self._reward_info["data"].append([time_stamp, n_forces, forces_data])
            # It's okay if we broke out early, but if the loop finished naturally, time should match
            if self._interface.is_coupling_ongoing():
                assert math.isclose(
                    time_stamp, latest_time_stamp
                ), f"Mismatched time data: {time_stamp} vs {self._t}"

        self.logger.debug(f"Did read forces")


    def _smooth_step(self, action):  # 'action' here is ALREADY in the base env scale [-0.075, 0.075]
        self.logger.debug(f'Step initiated, base action received: {action}')
        if self._previous_action is None:  # Initialize previous action if first step
            self._previous_action = action.copy()  # Start with the target action

        subcycle = 0
        terminated = False  # Initialize termination flags
        truncated = False

        while subcycle < self.action_interval:
            self.logger.debug(f'Subcycle start: {subcycle}')
            # Check if interval is already completed
            remaining_subcycles = self.action_interval - subcycle
            if remaining_subcycles <= 0: break  # Should not happen with < check, but safe

            action_fraction = 1.0 / remaining_subcycles  # Correct fraction application
            # Calculate the action to apply in this sub-step (interpolation)
            smoothed_action = self._previous_action + action_fraction * (
                    action - self._previous_action
            )

            self.logger.debug(
                f"Calling super().step with SMOOTHED action: {smoothed_action} (dtype: {smoothed_action.dtype if hasattr(smoothed_action, 'dtype') else type(smoothed_action)})")
            self.logger.debug(f"Action space being checked: {self.action_space}")
            try:
                # Ensure smoothed_action is within bounds and correct dtype before passing
                smoothed_action_clipped = np.clip(smoothed_action, self.action_space.low, self.action_space.high)
                smoothed_action_final = smoothed_action_clipped.astype(self.action_space.dtype)

                next_obs, reward, terminated, truncated, info = super().step(
                    smoothed_action_final  # Pass the interpolated and validated action
                )

                if self._interface is None or not self._interface.is_coupling_ongoing():
                    # Normal end of simulation
                    return next_obs, reward, False, True, info  # truncated=True

                # Populate the info dictionary with the latest calculated forces
                info['Cd'] = self.latest_cd
                info['Cl'] = self.latest_cl
            except AssertionError as e:
                # Catch the assertion error specifically if it happens again
                self.logger.error(f"AssertionError during super().step: {e}")
                self.logger.error(
                    f"  Action passed: {smoothed_action_final}, Type: {type(smoothed_action_final)}, Dtype: {smoothed_action_final.dtype}")
                self.logger.error(f"  Action space: {self.action_space}")
                # Decide how to handle: re-raise, terminate, etc.
                raise e

            subcycle += 1
            if terminated or truncated:
                self._previous_action = None  # Reset previous action on termination
                self.logger.debug(f'Step terminated early: {terminated}, truncated: {truncated}, subcycle: {subcycle}')
                break
            else:
                # Update previous_action using the SMOOTHED value for the next subcycle
                self._previous_action = smoothed_action  # Use the calculated smoothed value

            self.logger.debug(
                f'Subcycle finished: terminated: {terminated}, truncated: {truncated}, subcycle: {subcycle}')

        # If loop completes normally, set previous action to the target for the next full step
        if not (terminated or truncated):
            self._previous_action = action.copy()

        self.logger.debug(f'Step finished: terminated: {terminated}, truncated: {truncated}')
        # Return the *last* observation, reward, flags obtained from the loop
        if next_obs is None:  # Handle case where loop didn't run (e.g., action_interval=0) or failed first step
            # Need a valid observation; perhaps call reset or return a default?
            # This indicates a problem state. For now, raise error or return placeholder
            raise RuntimeError("_smooth_step completed without obtaining a valid next_obs")

        return next_obs, reward, terminated, truncated, info
