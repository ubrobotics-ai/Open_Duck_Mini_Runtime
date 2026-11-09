import time
import pickle

import numpy as np
from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.onnx_infer import OnnxInfer

from mini_bdx_runtime.raw_imu import Imu
from mini_bdx_runtime.poly_reference_motion import PolyReferenceMotion
from mini_bdx_runtime.xbox_controller import XBoxController
from mini_bdx_runtime.feet_contacts import FeetContacts
from mini_bdx_runtime.eyes import Eyes
from mini_bdx_runtime.sounds import Sounds
from mini_bdx_runtime.antennas import Antennas
from mini_bdx_runtime.projector import Projector
from mini_bdx_runtime.rl_utils import make_action_dict, LowPassActionFilter
from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.ports import SERIAL_PORT

import os

HOME_DIR = os.path.expanduser("~")


class RLWalk:
    def __init__(
        self,
        onnx_model_path: str,
        duck_config_path: str = f"{HOME_DIR}/duck_config.json",
        serial_port: str = SERIAL_PORT,
        control_freq: float = 50,
        pid=[30, 0, 0],
        action_scale=0.25,
        commands=False,
        pitch_bias=0,
        save_obs=False,
        replay_obs=None,
        cutoff_frequency=None,
    ):

        self.duck_config = DuckConfig(config_json_path=duck_config_path)

        self.commands = commands
        self.pitch_bias = pitch_bias

        self.onnx_model_path = onnx_model_path
        self.policy = OnnxInfer(self.onnx_model_path, awd=True)

        self.num_dofs = 14
        self.max_motor_velocity = 5.24  # rad/s

        # Control
        self.control_freq = control_freq
        self.pid = pid

        self.save_obs = save_obs
        if self.save_obs:
            self.saved_obs = []

        self.replay_obs = replay_obs
        if self.replay_obs is not None:
            self.replay_obs = pickle.load(open(self.replay_obs, "rb"))

        self.action_filter = None
        if cutoff_frequency is not None:
            self.action_filter = LowPassActionFilter(
                self.control_freq, cutoff_frequency
            )

        self.hwi = HWI(self.duck_config, serial_port)

        self.start()

        self.imu = Imu(
            sampling_freq=int(self.control_freq),
            user_pitch_bias=self.pitch_bias,
            upside_down=self.duck_config.imu_upside_down,
            axis_remap=self.duck_config.imu_axis_remap,
            accel_offsets=self.duck_config.imu_accel_offsets,
        )

        self.feet_contacts = FeetContacts()

        # Scales
        self.action_scale = action_scale

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        self.init_pos = list(self.hwi.init_pos.values())

        self.motor_targets = np.array(self.init_pos.copy())
        self.prev_motor_targets = np.array(self.init_pos.copy())

        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.last_ramp_t = None

        self.paused = self.duck_config.start_paused

        self.command_freq = 20  # hz
        if self.commands:
            # DUCK_CONTROLLER=network: commands from a laptop over UDP
            # (mini_bdx_runtime/network_controller.py); default: gamepad
            if os.environ.get("DUCK_CONTROLLER", "xbox") == "network":
                from mini_bdx_runtime.network_controller import NetworkController

                self.xbox_controller = NetworkController(self.command_freq)
            else:
                self.xbox_controller = XBoxController(self.command_freq)

        # Reference motion, but we only really need the length of one phase
        # TODO
        self.PRM = PolyReferenceMotion("./polynomial_coefficients.pkl")
        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.phase_frequency_factor = 1.0
        self.phase_frequency_factor_offset = (
            self.duck_config.phase_frequency_factor_offset
        )

        # Optional expression features
        if self.duck_config.eyes:
            self.eyes = Eyes()
        if self.duck_config.projector:
            self.projector = Projector()
        if self.duck_config.speaker:
            self.sounds = Sounds(
                volume=1.0, sound_directory="../mini_bdx_runtime/assets/"
            )
        if self.duck_config.antennas:
            self.antennas = Antennas()

    def get_obs(self):

        imu_data = self.imu.get_data()

        dof_pos = self.hwi.get_present_positions(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad

        dof_vel = self.hwi.get_present_velocities(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad/s

        if dof_pos is None or dof_vel is None:
            return None

        if len(dof_pos) != self.num_dofs:
            print(f"ERROR len(dof_pos) != {self.num_dofs}")
            return None

        if len(dof_vel) != self.num_dofs:
            print(f"ERROR len(dof_vel) != {self.num_dofs}")
            return None

        cmds = self.last_commands
        self.last_gz = float(imu_data["gyro"][2])  # for hold_heading (rad/s, + = turning left)

        feet_contacts = self.feet_contacts.get()

        obs = np.concatenate(
            [
                imu_data["gyro"],
                imu_data["accelero"],
                cmds,
                dof_pos - self.init_pos,
                dof_vel * 0.05,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                feet_contacts,
                self.imitation_phase,
            ]
        )

        return obs

    def start(self):
        kps = [self.pid[0]] * 14
        kds = [self.pid[2]] * 14

        # lower head kps
        kps[5:9] = [8, 8, 8, 8]
        # Duck 1 (9 Nov 2026): since the head rebuild the head_pitch needs 10-20 % torque to look
        # down; at kp 8 it stays near 0 instead of the policy's -0.19. DUCK_HEAD_PITCH_KP=30 restores it.
        kps[6] = int(os.environ.get("DUCK_HEAD_PITCH_KP", kps[6]))

        self.hwi.set_kps(kps)
        self.hwi.set_kds(kds)
        self.hwi.turn_on()

        time.sleep(2)

    def get_phase_frequency_factor(self, x_velocity):

        max_phase_frequency = 1.2
        min_phase_frequency = 1.0

        # Perform linear interpolation
        freq = min_phase_frequency + (abs(x_velocity) / 0.15) * (
            max_phase_frequency - min_phase_frequency
        )

        return freq

    # Speed limits per second for vx, vy, wz (m/s, m/s, rad/s). Stopping at once from a
    # fast walk tips the duck forward, so the commands follow the stick through a ramp.
    # Speeding up is quicker than slowing down. Head commands (3..6) pass straight through.
    RAMP_UP = (0.30, 0.40, 2.0)
    RAMP_DOWN = (0.15, 0.20, 1.5)

    def ramp_commands(self, target, t):
        dt = 0.0 if self.last_ramp_t is None else min(t - self.last_ramp_t, 0.1)
        self.last_ramp_t = t
        out = list(target)
        for k in range(3):
            cur, tgt = self.last_commands[k], target[k]
            speeding_up = abs(tgt) > abs(cur) and tgt * cur >= 0
            step = (self.RAMP_UP if speeding_up else self.RAMP_DOWN)[k] * dt
            out[k] = cur + max(-step, min(step, tgt - cur))
        return out

    # Heading hold (duck 1, Nov 2026): the stock walk drifts 30–50° in 15 s on the real duck, and
    # curved at up to 0.24 rad/s with bad zeros. While walking with no turn asked, integrate the
    # gyro's z and add a small turn command against the drift. Asking for a turn, or stopping,
    # takes the current heading as the new reference (after HOLD_SETTLE, while the turn ramps
    # down). In the simulator with duck 1's masses and a curving leg: 19° off in 10 s without it,
    # 1° with it. DUCK_HEADING_HOLD=0 disables it.
    HOLD_KP = 1.0  # rad/s of turn command per rad of heading error
    HOLD_MAX = 0.3  # rad/s, largest correction
    HOLD_DEAD = 0.02  # |wz| below this is "no turn asked"
    HOLD_SETTLE = 0.5  # s after a turn or a stop before holding

    def hold_heading(self, target, t):
        """target: the 7 commands from the controller (wz + = left); returns them with wz corrected."""
        if os.environ.get("DUCK_HEADING_HOLD", "1") == "0":
            return target
        dt = 0.0 if getattr(self, "_hold_t", None) is None else min(t - self._hold_t, 0.1)
        self._hold_t = t
        walking = abs(target[0]) > 0.01 or abs(target[1]) > 0.01
        if not walking or abs(target[2]) >= self.HOLD_DEAD:
            self.held_yaw = 0.0
            self._hold_from = t + self.HOLD_SETTLE
            return target
        if t < getattr(self, "_hold_from", 0.0):
            self.held_yaw = 0.0
            return target
        self.held_yaw = getattr(self, "held_yaw", 0.0) + getattr(self, "last_gz", 0.0) * dt
        out = list(target)
        out[2] = float(np.clip(-self.HOLD_KP * self.held_yaw, -self.HOLD_MAX, self.HOLD_MAX))
        return out

    # Fall detection: the accelerometer's z (obs[5]) stays below FALL_AZ (≈ 60° from upright)
    # for FALL_STEPS control steps -> torque off and stop. Without it the policy keeps driving
    # the servos against the floor after a fall: stall current, brownouts and a Radxa reboot
    # (duck 1, Oct–Nov 2026). DUCK_FALL_OFF=0 disables it.
    FALL_AZ = 5.0  # m/s²
    FALL_STEPS = 15  # 0.3 s at 50 Hz

    def fallen(self, obs):
        if os.environ.get("DUCK_FALL_OFF", "1") == "0":
            return False
        self._fall_count = getattr(self, "_fall_count", 0) + 1 if obs[5] < self.FALL_AZ else 0
        return self._fall_count >= self.FALL_STEPS

    def save_obs_partial(self):
        """Every 2 s, write the observations so far in a thread (fsync'd, atomic rename): when the
        BMS cuts the power on a fall (duck 1, 9 Nov 2026) the log up to the fall survives."""
        if getattr(self, "_saving", None) is not None and self._saving.is_alive():
            return
        snapshot = list(self.saved_obs)

        def write():
            tmp = "robot_saved_obs.pkl.tmp"
            with open(tmp, "wb") as f:
                pickle.dump(snapshot, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, "robot_saved_obs.pkl")

        import threading

        self._saving = threading.Thread(target=write, daemon=True)
        self._saving.start()

    def realtime(self):
        """Run the control loop on its own core with real-time priority, so other work on the
        Radxa (the duck service, the camera, Wi-Fi) doesn't delay the 50 Hz loop. Needs
        CAP_SYS_NICE (given by pato.service); without it the loop runs as before."""
        core = int(os.environ.get("DUCK_RT_CORE", "3"))
        try:
            os.sched_setaffinity(0, {core})
        except (AttributeError, OSError, ValueError) as e:
            print(f"realtime: could not pin to core {core}: {e}")
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(int(os.environ.get("DUCK_RT_PRIO", "50"))))
            print(f"realtime: SCHED_FIFO on core {core}")
        except (AttributeError, PermissionError, OSError) as e:
            print(f"realtime: no real-time priority ({e}); loop runs at normal priority")

    def run(self):
        self.realtime()
        loop_times = []
        late = 0
        stats_t = time.time()
        i = 0
        try:
            print("Starting")
            start_t = time.time()
            while True:
                left_trigger = 0
                right_trigger = 0
                t = time.time()

                if self.commands:
                    target_commands, self.buttons, left_trigger, right_trigger = (
                        self.xbox_controller.get_last_command()
                    )
                    target_commands = self.hold_heading(target_commands, t)
                    self.last_commands = self.ramp_commands(target_commands, t)
                    if self.buttons.dpad_up.triggered:
                        self.phase_frequency_factor_offset += 0.05
                        print(
                            f"Phase frequency factor offset {round(self.phase_frequency_factor_offset, 3)}"
                        )

                    if self.buttons.dpad_down.triggered:
                        self.phase_frequency_factor_offset -= 0.05
                        print(
                            f"Phase frequency factor offset {round(self.phase_frequency_factor_offset, 3)}"
                        )

                    if self.buttons.LB.is_pressed:
                        self.phase_frequency_factor = 1.3
                    else:
                        self.phase_frequency_factor = 1.0

                    if self.buttons.X.triggered:
                        if self.duck_config.projector:
                            self.projector.switch()

                    if self.buttons.B.triggered:
                        if self.duck_config.speaker:
                            self.sounds.play_random_sound()

                    if self.duck_config.antennas:
                        self.antennas.set_position_left(right_trigger)
                        self.antennas.set_position_right(left_trigger)

                    if self.buttons.A.triggered:
                        self.paused = not self.paused
                        if self.paused:
                            print("PAUSE")
                        else:
                            print("UNPAUSE")

                if self.paused:
                    time.sleep(0.1)
                    continue

                obs = self.get_obs()
                if obs is None:
                    continue

                if self.fallen(obs):
                    print(f"FALL: accel z {obs[5]:.1f} m/s² for {self.FALL_STEPS} steps, torque off", flush=True)
                    self.hwi.turn_off()
                    break

                self.imitation_i += 1 * (
                    self.phase_frequency_factor + self.phase_frequency_factor_offset
                )
                self.imitation_i = self.imitation_i % self.PRM.nb_steps_in_period
                self.imitation_phase = np.array(
                    [
                        np.cos(
                            self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi
                        ),
                        np.sin(
                            self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi
                        ),
                    ]
                )

                if self.save_obs:
                    self.saved_obs.append(obs)
                    if len(self.saved_obs) % 100 == 0:
                        self.save_obs_partial()

                if self.replay_obs is not None:
                    if i < len(self.replay_obs):
                        obs = self.replay_obs[i]
                    else:
                        print("BREAKING ")
                        break

                action = self.policy.infer(obs)

                self.last_last_last_action = self.last_last_action.copy()
                self.last_last_action = self.last_action.copy()
                self.last_action = action.copy()

                # action = np.zeros(10)

                self.motor_targets = self.init_pos + action * self.action_scale

                # self.motor_targets = np.clip(
                #     self.motor_targets,
                #     self.prev_motor_targets
                #     - self.max_motor_velocity * (1 / self.control_freq),  # control dt
                #     self.prev_motor_targets
                #     + self.max_motor_velocity * (1 / self.control_freq),  # control dt
                # )

                if self.action_filter is not None:
                    self.action_filter.push(self.motor_targets)
                    filtered_motor_targets = self.action_filter.get_filtered_action()
                    if (
                        time.time() - start_t > 1
                    ):  # give time to the filter to stabilize
                        self.motor_targets = filtered_motor_targets

                self.prev_motor_targets = self.motor_targets.copy()

                head_motor_targets = self.last_commands[3:] + self.motor_targets[5:9]
                self.motor_targets[5:9] = head_motor_targets

                action_dict = make_action_dict(
                    self.motor_targets, list(self.hwi.joints.keys())
                )

                self.hwi.set_position_all(action_dict)

                i += 1

                took = time.time() - t
                loop_times.append(took)
                if took > 1 / self.control_freq:
                    late += 1
                if t - stats_t > 10:
                    print(f"loop: mean {np.mean(loop_times)*1000:.1f} ms, max {np.max(loop_times)*1000:.1f} ms, "
                          f"late {late}/{len(loop_times)}, heading hold {np.degrees(getattr(self, 'held_yaw', 0.0)):+.0f}°")
                    loop_times, late, stats_t = [], 0, t
                # print("Full loop took", took, "fps : ", np.around(1 / took, 2))
                if (1 / self.control_freq - took) < 0:
                    print(
                        "Policy control budget exceeded by",
                        np.around(took - 1 / self.control_freq, 3),
                    )
                time.sleep(max(0, 1 / self.control_freq - took))

        except KeyboardInterrupt:
            if self.duck_config.antennas:
                self.antennas.stop()
            if self.duck_config.eyes:
                self.eyes.stop()
            if self.duck_config.projector:
                self.projector.stop()
            self.feet_contacts.stop()

        if self.save_obs:
            if getattr(self, "_saving", None) is not None:
                self._saving.join()  # a partial write must not land after the full one
            pickle.dump(self.saved_obs, open("robot_saved_obs.pkl", "wb"))
        print("TURNING OFF")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", type=str, required=True)
    parser.add_argument(
        "--duck_config_path",
        type=str,
        required=False,
        default=f"{HOME_DIR}/duck_config.json",
    )
    parser.add_argument("-a", "--action_scale", type=float, default=0.25)
    parser.add_argument("-p", type=int, default=30)
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-c", "--control_freq", type=int, default=50)
    parser.add_argument("--pitch_bias", type=float, default=0, help="deg")
    parser.add_argument(
        "--commands",
        action="store_true",
        default=True,
        help="external commands, keyboard or gamepad. Launch control_server.py on host computer",
    )
    parser.add_argument(
        "--save_obs",
        type=str,
        required=False,
        default=False,
        help="save the run's observations",
    )
    parser.add_argument(
        "--replay_obs",
        type=str,
        required=False,
        default=None,
        help="replay the observations from a previous run (can be from the robot or from mujoco)",
    )
    parser.add_argument("--cutoff_frequency", type=float, default=None)
    parser.add_argument(
        "--controller",
        choices=["xbox", "network"],
        default=os.environ.get("DUCK_CONTROLLER", "xbox"),
        help="xbox: Bluetooth gamepad on the robot; network: keyboard on a laptop over UDP",
    )

    args = parser.parse_args()
    os.environ["DUCK_CONTROLLER"] = args.controller
    pid = [args.p, args.i, args.d]

    print("Done parsing args")
    rl_walk = RLWalk(
        args.onnx_model_path,
        duck_config_path=args.duck_config_path,
        action_scale=args.action_scale,
        pid=pid,
        control_freq=args.control_freq,
        commands=args.commands,
        pitch_bias=args.pitch_bias,
        save_obs=args.save_obs,
        replay_obs=args.replay_obs,
        cutoff_frequency=args.cutoff_frequency,
    )
    print("Done instantiating RLWalk")
    rl_walk.run()
