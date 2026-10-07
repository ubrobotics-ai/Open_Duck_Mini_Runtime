"""Walking commands over the network, as a drop-in for XBoxController.

A laptop sends UDP datagrams (JSON) to the robot; this class exposes them through
the same get_last_command() interface as XBoxController, so v2_rl_walk_mujoco.py
does not change. Select it with DUCK_CONTROLLER=network (or --controller network).

Datagram (protocol version 1), sent ~20 times per second:
    {"v": 1, "seq": 42,
     "cmd": [lin_vel_x, lin_vel_y, ang_vel, neck_pitch, head_pitch, head_yaw, head_roll],
     "buttons": {"A": false, "B": false, "X": false, "Y": false,
                 "LB": false, "RB": false, "up": false, "down": false},
     "lt": 0.0, "rt": 0.0}

Each datagram is answered with {"v": 1, "ack": seq} so the sender can show that the
link is up. Commands are clamped to the gamepad ranges. If nothing arrives for
TIMEOUT seconds, commands drop to zero and buttons are released: the duck keeps
balancing in place instead of walking off with the last command.
"""

import json
import os
import socket
import time
from threading import Lock, Thread

from mini_bdx_runtime.buttons import Buttons

PROTOCOL_VERSION = 1
DEFAULT_PORT = 5005
TIMEOUT = 0.5  # s without datagrams before stopping

# same limits as XBoxController
X_RANGE = [-0.15, 0.15]
Y_RANGE = [-0.2, 0.2]
YAW_RANGE = [-1.0, 1.0]
NECK_PITCH_RANGE = [-0.34, 1.1]
HEAD_PITCH_RANGE = [-0.78, 0.3]
HEAD_YAW_RANGE = [-0.5, 0.5]
HEAD_ROLL_RANGE = [-0.5, 0.5]
RANGES = [X_RANGE, Y_RANGE, YAW_RANGE, NECK_PITCH_RANGE,
          HEAD_PITCH_RANGE, HEAD_YAW_RANGE, HEAD_ROLL_RANGE]
BUTTON_NAMES = ["A", "B", "X", "Y", "LB", "RB", "up", "down"]


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


class NetworkController:
    def __init__(self, command_freq=20, port=None, timeout=TIMEOUT, verbose=True):
        self.command_freq = command_freq
        self.timeout = timeout
        self.port = int(port or os.environ.get("DUCK_CONTROL_PORT", DEFAULT_PORT))
        self.buttons = Buttons()

        self._lock = Lock()
        self._commands = [0.0] * 7
        self._pressed = {name: False for name in BUTTON_NAMES}
        self._lt = 0.0
        self._rt = 0.0
        self._last_rx = 0.0
        self._peer = None
        self._was_connected = False
        self.verbose = verbose

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.settimeout(0.2)
        self._running = True
        Thread(target=self._receive, daemon=True).start()
        if self.verbose:
            print(f"Network controller listening on UDP {self.port}")

    def _receive(self):
        while self._running:
            try:
                data, addr = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode())
                if msg.get("v") != PROTOCOL_VERSION:
                    continue
                cmd = [float(x) for x in msg.get("cmd", [])][:7]
                cmd += [0.0] * (7 - len(cmd))
                cmd = [_clamp(c, lo, hi) for c, (lo, hi) in zip(cmd, RANGES)]
                buttons = msg.get("buttons", {})
                pressed = {name: bool(buttons.get(name, False)) for name in BUTTON_NAMES}
                lt = _clamp(float(msg.get("lt", 0.0)), 0.0, 1.0)
                rt = _clamp(float(msg.get("rt", 0.0)), 0.0, 1.0)
            except (ValueError, TypeError, AttributeError):
                continue
            with self._lock:
                self._commands = cmd
                self._pressed = pressed
                self._lt, self._rt = lt, rt
                self._last_rx = time.time()
                self._peer = addr
            try:
                self.sock.sendto(json.dumps({"v": PROTOCOL_VERSION, "ack": msg.get("seq")}).encode(), addr)
            except OSError:
                pass

    def connected(self):
        return time.time() - self._last_rx < self.timeout

    def get_last_command(self):
        with self._lock:
            fresh = time.time() - self._last_rx < self.timeout
            if fresh:
                commands = list(self._commands)
                pressed = dict(self._pressed)
                lt, rt = self._lt, self._rt
            else:
                commands = [0.0] * 7
                pressed = {name: False for name in BUTTON_NAMES}
                lt = rt = 0.0
        if self.verbose and fresh != self._was_connected:
            print(f"Network controller: {'link up from ' + str(self._peer[0]) if fresh else 'link lost, stopping'}")
        self._was_connected = fresh

        self.buttons.update(
            pressed["A"], pressed["B"], pressed["X"], pressed["Y"],
            pressed["LB"], pressed["RB"], pressed["up"], pressed["down"],
        )
        return commands, self.buttons, lt, rt

    def stop(self):
        self._running = False
        self.sock.close()


if __name__ == "__main__":
    c = NetworkController()
    while True:
        cmds, b, lt, rt = c.get_last_command()
        print(cmds[:3], "A" if b.A.is_pressed else "-", "LB" if b.LB.is_pressed else "--")
        time.sleep(0.1)
