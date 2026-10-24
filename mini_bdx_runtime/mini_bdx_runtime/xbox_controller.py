import os
import pygame
from pygame._sdl2 import controller as sdl_controller
from threading import Thread
from queue import Queue
import time
import numpy as np
from mini_bdx_runtime.buttons import Buttons


X_RANGE = [-0.15, 0.15]
Y_RANGE = [-0.2, 0.2]
YAW_RANGE = [-1.0, 1.0]

# rads
NECK_PITCH_RANGE = [-0.34, 1.1]
HEAD_PITCH_RANGE = [-0.78, 0.3]
HEAD_YAW_RANGE = [-0.5, 0.5]
HEAD_ROLL_RANGE = [-0.5, 0.5]

# Raw joystick indices differ between drivers: the Xbox One S over Bluetooth with hid-generic
# (Radxa) gives A0 B1 X2 Y3 LB4 RB5 and axes LX LY LT RX RY RT, while the layout below (X3 Y4 LB6
# RB7, axes LX LY RX RY RT LT) is what the original Pi setup reported. SDL's game controller API
# maps any controller it knows to the same Xbox layout, so use it when the controller is known.
RAW_AXES = {"lx": 0, "ly": 1, "rx": 2, "ry": 3, "rt": 4, "lt": 5}
RAW_BUTTONS = {"a": 0, "b": 1, "x": 3, "y": 4, "lb": 6, "rb": 7}


class _Pad:
    """Xbox-style reading of the first controller: sticks in [-1, 1], triggers in [-1, 1]."""

    def __init__(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        sdl_controller.init()
        self.c = None
        for i in range(pygame.joystick.get_count()):
            if sdl_controller.is_controller(i):
                self.c = sdl_controller.Controller(i)
                print(f"Loaded controller {self.c.name} (SDL mapping).")
                return
        self.j = pygame.joystick.Joystick(0)
        self.j.init()
        print(f"Loaded joystick {self.j.get_name()} with {self.j.get_numaxes()} axes (raw mapping).")

    def axis(self, name):
        if self.c is None:
            return self.j.get_axis(RAW_AXES[name])
        a = {
            "lx": pygame.CONTROLLER_AXIS_LEFTX,
            "ly": pygame.CONTROLLER_AXIS_LEFTY,
            "rx": pygame.CONTROLLER_AXIS_RIGHTX,
            "ry": pygame.CONTROLLER_AXIS_RIGHTY,
            "lt": pygame.CONTROLLER_AXIS_TRIGGERLEFT,
            "rt": pygame.CONTROLLER_AXIS_TRIGGERRIGHT,
        }[name]
        v = self.c.get_axis(a) / 32767.0
        if name in ("lt", "rt"):
            return v * 2 - 1  # SDL triggers are 0..1; the raw ones rest at -1
        return max(-1.0, min(1.0, v))

    def button(self, name):
        if self.c is None:
            return bool(self.j.get_button(RAW_BUTTONS[name]))
        b = {
            "a": pygame.CONTROLLER_BUTTON_A,
            "b": pygame.CONTROLLER_BUTTON_B,
            "x": pygame.CONTROLLER_BUTTON_X,
            "y": pygame.CONTROLLER_BUTTON_Y,
            "lb": pygame.CONTROLLER_BUTTON_LEFTSHOULDER,
            "rb": pygame.CONTROLLER_BUTTON_RIGHTSHOULDER,
        }[name]
        return bool(self.c.get_button(b))

    def up_down(self):
        if self.c is None:
            return self.j.get_hat(0)[1]
        return int(self.c.get_button(pygame.CONTROLLER_BUTTON_DPAD_UP)) - int(
            self.c.get_button(pygame.CONTROLLER_BUTTON_DPAD_DOWN)
        )


class XBoxController:
    def __init__(self, command_freq, only_head_control=False):
        self.command_freq = command_freq
        self.head_control_mode = only_head_control
        self.only_head_control = only_head_control

        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.last_left_trigger = 0.0
        self.last_right_trigger = 0.0
        self.p1 = _Pad()
        self.cmd_queue = Queue(maxsize=1)

        self.A_pressed = False
        self.B_pressed = False
        self.X_pressed = False
        self.Y_pressed = False
        self.LB_pressed = False
        self.RB_pressed = False

        self.buttons = Buttons()

        Thread(target=self.commands_worker, daemon=True).start()

    def commands_worker(self):
        while True:
            self.cmd_queue.put(self.get_commands())
            time.sleep(1 / self.command_freq)

    def get_commands(self):
        last_commands = self.last_commands
        left_trigger = self.last_left_trigger
        right_trigger = self.last_right_trigger

        pygame.event.pump()  # process event queue
        l_x = -1 * self.p1.axis("lx")
        l_y = -1 * self.p1.axis("ly")
        r_x = -1 * self.p1.axis("rx")
        r_y = -1 * self.p1.axis("ry")

        right_trigger = np.around((self.p1.axis("rt") + 1) / 2, 3)
        left_trigger = np.around((self.p1.axis("lt") + 1) / 2, 3)

        if left_trigger < 0.1:
            left_trigger = 0
        if right_trigger < 0.1:
            right_trigger = 0

        if not self.head_control_mode:
            lin_vel_y = l_x
            lin_vel_x = l_y
            ang_vel = r_x
            if lin_vel_x >= 0:
                lin_vel_x *= np.abs(X_RANGE[1])
            else:
                lin_vel_x *= np.abs(X_RANGE[0])

            if lin_vel_y >= 0:
                lin_vel_y *= np.abs(Y_RANGE[1])
            else:
                lin_vel_y *= np.abs(Y_RANGE[0])

            if ang_vel >= 0:
                ang_vel *= np.abs(YAW_RANGE[1])
            else:
                ang_vel *= np.abs(YAW_RANGE[0])

            last_commands[0] = lin_vel_x
            last_commands[1] = lin_vel_y
            last_commands[2] = ang_vel
        else:
            last_commands[0] = 0.0
            last_commands[1] = 0.0
            last_commands[2] = 0.0
            last_commands[3] = 0.0  # neck pitch 0 for now

            head_yaw = l_x
            head_pitch = l_y
            head_roll = r_x

            if head_yaw >= 0:
                head_yaw *= np.abs(HEAD_YAW_RANGE[0])
            else:
                head_yaw *= np.abs(HEAD_YAW_RANGE[1])

            if head_pitch >= 0:
                head_pitch *= np.abs(HEAD_PITCH_RANGE[0])
            else:
                head_pitch *= np.abs(HEAD_PITCH_RANGE[1])

            if head_roll >= 0:
                head_roll *= np.abs(HEAD_ROLL_RANGE[0])
            else:
                head_roll *= np.abs(HEAD_ROLL_RANGE[1])

            last_commands[4] = head_pitch
            last_commands[5] = head_yaw
            last_commands[6] = head_roll

        # held state of each button; Y toggles head control on the press only
        y_was = self.Y_pressed
        self.A_pressed = self.p1.button("a")
        self.B_pressed = self.p1.button("b")
        self.X_pressed = self.p1.button("x")
        self.Y_pressed = self.p1.button("y")
        self.LB_pressed = self.p1.button("lb")
        self.RB_pressed = self.p1.button("rb")
        if self.Y_pressed and not y_was and not self.only_head_control:
            self.head_control_mode = not self.head_control_mode

        up_down = self.p1.up_down()

        return (
            np.around(last_commands, 3),
            self.A_pressed,
            self.B_pressed,
            self.X_pressed,
            self.Y_pressed,
            self.LB_pressed,
            self.RB_pressed,
            left_trigger,
            right_trigger,
            up_down,
        )

    def get_last_command(self):
        A_pressed = False
        B_pressed = False
        X_pressed = False
        Y_pressed = False
        LB_pressed = False
        RB_pressed = False
        up_down = 0
        try:
            (
                self.last_commands,
                A_pressed,
                B_pressed,
                X_pressed,
                Y_pressed,
                LB_pressed,
                RB_pressed,
                self.last_left_trigger,
                self.last_right_trigger,
                up_down,
            ) = self.cmd_queue.get(
                False
            )  # non blocking
        except Exception:
            pass

        self.buttons.update(
            A_pressed,
            B_pressed,
            X_pressed,
            Y_pressed,
            LB_pressed,
            RB_pressed,
            up_down == 1,
            up_down == -1,
        )

        return (
            self.last_commands,
            self.buttons,
            self.last_left_trigger,
            self.last_right_trigger,
        )

if __name__ == "__main__":
    controller = XBoxController(20)

    while True:
        print(controller.get_last_command())
        time.sleep(0.05)
