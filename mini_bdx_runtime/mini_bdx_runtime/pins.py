"""GPIO pin map, per board.

The upstream runtime targets the Raspberry Pi Zero 2 W. The UB Robotics
"Pato Robo" uses a Radxa Zero 3W, whose 40-pin header has the same physical
layout but different GPIO names. Each signal keeps the same *physical* pin on
both boards, so the wiring diagram of the Open Duck Mini v2 still applies.

| Signal        | Phys. pin | Raspberry Pi | Radxa Zero 3W |
|---------------|-----------|--------------|---------------|
| left foot     | 15        | GPIO22       | GPIO3_B1      |
| right foot    | 13        | GPIO27       | GPIO3_B0      |
| left eye      | 16        | GPIO24       | GPIO3_B2      |
| right eye     | 18        | GPIO23       | GPIO4_C3      |
| projector     | 22        | GPIO25       | GPIO3_C1      |
| left antenna  | 33        | GPIO13 (PWM) | none (no PWM) |
| right antenna | 32        | GPIO12 (PWM) | none (no PWM) |
| IMU SDA / SCL | 3 / 5     | GPIO2 / 3    | GPIO1_A0 / A1 (I2C3-M0 overlay) |

Pins set to None are not available on that board; the matching feature must
stay disabled in duck_config.json.
"""

import board

if hasattr(board, "D3_B1"):  # Radxa Zero 3 (3W / 3E)
    BOARD_NAME = "radxa_zero3"
    LEFT_FOOT_PIN = board.D3_B1
    RIGHT_FOOT_PIN = board.D3_B0
    LEFT_EYE_PIN = board.D3_B2
    RIGHT_EYE_PIN = board.D4_C3
    PROJECTOR_PIN = board.D3_C1
    LEFT_ANTENNA_PIN = None
    RIGHT_ANTENNA_PIN = None
elif hasattr(board, "D22"):  # Raspberry Pi (upstream)
    BOARD_NAME = "raspberry_pi"
    LEFT_FOOT_PIN = board.D22
    RIGHT_FOOT_PIN = board.D27
    LEFT_EYE_PIN = board.D24
    RIGHT_EYE_PIN = board.D23
    PROJECTOR_PIN = board.D25
    LEFT_ANTENNA_PIN = board.D13
    RIGHT_ANTENNA_PIN = board.D12
else:
    raise RuntimeError(
        "Unknown board: no pin map for it in mini_bdx_runtime/pins.py"
    )


def require(pin, feature):
    if pin is None:
        raise RuntimeError(
            f"{feature} is not available on {BOARD_NAME}; "
            f"disable it in duck_config.json"
        )
    return pin
