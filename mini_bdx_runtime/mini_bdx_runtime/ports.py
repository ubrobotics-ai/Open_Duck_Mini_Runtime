"""Serial port of the servo bus.

Kept apart from pins.py so that it can be imported on a laptop (servo ID
programming) where Adafruit Blinka has no board to drive.

Override with the DUCK_SERIAL_PORT environment variable. On the Radxa the udev
rule shipped in the UBR provisioning kit creates /dev/duck_servos.
"""

import os

DEFAULT_SERIAL_PORT = "/dev/ttyACM0"
SERIAL_PORT = os.environ.get("DUCK_SERIAL_PORT", DEFAULT_SERIAL_PORT)
