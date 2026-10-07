"""Open the BNO055 on I2C at 0x28 or 0x29.

The Adafruit board answers at 0x28; other modules (e.g. Botnroll SEN16002)
may have the ADR pin high and answer at 0x29. DUCK_IMU_ADDR forces one.
"""

import os

import adafruit_bno055
import board
import busio


def open_bno055():
    i2c = busio.I2C(board.SCL, board.SDA)
    forced = os.environ.get("DUCK_IMU_ADDR")
    addresses = [int(forced, 0)] if forced else [0x28, 0x29]
    last = None
    for addr in addresses:
        try:
            return adafruit_bno055.BNO055_I2C(i2c, address=addr)
        except (ValueError, OSError, RuntimeError) as err:
            last = err
    raise RuntimeError(
        f"BNO055 not found on I2C at {', '.join(hex(a) for a in addresses)} "
        f"(check wiring, the I2C overlay and `i2cdetect -y 3`): {last}"
    )
