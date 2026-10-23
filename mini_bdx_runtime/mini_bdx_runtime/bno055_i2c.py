"""Open the BNO055 on I2C at 0x28 or 0x29.

The Adafruit board answers at 0x28; other modules (e.g. Botnroll SEN16002)
may have the ADR pin high and answer at 0x29. DUCK_IMU_ADDR forces one.

The library waits 0.7 s after resetting the chip. The Botnroll module on a
Radxa Zero 3W needs about 0.8 s and NACKs the next write (Errno 6), so the
reset waits a little longer here, and opening is retried: the first
attempt after the duck boots still fails now and then.
"""

import os
import time

import adafruit_bno055
import board
import busio


RESET_EXTRA_WAIT = 0.5  # s, on top of the library's 0.7 s
ATTEMPTS = 3


class _BNO055_I2C(adafruit_bno055.BNO055_I2C):
    def _reset(self):
        super()._reset()
        time.sleep(RESET_EXTRA_WAIT)


def open_bno055():
    i2c = busio.I2C(board.SCL, board.SDA)
    forced = os.environ.get("DUCK_IMU_ADDR")
    addresses = [int(forced, 0)] if forced else [0x28, 0x29]
    errors = {}
    for attempt in range(ATTEMPTS):
        for addr in addresses:
            try:
                return _BNO055_I2C(i2c, address=addr)
            except (ValueError, OSError, RuntimeError) as err:
                errors[addr] = err
        time.sleep(1.0)
    last = "; ".join(f"{hex(a)}: {e}" for a, e in errors.items())
    raise RuntimeError(
        f"BNO055 not found on I2C at {', '.join(hex(a) for a in addresses)} "
        f"(check wiring, the I2C overlay and `i2cdetect -y 3`): {last}"
    )
