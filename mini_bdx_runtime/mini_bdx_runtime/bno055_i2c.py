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


_AXES = {"X": adafruit_bno055.AXIS_REMAP_X, "Y": adafruit_bno055.AXIS_REMAP_Y, "Z": adafruit_bno055.AXIS_REMAP_Z}
_SIGNS = {"+": adafruit_bno055.AXIS_REMAP_POSITIVE, "-": adafruit_bno055.AXIS_REMAP_NEGATIVE}


def axis_remap_from_config(spec):
    """Turn ["Y", "X", "Z", "+", "-", "+"] into a BNO055 axis_remap tuple.

    The first three name the sensor axis that becomes the duck's X, Y and Z
    (X forward, Y left, Z up, as the imu site in the MJCF); the last three are
    their signs. Example: the UBR adapter under the trunk_top tab has the
    sensor's y forward and x to the right, so ["Y", "X", "Z", "+", "-", "+"].
    """
    if len(spec) != 6 or sorted(spec[:3]) != ["X", "Y", "Z"] or any(v not in _SIGNS for v in spec[3:]):
        raise ValueError(f"imu_axis_remap must look like [\"Y\", \"X\", \"Z\", \"+\", \"-\", \"+\"], got {spec}")
    return tuple(_AXES[a] for a in spec[:3]) + tuple(_SIGNS[v] for v in spec[3:])


def apply_accel_offsets(imu, offsets, run_mode):
    """Write accelerometer offsets (m/s²) in the chip's output axes.

    The chip applies them after its axis remap, so with imu_axis_remap set they
    are in the duck's frame (X forward, Y left, Z up). It subtracts them in its
    fusion modes (NDOF, IMUPLUS), not in the raw ACCGYRO mode, and only accepts
    ±5.0 m/s² per axis. Measure them with the duck still in a few poses: the
    offset is the reading minus the gravity it should see.
    Checked on a Zero 3W, 23 Oct 2026: 2.0 on X lowered the forward reading by 2.
    """
    regs = tuple(int(round(v * 100)) for v in offsets)
    if len(regs) != 3 or any(abs(r) > 500 for r in regs):
        raise ValueError(f"imu_accel_offsets must be 3 values within ±5.0 m/s², got {offsets}")
    imu.mode = adafruit_bno055.CONFIG_MODE
    time.sleep(0.05)
    imu.offsets_accelerometer = regs
    imu.mode = run_mode
    time.sleep(0.05)
