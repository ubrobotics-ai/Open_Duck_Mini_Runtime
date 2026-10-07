# Running on a Radxa Zero 3W

Branch `radxa-zero3w` of the UB Robotics fork. It changes as little as possible
so that upstream `v2` can still be merged in.

## What changes against the Raspberry Pi Zero 2 W

- **GPIO pins** live in `mini_bdx_runtime/pins.py`, chosen by board. Every
  signal keeps the same *physical* header pin, so the upstream wiring diagram
  still applies. The Radxa has no PWM on pins 32/33: antennas must stay
  disabled in `duck_config.json`.
- **Serial port** of the servo bus lives in `mini_bdx_runtime/ports.py`. Set
  `DUCK_SERIAL_PORT` (the UBR provisioning kit sets `/dev/duck_servos` through
  a udev rule). Default stays `/dev/ttyACM0`.
- **I2C (IMU)**: enable the `I2C3-M0` overlay (`rsetup` → Overlays → Manage
  overlays). SDA is pin 3, SCL pin 5, as on the Pi. Blinka picks the bus.
- **USB**: the servo driver is plugged into the OTG port, which must be in
  host mode.
- **OS**: Debian Bookworm (Python 3.11). Bullseye's Python 3.9 has no wheels
  for `rustypot` or `onnxruntime==1.18.1`.
- **IMU address**: `mini_bdx_runtime/bno055_i2c.py` tries 0x28 then 0x29 (some
  BNO055 modules have ADR high). `DUCK_IMU_ADDR=0x29` forces one.
- New script `scripts/check_feet.py` to test the foot switches.

Nothing else changes: the policy, the servo IDs (10–14, 20–24, 30–33), the
offsets procedure and `v2_rl_walk_mujoco.py` are upstream's.
