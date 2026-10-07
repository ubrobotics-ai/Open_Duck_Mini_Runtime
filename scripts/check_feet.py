"""Print the state of both foot switches, 20 times per second.

Press each foot on the table: its value must go to True, and back to False
when lifted. Ctrl-C to stop.
"""

import time

from mini_bdx_runtime.feet_contacts import FeetContacts
from mini_bdx_runtime.pins import BOARD_NAME

feet = FeetContacts()
print(f"board: {BOARD_NAME}  (left, right)")
last = None
try:
    while True:
        state = tuple(feet.get())
        if state != last:
            print(f"left={state[0]!s:5}  right={state[1]!s:5}")
            last = state
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    feet.stop()
