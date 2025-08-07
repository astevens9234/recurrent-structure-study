"""Script to log mouse activity, future training data.

c.f. <https://pynput.readthedocs.io/en/latest/mouse.html>
"""

import csv
import sys
import time

from pynput import mouse
from datetime import datetime as dt

file = input("Enter filename: ") + ".csv"
cols = ["ts", "event", "x", "y", "button", "action"]
last_move_time = 0
# interval = 0.05  # 50ms
interval = 0.1  # 100ms

##### NOTE #####
# features: acceleration    ax = dvx/dt & ay = dvy/dt
#           magnitude       sqrt(ax^2 + ay^2)
################

def write_to_csv(event, x, y, button=None, action=None):
    with open(file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writerow(
            {
                "ts": dt.now().timestamp(),
                "event": event,
                "x": x,
                "y": y,
                "button": button if button else "",
                "action": action if action else "",
            }
        )


def on_move(x, y):
    global last_move_time
    ct = time.time()
    if ct - last_move_time >= interval:
        write_to_csv("move", x, y)


def on_click(x, y, button, pressed):
    action = "press" if pressed else "released"
    write_to_csv("click", x, y, str(button), action)


def on_scroll(x, y, dx, dy):
    write_to_csv("scroll", x, y, f"delta({dx},{dy})")  # NOTE: format


def main():
    with open(file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

    with mouse.Listener(
        on_move=on_move, on_click=on_click, on_scroll=on_scroll
    ) as listener:
        print("mouse monitoring started. Keyboard interupt to stop.")
        try:
            listener.join()
        except KeyboardInterrupt:
            print("stopped")
            sys.exit(0)


if __name__ == "__main__":
    main()
