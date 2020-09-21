#!/usr/bin/env python

from __future__ import print_function
import tty
import termios
import select
import sys
import numpy as np


# import threading


msg = """
Reading from the keyboard  and Publishing to Twist!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .
For Holonomic mode (strafing), hold down the shift key:
---------------------------
   U    I    O
   J    K    L
   M    <    >
t : up (+z)
b : down (-z)
anything else : stop
q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%
CTRL-C to quit
"""


def getKey(key_timeout, settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], key_timeout)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def vels(speed, turn):
    return "currently:\tspeed %s\tturn %s " % (speed, turn)


def get_action_closure(settings):
    # nonlocal moveBindings, speedBindings
    # settings = termios.tcgetattr(sys.stdin)

    speed = 0.5
    turn = 1.0
    key_timeout = 0.0
    if key_timeout == 0.0:
        key_timeout = None

    x = 0
    y = 0
    th = 0
    status = 0

    moveBindings = {
            'i': (1, 0, 0, 0),
            'o': (1, 0, 0, -1),
            'j': (0, 0, 0, 1),
            'l': (0, 0, 0, -1),
            'u': (1, 0, 0, 1),
            ',': (-1, 0, 0, 0),
            '.': (-1, 0, 0, 1),
            'm': (-1, 0, 0, -1),
            'O': (1, -1, 0, 0),
            'I': (1, 0, 0, 0),
            'J': (0, 1, 0, 0),
            'L': (0, -1, 0, 0),
            'U': (1, 1, 0, 0),
            '<': (-1, 0, 0, 0),
            '>': (-1, -1, 0, 0),
            'M': (-1, 1, 0, 0),
            't': (0, 0, 1, 0),
            'b': (0, 0, -1, 0),
        }

    speedBindings = {
            'q': (1.1, 1.1),
            'z': (.9, .9),
            'w': (1.1, 1),
            'x': (.9, 1),
            'e': (1, 1.1),
            'c': (1, .9),
        }
    print(msg)
    print(vels(speed, turn))

    def get_action(obs):
        nonlocal turn, speed, status, x, y, th, settings, speedBindings, moveBindings
        try:
            key = None
            while key is None:
                key = getKey(key_timeout, settings)
            if (key == '\x03'):
                raise KeyboardInterrupt("User killed the program")
            if key in moveBindings.keys():
                x = moveBindings[key][0]
                th = moveBindings[key][3]
                return get_action_box(x*speed, th * turn)
            elif key in speedBindings.keys():
                speed = speed * speedBindings[key][0]
                turn = turn * speedBindings[key][1]
                print(vels(speed, turn))
                if (status == 14):
                    print(msg)
                status = (status + 1) % 15
            else:
                # Skip updating cmd_vel if key timeout and robot already
                # stopped.
                x = 0
                y = 0
                th = 0
            return get_action_box(0, 0)
        except Exception as e:
            print(e)
        # return np.array([0, 0])
    return get_action


def dummy_controller(obs):
    return np.array([0, 0])


def get_action_box(speed, turn):
    v = speed / 10
    th = turn / 20
    if v == 0:
        v = 0.001
    # print("The th is:", th)
    return np.array((v, th))


def close_io(settings):
    return termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


if __name__ == "__main__":
    settings = termios.tcgetattr(sys.stdin)
    get_action = get_action_closure(settings)
    try:
        while(1):
            print(get_action(1))
    except Exception as e:
        print(e)
    finally:
        close_io(settings)
