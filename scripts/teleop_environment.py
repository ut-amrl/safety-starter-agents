#!/usr/bin/env python
from test_policy import run_policy
from teleop_twist_keyboard_gym import close_io, get_action_closure
from safe_rl.utils.load_utils import load_policy
import termios
import sys

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, _, _ = load_policy(args.fpath,
                            args.itr if args.itr >= 0 else 'last',
                            args.deterministic)
    settings = termios.tcgetattr(sys.stdin)
    get_action = get_action_closure(settings)
    try:
        run_policy(env, get_action, args.len, args.episodes, True)
    except Exception as e:
        print(e)
    finally:
        close_io(settings)
