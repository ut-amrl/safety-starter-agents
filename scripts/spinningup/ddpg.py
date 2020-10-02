#!/usr/bin/env python

import safety_gym
import gym
from spinup import ddpg_tf1 as ddpg
from safe_rl.utils.run_utils import setup_logger_kwargs
from datetime import date


def ddpg_wrapper(exp_name='', seed=10):
    if exp_name == '':
        today = date.today()
        date_fmt = today.strftime("%b_%d_%Y")
        exp_name = (date_fmt.lower() + '_ddpg' + '_point_button1')
    print('The experiment name is:', exp_name)
    print('The seed is:', seed)

    num_steps = 1e7
    steps_per_epoch = 30000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    logger_kwargs = setup_logger_kwargs(exp_name, seed)
    ddpg(lambda: gym.make('Safexp-PointButton1-v0'),
         steps_per_epoch=steps_per_epoch,
         epochs=epochs,
         logger_kwargs=logger_kwargs,
         save_freq=save_freq
         )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()
    ddpg_wrapper(args.exp_name, args.seed)
