#!/usr/bin/env python
import gym
import safety_gym
import safe_rl
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork
from run_existing_agent import run_polopt_agent
import sys


def main(robot, task, algo, seed, exp_name, cpu, fpath, itr, deterministic, hid, lhm, len, cost_lim):

    # Verify experiment
    robot_list = ['point', 'car', 'doggo', 'point_shift_extreme']
    task_list = ['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']
    algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']

    algo = algo.lower()
    task = task.capitalize()
    robot = robot.capitalize()
    assert algo in algo_list, "Invalid algo"
    assert task.lower() in task_list, "Invalid task"
    assert robot.lower() in robot_list, "Invalid robot"

    # Hyperparameters
    if exp_name is None:
        exp_name = (algo + '_' + robot.lower() + task.lower())
    if robot == 'Doggo':
        num_steps = 1e8
        steps_per_epoch = 60000
    else:
        num_steps = 1e7
        steps_per_epoch = 30000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01
    cost_lim = 25

    # Fork for parallelizing
    mpi_fork(cpu)

    # Prepare Logger
    logger_kwargs = setup_logger_kwargs(exp_name, seed)

    run_polopt_agent(fpath,
                     itr,
                     deterministic,
                     ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                     seed=seed,
                     # Experience collection:
                     steps_per_epoch=steps_per_epoch,
                     epochs=epochs,
                     max_ep_len=len,
                     cost_lim=cost_lim,
                     target_kl=target_kl,
                     # Logging:
                     logger_kwargs=logger_kwargs,
                     save_freq=save_freq
                     )
    sys.exit()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--robot', type=str, default='Point')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--hidd_mult', type=int, default=2)
    parser.add_argument('--len', type=int, default=1000)
    parser.add_argument('--cost_lim', type=float, default=10)
    args = parser.parse_args()
    exp_name = args.exp_name if not(args.exp_name == '') else None
    fpath = args.fpath
    itr = args.itr
    deterministic = args.deterministic
    hid = args.hid
    lhm = args.hidd_mult
    len = args.len
    cost_lim = args.cost_lim
    main(args.robot, args.task, args.algo, args.seed, exp_name, args.cpu, fpath, itr, deterministic, hid, lhm, len, cost_lim, args)
