#!/usr/bin/env python

from domain_adaptation import get_range_vals, shift_dynamics
import experiment


def generate_nn_in_d2(robot, task, algo, seed, cpu, dynamic, xml_file):
    assert dynamic != "", "You did not specify a dynamic"
    forcerange_vals = get_range_vals(dynamic)
    for x_val in forcerange_vals:
        shift_dynamics(dynamic, xml_file, x_val)
        exp_name = dynamic + '_' + str(x_val) + '_' + str(algo) + '_' + robot.lower() + task.lower()
        try:
            experiment.main(robot, task, algo, seed, exp_name, cpu)
        except:
            print("hopefully killed processes")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='Point')
    parser.add_argument('--task', type=str, default='Goal1')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--dynamic', '-dn', type=str, default="")
    parser.add_argument('--xml_file', '-xf', type=str, default="")
    args = parser.parse_args()
    generate_nn_in_d2(args.robot, args.task, args.algo, args.seed, args.cpu, args.dynamic, args.xml_file)
