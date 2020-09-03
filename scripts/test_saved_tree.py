#!/usr/bin/env python

import joblib
from safe_rl.utils.load_utils import load_policy
from sklearn.tree import DecisionTreeRegressor
import argparse
from generate_tree import label_data


def test_tree(env_path, tree_pickle, itr, deterministic, len, episodes, render=False):
    env, _, _ = load_policy(env_path, itr, deterministic)
    tree_data = joblib.load(tree_pickle)
    tree_program = tree_data['tree']
    label_data(env, lambda a: tree_program.predict([a]), len, episodes, render)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('pickle_name', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--render', '-nr', action='store_true')
    args = parser.parse_args()
    test_tree(args.fpath,
              args.pickle_name,
              args.itr if args.itr >= 0 else 'last',
              args.deterministic,
              args.len,
              args.episodes,
              args.render)
