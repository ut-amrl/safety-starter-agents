#!/usr/bin/env python

import joblib
from statistics import mean, stdev
from matplotlib import pyplot as plt


def graph_dynamics(dynamic='velocity'):
    if dynamic == 'velocity':
        nn_path = '/Users/JoshHoffman/ut_austin/bayes_ps/safety-starter-agents/data/shifted_dynamics/shift_data_velocity_point_2020-09-03.pkl'
        tree_path = '/Users/JoshHoffman/ut_austin/bayes_ps/safety-starter-agents/data/shifted_dynamics/shifted_tree_data/shift_tree_data_velocity_point_2020-09-13.pkl'
    elif dynamic == 'motor':
        nn_path = '/Users/JoshHoffman/ut_austin/bayes_ps/safety-starter-agents/data/shifted_dynamics/shift_data_motor_point_2020-08-29.pkl'
        tree_path = '/Users/JoshHoffman/ut_austin/bayes_ps/safety-starter-agents/data/shifted_dynamics/shifted_tree_data/shift_tree_data_motor_point_2020-09-13.pkl'
    elif dynamic == 'pointarrow':
        nn_path = '/Users/JoshHoffman/ut_austin/bayes_ps/safety-starter-agents/data/shifted_dynamics/shift_data_pointarrow_point_2020-09-04.pkl'
        tree_path = '/Users/JoshHoffman/ut_austin/bayes_ps/safety-starter-agents/data/shifted_dynamics/shifted_tree_data/shift_tree_data_pointarrow_point_2020-09-13.pkl'
    elif dynamic == 'robot':
        nn_path = '/Users/JoshHoffman/ut_austin/bayes_ps/safety-starter-agents/data/shifted_dynamics/shift_data_robot_point_2020-09-04.pkl'
        tree_path = '/Users/JoshHoffman/ut_austin/bayes_ps/safety-starter-agents/data/shifted_dynamics/shifted_tree_data/shift_tree_data_robot_point_2020-09-13.pkl'
    nn_data = joblib.load(nn_path)
    tree_data = joblib.load(tree_path)

    sorted_nn_keys = sorted(nn_data.keys())
    sorted_tree_keys = sorted(tree_data.keys())
    # Get x values
    xnn = [i for i in sorted_nn_keys]
    xtree = [i for i in sorted_tree_keys]
    # Get y values
    ynn = [mean(nn_data[i]) for i in sorted_nn_keys]
    ytree = [mean(tree_data[i]) for i in sorted_tree_keys]

    # Get stdev values - For NN
    nn_error = [stdev(nn_data[i]) for i in sorted_nn_keys]
    nn_lower_bound = [ynn[i] - nn_error[i] for i in range(len(ynn))]
    nn_upper_bound = [ynn[i] + nn_error[i] for i in range(len(ynn))]
    # For Tree form
    tree_error = [stdev(tree_data[i]) for i in sorted_tree_keys]
    tree_lower_bound = [ytree[i] - tree_error[i] for i in range(len(ytree))]
    tree_upper_bound = [ytree[i] + tree_error[i] for i in range(len(ytree))]

    if dynamic == 'velocity':
        xs = []
        for i in sorted_tree_keys:
            mv = mean(tree_data[i])
            if mv >= -10.0:
                xs.append(i)
        print("The min x is:", min(xs))
        print("The max x is:", max(xs))
        greater_th = [1 for i in sorted_tree_keys if (i in nn_data.keys()) and tree_data[i] >= nn_data[i]]
        print("Under acceptable episodic returns, the tree performs better:", str(sum(greater_th) / len(xs)))
    elif dynamic == 'robot':
        xs = []
        for i in sorted_tree_keys:
            mv = mean(tree_data[i])
            if mv >= 3.0:
                xs.append(i)
        print("The min x is:", min(xs))
        print("The max x is:", max(xs))
        greater_th = [1 for i in sorted_tree_keys if tree_data[i] >= nn_data[i]]
        print("Under acceptable episodic returns, the tree performs better:", str(sum(greater_th) / len(xs)))

    plt.plot(xnn, ynn, '-b', label="NN")
    plt.fill_between(xnn, nn_lower_bound, nn_upper_bound, facecolor='blue', alpha=0.5)
    plt.plot(xtree, ytree, '-g', label="Tree")
    plt.fill_between(xtree, tree_lower_bound, tree_upper_bound, facecolor='green', alpha=0.5)
    plt.xlabel('Value for Dynamic')
    plt.ylabel('Average Episodic Returns')
    title = 'Average Episodic Returns when changing the ' + str(dynamic) + ' dynamic'
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dynamic', '-dn', type=str, default="")
    args = parser.parse_args()

    if args.dynamic == "":
        graph_dynamics()
    else:
        graph_dynamics(args.dynamic)
