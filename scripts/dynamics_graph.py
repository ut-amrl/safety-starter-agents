#!/usr/bin/env python

import joblib
from statistics import mean, stdev
from matplotlib import pyplot as plt


def graph_dynamics(data_object):
    data = joblib.load(data_object)

    # Print out each x and associated mean value that still has relatively
    # high rewards to be able to see both the raw data and the graph
    xs = []
    for i in sorted(data.keys()):
        mv = mean(data[i])
        if mv >= 8.0:
            xs.append(i)
    print("The min acceptable velocity is:", min(xs))
    print("The max acceptable velocity is:", max(xs))

    sorted_keys = sorted(data.keys())
    x = [i for i in sorted_keys]
    y = [mean(data[i]) for i in sorted_keys]
    error = [stdev(data[i]) for i in sorted_keys]
    lower_bound = [y[i] - error[i] for i in range(len(y))]
    upper_bound = [y[i] + error[i] for i in range(len(y))]

    plt.plot(x, y, '-k')
    plt.fill_between(x, lower_bound, upper_bound)
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)

    args = parser.parse_args()
    graph_dynamics(args.fpath)
