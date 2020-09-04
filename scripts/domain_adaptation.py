#!/usr/bin/env python

from test_policy import run_policy
from safe_rl.utils.load_utils import load_policy
import xml.etree.ElementTree as ET
import joblib
import time


def shift_dynamics(dynamic, xml, new_val):
    '''
    Shifts some value in a specified mujoco xml file, and then it writes the
    new value to the xml fileself.

    Args:
        dynamic: A string representing a the certain field in the xml that
                 represents a dynamic that will be shifted
        xml: A string that is a file path to an XML that will be edited
        new_val: A float that will be the new value for that dynamic
    '''
    tree = ET.parse(xml)
    root = tree.getroot()
    if dynamic == 'motor':
        x = [i.attrib for i in root.iter('motor')]
        x[0]['forcerange'] = '-' + str(new_val) + ' ' + str(new_val)
        i = 0
        for motor in root.iter('motor'):
            motor.attrib = x[i]
            i += 1
        tree.write(xml)
    elif dynamic == 'velocity':
        x = [i.attrib for i in root.iter('velocity')]
        x[0]['forcerange'] = '-' + str(new_val) + ' ' + str(new_val)
        i = 0
        for veloc in root.iter('velocity'):
            veloc.attrib = x[i]
            i += 1
        tree.write(xml)
        print("Updated dynamics for", dynamic)
    elif dynamic == 'pointarrow':
        x = [i.attrib for i in root.iter('geom')]
        x[3]['size'] = str(new_val) + ' ' + str(new_val) + ' ' + str(new_val)
        i = 0
        for pa in root.iter('geom'):
            pa.attrib = x[i]
            i += 1
        tree.write(xml)
    elif dynamic == 'robot':
        x = [i.attrib for i in root.iter('geom')]
        x[2]['size'] = str(new_val)
        i = 0
        for pa in root.iter('geom'):
            pa.attrib = x[i]
            i += 1
        tree.write(xml)
    elif dynamic == 'motor_false':
        raise NotImplementedError('Have not implemented')
        # TODO: Need to implement
    elif dynamic == 'jointx':
        # TODO: implement joint
        raise NotImplementedError('Have not implemented')
    elif dynamic == 'jointy':
        # TODO: implement joint
        raise NotImplementedError('Have not implemented')
    elif dynamic == 'jointz':
        # TODO: implement joint
        raise NotImplementedError('Have not implemented')


def get_range_vals(dynamic):
    '''
    Returns the list of new values that will be written to an XML file

    Args:
        dynamics: A string representing the dynamic to shift

    Returns:
        The list of floats of the new values
    '''
    if dynamic == 'motor' or dynamic == 'velocity' or dynamic == 'pointarrow':
        return [0.0001, 0.0005, 0.0025, 0.002, 0.001, 0.005, 0.01, 0.02, 0.03,
                0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35,
                0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                0.95, 1.0, 0.035, 0.0325, 0.0375, 0.041, 0.042, 0.043, 0.044,
                0.045, 0.046, 0.047, 0.048, 0.049, 0.051, 0.052, 0.053, 0.054,
                0.055, 0.056, 0.057, 0.058, 0.059, 0.055, 0.0575, 0.0625, 0.065,
                0.0675]
        # return [0.005]
    elif dynamic == 'joint':
        return [0.001]
    elif dynamic == 'size':
        return [0.001, 0.0005, 0.0001, 0.0025, 0.005, 0.0075, 0.01, 0.0125,
                0.015, 0.0175, 0.02, 0.021, 0.022, 0.0225, 0.023, 0.024,
                0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.031, 0.033, 0.034,
                0.036, 0.037, 0.038, 0.039, 0.0325, 0.032, 0.035, 0.0375, 0.04,
                0.0425, 0.045, 0.0475, 0.041, 0.042, 0.043, 0.044, 0.046,
                0.047, 0.048, 0.049, 0.05, 0.0525, 0.055, 0.0575, 0.06,
                0.0625, 0.065, 0.0675, 0.07, 0.0725, 0.075, 0.0775, 0.08, 0.825,
                0.085, 0.0875, 0.09, 0.0925, 0.095, 0.0975, 1.0]
    elif dynamic == 'robot':
        return [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275, 0.03,
                0.0325, 0.035, 0.0375, 0.04, 0.0425, 0.0475, 0.05, 0.0525,
                0.055, 0.0575, 0.06, 0.0625, 0.065, 0.0675, 0.07, 0.0725, 0.075,
                0.0775, 0.08, 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087,
                0.088, 0.089, 0.09, 0.091, 0.092, 0.093, 0.094, 0.095, 0.096,
                0.097, 0.098, 0.099, 0.0999, 0.1, 0.15, 0.2]
    raise NotImplementedError('Have not implemented the dynamic:', dynamic)


def test_dynamics(env, get_action, dynamic, xml_file, episodes=20):
    '''
    It shifts a certain dynamic value to a new one, and it then tests a policy
    against these new values for a specified number of episodes. It then
    collects the average episodic returns of each episode for each policy, and
    then dumps the collected data to a pickle file that can be used later for
    visualizations or other analysis.

    Args:
        env: The openAI gym environment
        get_action: An openAI policy function that consumes obseravtions and returns actions
        dynamic: A string representing the dynamic to shift
        xml_file: A string of a path to a XML file to edit
        episodes: The number of episodes to run a policy for

    Returns:
        Nothing, but dumps a pickle file with the collected data
    '''
    assert dynamic != "", "You did not specify a dynamic"
    points = {}
    forcerange_vals = get_range_vals(dynamic)
    for x_val in forcerange_vals:
        shift_dynamics(dynamic, xml_file, x_val)
        y_vals = run_policy(env, get_action, 0, episodes, False)
        points[x_val] = y_vals
    pkl_name = str("shift_data_" + str(dynamic) + '_point_' + str(time.strftime("%Y-%m-%d")) + ".pkl")
    joblib.dump(points, pkl_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=20)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--dynamic', '-dn', type=str, default="")
    parser.add_argument('--xml_file', '-xf', type=str, default="")
    args = parser.parse_args()
    env, get_action, _ = load_policy(args.fpath,
                                     args.itr if args.itr >= 0 else 'last',
                                     args.deterministic)
    test_dynamics(env, get_action, args.dynamic, args.xml_file, args.episodes)
