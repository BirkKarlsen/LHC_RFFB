'''
File with functions to import and treat measurements related to the LHC Cavity Loop.

Author: Birk Emil Karlsen-Bæck
'''

# Imports -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import json

# Functions -----------------------------------------------------------------------------------------------------------

def import_tf_measurement(directory, file_name):
    r'''
    Takes in the directory and the file name of a LHC TF measurement and returns the transfer function,
    the phase of the transfer function and the frequency array value of each measurement point.

    :param directory: string - directory of file
    :param file_name: string - name of file
    :return: H, H_phase and freq
    '''

    # Opens the file
    with open(directory + file_name) as f:
        # Finds the data related to the transfer function (TF)
        data = json.load(f)['environment']['prm']['fit_prm']['TF']

        # Extracts the TF value, H, the phase, H_phase, and the corresponding frequency values, freq.
        H = np.array(data['H'], dtype=complex)
        H_phase = np.array(data['H_phase'])
        freq = np.array(data['freq'])

    return H, H_phase, freq

def get_tf_measurement_conditions(directory, file_name):
    r'''
    Takes in the directory and the file name of a LHC TF measurement and prints the settings of the cavity.

    :param directory: string - directory of file
    :param file_name: string - name of file
    :return: Prints parameters
    '''

    with open(directory + file_name) as f:
        # Finds the data related to the transfer function (TF)
        data = json.load(f)['environment']['prm']['fit_prm']

        print(data.keys())
        print(data['analog'].keys())
        print(data['cavity'].keys())
        print(data['delay'].keys())
        print(data['digital'].keys())

    print('--- Analog Feedback ---')
    print(f"Gain: {data['analog']['gain']}")
    print(f"Tau: {data['analog']['tau']}")
    print('--- Digital Feedback ---')
    print(f"Gain: {data['digital']['gain']}")
    print(f"Tau: {data['digital']['tau']}")
    print(f"Phase: {data['digital']['phase']}")
    print('--- Cavity ---')
    


def find_closes_value(array, value):
    r'''
    Finding the index to the closes value in an array.

    :param array: array
    :param value: float
    :return: index of the closes value
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def smooth_out_central_peak(H, freq, points, n_avg=3):
    zero_ind = find_closes_value(freq, 0)
    points_to_average = np.linspace(zero_ind - points, zero_ind + points, 2 * points + 1, dtype=int)

    for i in range(len(points_to_average)):
        H[points_to_average[i]] = np.mean(H[points_to_average[i] - n_avg: points_to_average[i] + n_avg])

    return H



directory = '../transfer_function_measurements/closed_loop/'
file_name = '1B1.json'

with open(directory + file_name) as f:
    # Finds the data related to the transfer function (TF)
    data = json.load(f)['environment']['prm']['fit_prm']

    print(data.keys())
    print(data['analog'].keys())
    print(data['cavity'].keys())
    print(data['delay'].keys())
    print(data['digital'].keys())

