'''
File with functions to import and treat measurements related to the LHC Cavity Loop.

Author: Birk Emil Karlsen-Bæck
'''

# Imports -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

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

def get_tf_measurement_conditions(directory, file_name, open_loop=False):
    r'''
    Takes in the directory and the file name of a LHC TF measurement and prints the settings of the cavity.

    :param directory: string - directory of file
    :param file_name: string - name of file
    :return: Prints parameters
    '''

    with open(directory + file_name) as f:
        # Finds the data related to the transfer function (TF)
        data = json.load(f)['environment']['prm']['fit_prm']

    print('--- Analog Feedback ---')
    print(f"Gain: {data['analog']['gain']}")
    print(f"Tau: {data['analog']['tau']}")
    print('--- Digital Feedback ---')
    print(f"Gain: {data['digital']['gain']}")
    print(f"Tau: {data['digital']['tau']}")
    print(f"Phase: {data['digital']['phase']}")
    print('--- Cavity ---')
    print(f"Goo: {data['cavity']['Goo']}")
    print(f"Q: {data['cavity']['Q']}")
    if not open_loop:
        print(f"Qcl: {data['cavity']['Qcl']}")
    print(f"R: {data['cavity']['R']}")
    print(f"detune: {data['cavity']['detune']}")
    print(f"wr: {data['cavity']['wr']}")
    print('--- Delay ---')
    print(f"controller: {data['delay']['controller']}")
    print(f"loop: {data['delay']['loop']}")


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


def import_klystron_data(filename):
    #data = {}
    #with open(filename, 'r') as f:
    #    Lines = f.readlines()
    cavities = ['VARIABLE: ACSKL.UX45.L1B1:RF_KLYSTRON_FWD', 'VARIABLE: ACSKL.UX45.L1B2:RF_KLYSTRON_FWD',
                'VARIABLE: ACSKL.UX45.L2B1:RF_KLYSTRON_FWD', 'VARIABLE: ACSKL.UX45.L2B2:RF_KLYSTRON_FWD',
                'VARIABLE: ACSKL.UX45.L3B1:RF_KLYSTRON_FWD', 'VARIABLE: ACSKL.UX45.L3B2:RF_KLYSTRON_FWD'
                'VARIABLE: ACSKL.UX45.L4B1:RF_KLYSTRON_FWD', 'VARIABLE: ACSKL.UX45.L4B2:RF_KLYSTRON_FWD',
                'VARIABLE: ACSKL.UX45.L5B1:RF_KLYSTRON_FWD', 'VARIABLE: ACSKL.UX45.L5B2:RF_KLYSTRON_FWD',
                'VARIABLE: ACSKL.UX45.L6B1:RF_KLYSTRON_FWD', 'VARIABLE: ACSKL.UX45.L6B2:RF_KLYSTRON_FWD',
                'VARIABLE: ACSKL.UX45.L7B1:RF_KLYSTRON_FWD', 'VARIABLE: ACSKL.UX45.L7B2:RF_KLYSTRON_FWD',
                'VARIABLE: ACSKL.UX45.L8B1:RF_KLYSTRON_FWD', 'VARIABLE: ACSKL.UX45.L8B2:RF_KLYSTRON_FWD']
    data = pd.read_csv(filename, names=cavities)


    return data

def import_klystron_data2(filename, time_interval, sampling_rate):
    n_points = time_interval // sampling_rate
    data = {}
    with open(filename, 'r') as f:
        Lines = f.readlines()
        i = 0
        name_i = ''
        timestamps_i = []
        values_i = []
        for line in Lines:
            if line.startswith('VARIABLE: '):
                data[name_i] = {'Timestamps': np.array(timestamps_i),
                                'Power': np.array(values_i, dtype=float)}
                name_i = line[len('VARIABLE: '):-1]
                timestamps_i = []
                values_i = []
            elif line != '\n' and not line.startswith('Timestamp'):
                line_parts = line.split(',')
                timestamps_i.append(line_parts[0])
                values_i.append(line_parts[1])


    del data['']

    return data


def into_second(time):
    return sum(x * float(t) for x, t in zip([3600, 60, 1], time.split(":")))