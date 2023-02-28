'''
File to import an analyze the data that was taken of the Klystrons
during the LHC MD on Saturday 25/06/2022.

Author: Birk Emil Karlsen-Bæck
'''

# Import
import numpy as np
import matplotlib.pyplot as plt

import utility_files.measurement_utilites as mut
import utility_files.signal_utilities as sut


plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })


# Options
PLT_MEAS = False

# Importing the data
file_dir = f'../../data_files/TIMBER_data.csv'
data = mut.import_klystron_data2(file_dir, 5100, 1)

klystrons = np.array(list(data.keys()))
times = {}
for i in range(len(klystrons)):
    time_i = np.zeros(len(data[klystrons[i]]['Timestamps']))
    for j in range(len(data[klystrons[i]]['Timestamps'])):
        stamp_str = data[klystrons[i]]['Timestamps'][j]

        time_i[j] = mut.into_second(stamp_str[len('2022-06-25 '):])
    times[klystrons[i]] = time_i

if PLT_MEAS:
    plt.figure()
    plt.title('Klystron Power')
    start_plot = 2000
    for i in range(len(klystrons)):
        plt.plot(times[klystrons[i]] - times[klystrons[0]][0] - start_plot, data[klystrons[i]]['Power'])

    plt.xlabel(f'Time [s]')
    plt.ylabel(f'Power [kW]')
    plt.xlim(2000 - start_plot, 3500 - start_plot)
    plt.ylim(0, 140)
    plt.show()


# Measurement of each Klystron from MD
B1 = np.array([[15, 12, 10, 10, 10, 12, 9, 11],
               [21, 18, 16, 16, 17, 18, 15, 17],
               [30, 26, 24, 23, 24, 26, 22, 25],
               [40, 35, 32, 32, 34, 34, 29, 35],
               [52, 45, 42, 42, 45, 44, 38, 58],
               [66, 57, 53, 54, 59, 55, 48, 58],
               [82, 71, 65, 68, 75, 67, 60, 73],
               [101, 87, 78, 84, 94, 81, 73, 89],
               [121, 105, 92, 102, 115, 96, 86, 108]])

B2 = np.array([[12, 13, 11, 13, 10, 11, 11, 12],
               [18, 20, 18, 20, 16, 17, 17, 19],
               [26, 27, 26, 29, 22, 24, 24, 27],
               [35, 37, 34, 39, 30, 33, 32, 38],
               [46, 48, 44, 50, 39, 43, 42, 51],
               [58, 59, 57, 62, 48, 55, 54, 66],
               [71, 73, 70, 76, 59, 68, 66, 84],
               [86, 87, 84, 90, 70, 82, 80, 105],
               [101, 103, 100, 106, 81, 98, 85, 129]])

Voltages = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12])

print('MD Measurements')
avg_power = (np.mean(B1, axis=1) + np.mean(B2, axis=1))/2
for i in range(len(Voltages)):
    print(f'At {Voltages[i]} MV the power was {avg_power[i]:.2f} kW +- {avg_power[i] * 0.2:.2f}')

