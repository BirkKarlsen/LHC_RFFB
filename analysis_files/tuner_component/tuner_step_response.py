'''
Test of the step response for my tuner model

Author: Birk Emil Karlsen-Bæck
'''

# Imports
import numpy as np

def CIC_component(signal):
    output = np.zeros(signal.shape)
    for i in range(16, len(signal)):
        output[i] = (1 / 64) * (signal[i] - 2 * signal[i - 8] + signal[i - 16]) + \
                                      2 * output[i - 1] - output[i - 2]

def crossproduct_component(signal):
    output = np.zeros(signal.shape)

