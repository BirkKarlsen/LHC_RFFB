import numpy as np


def convert_to_waveform(signal, omega, t):
    r'''Coverts IQ signal into its waveform equivalent.'''
    amp = np.abs(signal)
    phase = np.angle(signal)

    return amp * np.sin(omega * t + phase + np.pi / 2)

def LHC_analytic_generator_current(V_a, Q_L=20000, R_over_Q=45, df=0, f_c=400.789e6, I_b=0.6, return_power=False):
    r'''Calculates the generator current assuming constant antenna
    voltage and beam current.'''

    I_g = V_a / (2 * R_over_Q) * (1/Q_L - 2 * 1j * df/f_c) + 1/2 * I_b

    if return_power:
        power = 1/2 * R_over_Q * Q_L * np.abs(I_g)**2
        return I_g, power
    else:
        return I_g


def linear_to_dB(value):
    return 20 * np.log10(value)

def dB_to_linear(value):
    return 10**(value/20)