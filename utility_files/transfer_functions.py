'''
File with analytic versions of the transfer functions in the LHC Cavity Loop.

Taken from LHC-RF - LHC-LLRF repository on gitlab.

Author: Birk Emil Karlsen-Bæck
'''

import numpy as np


def FeedbackTF(f, g_d, tau_d, g_a, tau_a, delta_phi):
    """Evaluate the feedback transfer function for the given array of frequencies (f).

    Parameters
    ----------
     f   :   numpy.ndarray or Number
        array of frequencies at which the transfer function should be evaluated
    """

    # NOTE: changed to positive Ga, but was:  -self.gain_analog

    w = 2 * np.pi * f

    # fraction 1: \\frac{G_{D} e^{\\Delta\\varphi \\frac{\\pi}{180} i}}{1+\\omega \\tau_{D} i}
    num1 = g_d * np.exp(np.radians(delta_phi) * 1j)
    den1 = 1 + w * tau_d * 1j

    # fraction 2: \\frac{\\tau_{A} \\omega}{\\tau_{A} \\omega-i}
    num2 = tau_a * w
    den2 = tau_a * w - 1j

    resp = -g_a * (num1 / den1 + num2 / den2)

    return resp

def CavityTF(f, f_rf, delta_f, g_oo, r_over_q, q):
    """Evaluate the cavity transfer function for the given array of frequencies (f).

    Parameters
    ----------
    f   :   numpy.ndarray or Number
        array of frequencies at which the transfer function should be evaluated
    """

    w = 2 * np.pi * f  # omega
    s = 1j * (w + 2 * np.pi * f_rf)

    w_r = 2 * np.pi * (f_rf + delta_f)  # resonant frequency

    resp = g_oo * r_over_q * w_r * s / np.sqrt(q) / (s ** 2 + (w_r / q) * s + w_r ** 2)

    return resp


def H_a(f, g_a, tau_a):

    s = 1j * 2 * np.pi * f
    return g_a * tau_a * s / (1 + tau_a * s)

def H_d(f, g_a, g_d, tau_d, dphi_ad):

    s = 1j * 2 * np.pi * f
    dphi_ad = dphi_ad * np.pi / 180
    return g_a * g_d * np.exp(1j * dphi_ad) / (1 + tau_d * s)

def Z_cav(f, df, f_rf, R_over_Q, Q_L):

    s = 1j * 2 * np.pi * f
    domega = 2 * np.pi * df
    omega_rf = 2 * np.pi * f_rf

    return R_over_Q * Q_L / (1 + 2 * Q_L * (s - 1j * domega)/omega_rf)

def Z_cav2(f, df, f_rf, R_over_Q, Q_L):
    s = 1j * 2 * np.pi * (f + f_rf)
    domega = 2 * np.pi * df
    omega_rf = 2 * np.pi * f_rf
    omega_r = omega_rf + domega

    return R_over_Q * omega_r * s / Q_L / (s**2 + (omega_r/Q_L) * s + omega_r)

def H_cl(f, tau_loop, H_a, H_d, Z_cav):
    H_ad = H_a + H_d
    s = 1j * 2 * np.pi * f
    return 2 * np.exp(-tau_loop * s) * H_ad * Z_cav / (1 + 2 * np.exp(-tau_loop * s) * H_ad * Z_cav)

def H_open(f, tau_loop, H_a, H_d, Z_cav):
    H_ad = H_a + H_d
    s = 1j * 2 * np.pi * f
    return 2 * np.exp(-tau_loop * s) * H_ad * Z_cav


