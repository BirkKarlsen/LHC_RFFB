'''
File to scan and plot different values of the RF power for the simulation of RFFB with
12 + 36 + 72 bunches.

Author: Birk Emil Karlsen-Bæck
'''

# Import
import numpy as np
import matplotlib.pyplot as plt




df_half_detuning = -6675.504384129072
drs = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
dfs = df_half_detuning * drs
colors = ['black', 'r', 'b', 'g', 'y']


PLT_1 = False
PLT_2 = True
if PLT_1:
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 9
    })
    lw = 0.7
    fig, ax = plt.subplots(3, 1, figsize=(5, 7.5))
    t_s = 1e6
    V_s = 1e-3
    P_s = 1e-3
    f_s = 1e-3

    ax[0].set_title(f'RF Voltage - Amplitude')
    ax[0].set_ylabel(f'RF Voltage [kV]')
    ax[0].set_xlabel(f'$\Delta t$ [$\mu$s]')

    ax[1].set_title(f'RF Voltage - Phase')
    ax[1].set_ylabel(f'Phase [degrees]')
    ax[1].set_xlabel(f'$\Delta t$ [$\mu$s]')

    ax[2].set_title(f'Generator Power')
    ax[2].set_ylabel(f'Power [kW]')
    ax[2].set_xlabel(f'$\Delta t$ [$\mu$s]')

    for i in range(len(drs)):
        file_i = f'signals_from_{100 * drs[i]:.0f}.npy'

        signals = np.load(file_i)
        t_coarse = signals[0, :]
        voltage_amp = signals[1, :]
        voltage_deg = signals[2, :]
        power = signals[3, :]

        ax[0].plot(t_coarse * t_s, voltage_amp * V_s, color=colors[i], linewidth=lw,
                   label=f'$\Delta f$ = {dfs[i] * f_s:.2f} kHz')
        ax[0].set_xlim((t_coarse[0] * t_s, t_coarse[-1] * t_s))

        ax[1].plot(t_coarse * t_s, voltage_deg, color=colors[i], linewidth=lw,
                   label=f'$\Delta f$ = {dfs[i] * f_s:.2f} kHz')
        ax[1].set_xlim((t_coarse[0] * t_s, t_coarse[-1] * t_s))

        ax[2].plot(t_coarse * t_s, power * P_s, color=colors[i], linewidth=lw,
                   label=f'$\Delta f$ = {dfs[i] * f_s:.2f} kHz')
        ax[2].set_xlim((t_coarse[0] * t_s, t_coarse[-1] * t_s))

    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.03), ncol=3)


if PLT_2:
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })
    plt.figure()
    plt.title(f'Generator Power with Beam')
    plt.ylabel(f'Power [kW]')
    plt.xlabel(f'$\Delta t$ [$\mu$s]')

    t_s = 1e6
    V_s = 1e-3
    P_s = 1e-3
    f_s = 1e-3

    for i in range(len(drs)):
        file_i = f'signals_from_{100 * drs[i]:.0f}.npy'

        signals = np.load(file_i)
        t_coarse = signals[0, :]
        voltage_amp = signals[1, :]
        voltage_deg = signals[2, :]
        power = signals[3, :]

        plt.plot(t_coarse * t_s, power * P_s, color=colors[i],
                   label=f'$\Delta f$ = {dfs[i] * f_s:.2f} kHz')
        plt.xlim((t_coarse[0] * t_s, t_coarse[-1] * t_s))

    #plt.legend(loc='center', bbox_to_anchor=(0.5, 0.03), ncol=3)
    plt.legend(loc='right')

plt.show()













