'''
Simulation to compare power and voltage with the "The LHC One-Turn Feedback" paper.

Author: Birk Emil Karlsen-Bæck
'''

# Parser
import argparse

parser = argparse.ArgumentParser(description="Simulating LHC RFFB with 12+36+72 bunches.")

# Arguments related to retrieving and saving
parser.add_argument("--detuning_ratio", "-dr", type=float, default=1.0,
                    help="Option to change detuning.")

args = parser.parse_args()



# Imports -------------------------------------------------------------------------------------------------------------
print('Importing...\n')
# General libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tqdm

# Helpful files
import utility_files.signal_utilities as sut

# BLonD Library
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import Profile, CutOptions
from blond.trackers.tracker import RingAndRFTracker
from blond.llrf.cavity_feedback import LHCRFFeedback, LHCCavityLoop

plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })

# Parameters ----------------------------------------------------------------------------------------------------------

# Accelerator parameters
C = 26658.883                       # Machine circumference [m]
p_s = 450e9                         # Synchronous momentum [eV/c]
h = 35640                           # Harmonic number [-]
gamma_t = 53.8                      # Transition gamma [-]
alpha = 1./gamma_t/gamma_t          # First order mom. comp. factor [-]
V = 6e6                             # RF voltage [V]
dphi = 0                            # Phase modulation/offset [rad]

# RFFB parameters
G_a = 6.79e-6                       # Analog FB gain [A/V]
G_d = 10                            # Digital FB gain [-]
tau_loop = 650e-9                   # Overall loop delay [s]
tau_a = 170e-6                      # Analog FB delay [s]
tau_d = 400e-6                      # Digital FB delay [s]
a_comb = 15/16                      # Comb filter alpha [-]
Q_L = 20000                         # Loaded Quality factor [-]
dr = args.detuning_ratio
dr = 1.0

# Beam parameters
bl = 1.2e-9                         # Bunch length [s]
N_p = 1.15e11                       # Bunch intensity [p/b]
N_bunches = 120                     # Total number of bunches [-]
bunch_per_batch = [12, 36, 72]      # Number of bunches per batch [-]
bunch_spacing = 10                  # Spacing between bunches [RF buckets]
batch_spacing = 3000                # Spacing between batches [RF buckets]
first_bunch = 8000                  # Position of first bunch [RF buckets]

# Simulation parameters
N_t = 1000                          # Number of turns
N_m = int(5e5)                      # Number of macroparticles

# Objects for simulation ----------------------------------------------------------------------------------------------
print('Initializing Objects...\n')
# LHC ring
ring = Ring(C, alpha, p_s, Proton(), n_turns=N_t)

# 400MHz RF station
rfstation = RFStation(ring, [h], [V], [dphi])

# Beam
beam_single_bunch = Beam(ring, N_m, N_p)
bigaussian(ring, rfstation, beam_single_bunch, sigma_dt=bl/4, seed=1234)

beam = Beam(ring, N_bunches * N_m, N_bunches * N_p)

j = 0
bunch_pos_j = first_bunch
for batch in bunch_per_batch:
    for i in range(batch):
        beam.dE[j * N_m: (j + 1) * N_m] = beam_single_bunch.dE
        beam.dt[j * N_m: (j + 1) * N_m] = beam_single_bunch.dt + bunch_pos_j * rfstation.t_rf[0, 0]

        if i + 1 < batch:
            bunch_pos_j += bunch_spacing
        else:
            bunch_pos_j += batch_spacing

        j += 1

# Beam Profile
profile = Profile(beam, CutOptions(cut_left=(first_bunch - 3) * rfstation.t_rf[0, 0],
                                   cut_right=bunch_pos_j * rfstation.t_rf[0, 0],
                                   n_slices=(bunch_pos_j - first_bunch) * 2**7))

profile.track()

# LHC Cavity Controller
RFFB = LHCRFFeedback(alpha=a_comb, G_a=G_a, G_d=G_d, tau_a=tau_a, tau_d=tau_d,
                     open_otfb=False, open_loop=False, excitation=False)

CavFB = LHCCavityLoop(rfstation, profile, f_c=rfstation.omega_rf[0,0]/(2*np.pi),
                      RFFB=RFFB, Q_L=Q_L, tau_loop=tau_loop, n_pretrack=5, tau_otfb=1.2e-6)

CavFB.rf_beam_current()
I_rf_pk = np.max(np.absolute(CavFB.I_BEAM))
d_f = LHCCavityLoop.half_detuning(rfstation.omega_rf[0,0]/(2*np.pi), I_rf_pk, 45, V/8) * dr
print(d_f)

CavFB = LHCCavityLoop(rfstation, profile, f_c=rfstation.omega_rf[0,0]/(2*np.pi) - d_f,
                      RFFB=RFFB, Q_L=Q_L, tau_loop=tau_loop, n_pretrack=100, tau_otfb=1.2e-6)


PLOT_NO_BEAM = False
if PLOT_NO_BEAM:
    plt.figure('Generator current')
    plt.plot(np.real(CavFB.I_GEN), label='real')
    plt.plot(np.imag(CavFB.I_GEN), label='imag')
    plt.xlabel('Samples [at 40 MS/s]')
    plt.ylabel('Generator current [A]')

    plt.figure('Antenna voltage')
    plt.plot(np.real(CavFB.V_ANT)*1e-6, label='real')
    plt.plot(np.imag(CavFB.V_ANT)*1e-6, label='imag')
    plt.xlabel('Samples [at 40 MS/s]')
    plt.ylabel('Antenna voltage [MV]')
    plt.legend()
    plt.show()

# Tracker object
rftracker = RingAndRFTracker(rfstation, beam, CavityFeedback=CavFB,
                             Profile=profile, interpolation=True)

# Simulate ------------------------------------------------------------------------------------------------------------
print('Simulation...\n')

t_coarse = np.linspace(0, CavFB.n_coarse * 10, CavFB.n_coarse) * rfstation.t_rf[0, 0]

dt_print = 10

dt_plot = 10
power_evolution = np.zeros((N_t//dt_plot, CavFB.n_coarse))
vant_evolution = np.zeros((N_t//dt_plot, CavFB.n_coarse), dtype=complex)
cmap = mpl.cm.get_cmap('jet', N_t//dt_plot)
j = 0
PLT_DURING_TRACKING = False
SHOW_PLT = True
PLT_ALL = False
PLT_TBT_SIG = True
PLT_FINAL_TURN = True

dt_check_phase = 100
PLT_CHECK_PHASE = False

dt_check_beam = 10
PLT_CHECK_BEAM = False
SAVE_SIGS = True
# Main for loop
for i in tqdm.tqdm(range(N_t)):
    rftracker.track()
    profile.track()
    CavFB.track()

    if i == 0 and False:
        plt.figure()
        plt.plot(np.angle(CavFB.I_BEAM, deg=True), color='r')
        #plt.plot(CavFB.I_BEAM.imag, color='r', linestyle='--')

        plt.plot(np.angle(CavFB.V_SET, deg=True), color='b')
        #plt.plot(CavFB.V_SET.imag, color='b', linestyle='--')
        plt.show()

    if i == 1 and False:
        fig = plt.figure('Antenna voltage, first turns with beam', figsize=(10, 5))
        gs = plt.GridSpec(2, 4)
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.plot(np.absolute(CavFB.V_ANT) * 1e-6, 'b', linewidth=0.3)
        ax1.set_xlabel('Samples [at 40 MS/s]')
        ax1.set_ylabel('Antenna voltage [MV]')
        ax2 = fig.add_subplot(gs[1, 0:2], sharex=ax1)
        ax2.plot(np.angle(CavFB.V_ANT, deg=True), 'b', linewidth=0.3)
        ax2.set_xlabel('Samples [at 40 MS/s]')
        ax2.set_ylabel('Phase [degrees]')
        ax3 = fig.add_subplot(gs[:, 2:4])
        ax3.scatter(CavFB.V_ANT.real * 1e-6, CavFB.V_ANT.imag * 1e-6)
        ax3.set_xlabel('Voltage, I [MV]')
        ax3.set_ylabel('Voltage, Q [MV]')
        plt.tight_layout()

        fig = plt.figure('Generator current, first turns with beam', figsize=(10, 5))
        gs = plt.GridSpec(2, 4)
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.plot(np.absolute(CavFB.I_GEN), 'b', linewidth=0.3)
        ax1.set_xlabel('Samples [at 40 MS/s]')
        ax1.set_ylabel('Generator current [A]')
        ax2 = fig.add_subplot(gs[1, 0:2], sharex=ax1)
        ax2.plot(np.angle(CavFB.I_GEN, deg=True), 'b', linewidth=0.3)
        ax2.set_xlabel('Samples [at 40 MS/s]')
        ax2.set_ylabel('Phase [degrees]')
        ax3 = fig.add_subplot(gs[:, 2:4])
        ax3.scatter(CavFB.I_GEN.real, CavFB.I_GEN.imag)
        ax3.set_xlabel('Current, I [A]')
        ax3.set_ylabel('Current, Q [A]')
        plt.tight_layout()
        plt.show()


#    if i % dt_print == 0:
#        print(f'Turn {i}')

    if i % dt_plot == 0:

        power_evolution[j, :] = CavFB.generator_power()[-CavFB.n_coarse:]
        vant_evolution[j, :] = CavFB.V_ANT[-CavFB.n_coarse:]

        if PLT_DURING_TRACKING:
            plt.figure()
            plt.title('Power')
            for k in range(j + 1):
                plt.plot(t_coarse, power_evolution[k, :], color=cmap(k))

            plt.figure()
            plt.title('Voltage')
            for k in range(j + 1):
                plt.plot(t_coarse, vant_evolution[k, :].real, color=cmap(k))
                plt.plot(t_coarse, vant_evolution[k, :].imag, color=cmap(k), linestyle='--')

        j += 1
        if PLT_ALL:
            plt.figure()
            plt.title('I test')
            plt.plot(t_coarse, CavFB.I_TEST[-CavFB.n_coarse:].real)
            plt.plot(t_coarse, CavFB.I_TEST[-CavFB.n_coarse:].imag)

            plt.figure()
            plt.title('I gen')
            plt.plot(t_coarse, CavFB.I_GEN[-CavFB.n_coarse:].real)
            plt.plot(t_coarse, CavFB.I_GEN[-CavFB.n_coarse:].imag)

        if PLT_ALL or PLT_DURING_TRACKING:
            plt.show()


    if i % dt_check_beam == 0 and PLT_CHECK_BEAM:
        I_beam = sut.convert_to_waveform(CavFB.I_BEAM_FINE, rfstation.omega_rf[0,0], profile.bin_centers)

        plt.figure()
        plt.title('Beam profile')
        plt.plot(profile.bin_centers, profile.n_macroparticles * beam.ratio)
        plt.plot(profile.bin_centers, I_beam * beam.ratio * np.sum(profile.n_macroparticles) / np.sum(I_beam))

        plt.figure()
        plt.plot(profile.bin_centers, np.abs(CavFB.I_BEAM_FINE))
        plt.plot(t_coarse, np.abs(CavFB.I_BEAM[-CavFB.n_coarse:]))

        plt.figure()
        plt.plot(t_coarse, np.abs(CavFB.I_BEAM[-CavFB.n_coarse:]))

        plt.show()



if PLT_TBT_SIG:
    plt.figure()
    plt.title('Power')
    for k in range(N_t//dt_plot):
        plt.plot(t_coarse, power_evolution[k, :], color=cmap(k))

    plt.figure()
    plt.title('Voltage Amp')
    for k in range(N_t//dt_plot):
        plt.plot(t_coarse, np.abs(vant_evolution[k, :]), color=cmap(k))

    plt.figure()
    plt.title('Voltage Degrees')
    for k in range(N_t//dt_plot):
        plt.plot(t_coarse, np.angle(vant_evolution[k, :], deg=True), color=cmap(k))

    plt.figure()
    plt.title('Beam Profile')
    plt.plot(profile.bin_centers, profile.n_macroparticles * beam.ratio)

if PLT_FINAL_TURN:
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
    power = CavFB.generator_power()

    ax[0].set_title(f'RF Voltage - Amplitude')
    ax[0].plot(t_coarse * t_s, np.abs(CavFB.V_ANT[-CavFB.n_coarse:]) * V_s, color='r', linewidth=lw)
    ax[0].set_ylabel(f'RF Voltage [kV]')
    ax[0].set_xlabel(f'$\Delta t$ [$\mu$s]')
    ax[0].set_xlim((t_coarse[0] * t_s, t_coarse[-1] * t_s))

    ax[1].set_title(f'RF Voltage - Phase')
    ax[1].plot(t_coarse * t_s, np.angle(CavFB.V_ANT[-CavFB.n_coarse:], deg=True), color='r', linewidth=lw)
    ax[1].set_ylabel(f'Phase [degrees]')
    ax[1].set_xlabel(f'$\Delta t$ [$\mu$s]')
    ax[1].set_xlim((t_coarse[0] * t_s, t_coarse[-1] * t_s))

    ax[2].set_title(f'Generator Power')
    ax[2].plot(t_coarse * t_s, power[-CavFB.n_coarse:] * 1e-3, color='r', linewidth=lw)
    ax[2].set_ylabel(f'Power [kW]')
    ax[2].set_xlabel(f'$\Delta t$ [$\mu$s]')
    ax[2].set_xlim((t_coarse[0] * t_s, t_coarse[-1] * t_s))


if SHOW_PLT:
    plt.show()

if SAVE_SIGS:
    sigs = np.zeros((4, len(t_coarse)))
    sigs[0, :] = t_coarse
    sigs[1, :] = np.abs(CavFB.V_ANT[-CavFB.n_coarse:])
    sigs[2, :] = np.angle(CavFB.V_ANT[-CavFB.n_coarse:], deg=True)
    sigs[3, :] = power[-CavFB.n_coarse:]

    np.save(f'signals_from_{dr * 100:.0f}.npy', sigs)