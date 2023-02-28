'''
Simulation to compare power and voltage with the "The LHC One-Turn Feedback" paper.

Author: Birk Emil Karlsen-Bæck
'''


# Imports -------------------------------------------------------------------------------------------------------------
print('Importing...\n')
# General libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

'''
0.5432
0.5432
0.5432
0.5432
0.5432
0.5432
0.5432
0.5432
0.5432
'''

# Parameters ----------------------------------------------------------------------------------------------------------

# Accelerator parameters
C = 26658.883                       # Machine circumference [m]
p_s = 450e9                         # Synchronous momentum [eV/c]
h = 35640                           # Harmonic number [-]
gamma_t = 53.8                      # Transition gamma [-]
alpha = 1./gamma_t/gamma_t          # First order mom. comp. factor [-]
V = 12e6                             # RF voltage [V]
dphi = 0                            # Phase modulation/offset [rad]

# RFFB parameters
G_a = 6.79e-6                       # Analog FB gain [A/V]
G_d = 10                            # Digital FB gain [-]
tau_loop = 650e-9                   # Overall loop delay [s]
tau_a = 170e-6                      # Analog FB delay [s]
tau_d = 400e-6                      # Digital FB delay [s]
a_comb = 15/16                      # Comb filter alpha [-]
Q_L = 60000                         # Loaded Quality factor [-]

# Beam parameters
bl = 1.2e-9                         # Bunch length [s]
N_p = 1.15e11                       # Bunch intensity [p/b]
N_bunches = 120                     # Total number of bunches [-]
bunch_per_batch = [12, 36, 72]      # Number of bunches per batch [-]
bunch_spacing = 10                  # Spacing between bunches [RF buckets]
batch_spacing = 3000                # Spacing between batches [RF buckets]
first_bunch = 8000                  # Position of first bunch [RF buckets]

# Simulation parameters
N_t = 200                          # Number of turns
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
                     open_otfb=True, open_loop=False, excitation=False)

CavFB = LHCCavityLoop(rfstation, profile, f_c=rfstation.omega_rf[0,0]/(2*np.pi),
                      RFFB=RFFB, Q_L=Q_L, tau_loop=tau_loop, n_pretrack=5)

CavFB.rf_beam_current()
I_rf_pk = np.max(np.absolute(CavFB.I_BEAM))
d_f = LHCCavityLoop.half_detuning(rfstation.omega_rf[0,0]/(2*np.pi), I_rf_pk, 45, V/8) * 0.0

CavFB = LHCCavityLoop(rfstation, profile, f_c=rfstation.omega_rf[0,0]/(2*np.pi) - d_f,
                      RFFB=RFFB, Q_L=Q_L, tau_loop=tau_loop, n_pretrack=100)

P_g_sim = CavFB.generator_power()
P_g_s = np.mean(P_g_sim)
V_a_s = np.mean(CavFB.V_ANT)
print(V_a_s)
print((V/8 - np.abs(V_a_s))/V/8 * 100)
print((V/8 - np.abs(V_a_s))/V/8 * 100 * 2)

I_g, P_g = sut.LHC_analytic_generator_current(1j * V/8, Q_L=Q_L, R_over_Q=45, df=d_f,
                                              f_c=rfstation.omega_rf[0,0]/(2*np.pi), I_b=0, return_power=True)

PLT_POWER = False

print('Power Scaling Estimates')
print(f'Total RF Voltage {V * 1e-6:.4f} MV')
print(f'RF Voltage {V/8 * 1e-6:.4f} MV')
print(f'Analytic Power {P_g * 1e-3:.4f} kW')
print(f'Simulated Power {P_g_s * 1e-3:.4f} kW')
print(f'Error {(P_g - P_g_s)/P_g * 100:.4f} %')

if PLT_POWER:
    plt.figure()
    plt.title(f'Power from BLonD')
    plt.plot(P_g_sim)


plt.show()
