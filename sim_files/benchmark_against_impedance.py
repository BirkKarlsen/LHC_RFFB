'''
File to simulate the full ring and compare with I. Karpov paper:
"Consequences of longitudinal coupled-bunch instability mitigation on power requirements during the HL-LHC filling"

Author: Birk Emil Karlsen-Bæck
'''

# Imports -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import matched_from_line_density
from blond.beam.profile import Profile, CutOptions
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.llrf.cavity_feedback import LHCRFFeedback, LHCCavityLoop
from blond.llrf.signal_processing import cartesian_to_polar
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF

from blond.impedances.impedance_sources import Resonators
from blond.impedances.impedance import InducedVoltageTime, TotalInducedVoltage

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
gamma_t = 53.606713                 # Transition gamma [-]
alpha = 1./gamma_t/gamma_t          # First order mom. comp. factor [-]
V = 10e6                            # RF voltage [V]
dphi = 0                            # Phase modulation/offset [rad]

# RFFB parameters
G_a = 6.79e-6                       # Analog FB gain [A/V]
G_d = 10                            # Digital FB gain [-]
tau_loop = 650e-9                   # Overall loop delay [s]
tau_a = 170e-6                      # Analog FB delay [s]
tau_d = 400e-6                      # Digital FB delay [s]
a_comb = 15/16                      # Comb filter alpha [-]
Q_L = 60000                         # Loaded Quality factor [-]
G_otfb = 10
tau_comp = 1175e-9                  # Complimentary delay in OTFB [s]
G_gen = 1
tau_o = 110e-6

df = -3e5

# Beam parameters
N_p = 1.2e11                        # Bunch intensity [p/b]
N_p_tmp = 1.2e11                    # Bunch intensity [p/b]
mu = 1.5                            # Binomial exponent
N_buckets = 30
bl = 1e-9
ddt = 0

# Simulation parameters
N_t = 100                           # Number of turns
N_m = int(5e5)                      # Number of macroparticles


# Objects for simulation ----------------------------------------------------------------------------------------------
print('Initializing Objects...\n')
# LHC ring
ring = Ring(C, alpha, p_s, Proton(), n_turns=N_t)

# 400MHz RF station
rfstation = RFStation(ring, [h], [V], [dphi])


# Beam single bunch
beam = Beam(ring, N_m, N_p)
tracker = RingAndRFTracker(rfstation, beam)
full_tracker = FullRingAndRF([tracker])
matched_from_line_density(beam, full_ring_and_RF=full_tracker, bunch_length=bl,
                          line_density_exponent=mu, line_density_type='binomial')

beam.dt += rfstation.t_rf[0, 0] * ddt

# Beam Profile
profile = Profile(beam, CutOptions(cut_left=(-0.5 + ddt) * rfstation.t_rf[0, 0],
                                   cut_right=(N_buckets + 0.5 + ddt) * rfstation.t_rf[0, 0],
                                   n_slices=(N_buckets + 1) * 2**7))
profile.track()
plt.figure()
plt.plot(profile.bin_centers, profile.n_macroparticles * beam.ratio)

# Impedance Model
R_S = Q_L * 45 * 8
resonator = Resonators(R_S=R_S, frequency_R=rfstation.omega_rf[0, 0]/(2 * np.pi) + df, Q=Q_L)
inducedvoltage = InducedVoltageTime(beam, profile, [resonator])
tot_imp = TotalInducedVoltage(beam, profile, [inducedvoltage])
tot_imp.induced_voltage_sum()

# LHC Cavity Controller
print(f'Pre-tracking Cavity Loop...\n')
RFFB = LHCRFFeedback(G_a=G_a, G_d=G_d, tau_d=tau_d, tau_a=tau_a, alpha=a_comb, tau_o=tau_o,
                     open_otfb=True, G_o=G_otfb, mu=-10, open_tuner=True, open_loop=False,
                     open_rffb=True, open_drive=True)

CL = LHCCavityLoop(rfstation, profile, RFFB=RFFB,
                   f_c=rfstation.omega_rf[0, 0]/(2 * np.pi) + df,
                   Q_L=Q_L, tau_loop=tau_loop, n_pretrack=1, tau_otfb=tau_comp,
                   G_gen=G_gen)

OLD = False
PLT_BEFORE_TRACKING = False

if PLT_BEFORE_TRACKING:
    plt.figure()
    plt.plot(CL.v_ant_trans)

    plt.figure()
    plt.plot(np.abs(CL.V_ANT))
    plt.plot(np.abs(CL.V_SET))

    plt.figure()
    plt.title('V FB IN')
    plt.plot(np.abs(CL.V_FB_IN))

    plt.figure()
    plt.title('I TEST')
    plt.plot(np.abs(CL.I_TEST))

    plt.figure()
    plt.title('IGEN')
    plt.plot(np.abs(CL.I_GEN))


    plt.figure()
    plt.title('IBEAM')
    plt.plot(np.abs(CL.I_BEAM))

    plt.show()

# Trackers
rf_tracker = RingAndRFTracker(rfstation, beam, TotalInducedVoltage=None,
                              CavityFeedback=None, Profile=profile, interpolation=True)

rf_tracker_with_imp = RingAndRFTracker(rfstation, beam, TotalInducedVoltage=tot_imp,
                                       CavityFeedback=None, Profile=profile, interpolation=True)

rf_tracker_with_cl = RingAndRFTracker(rfstation, beam, TotalInducedVoltage=None,
                                      CavityFeedback=CL, Profile=profile, interpolation=True)

# Track one turn
import time

start_time = time.time()
CL.track()
end_time = time.time()
print(f'Time of execusion was {end_time - start_time} s')
rf_tracker_with_cl.track()
profile.track()
tot_imp.induced_voltage_sum()

rf_tracker.rf_voltage_calculation()
rf_tracker_with_imp.rf_voltage_calculation()
rf_tracker_with_cl.rf_voltage_calculation()

# Compare generator induced and beam induced contributions
if not OLD:
    beam_ind = CL.V_ANT_FINE[-profile.n_slices:]
else:
    beam_ind = CL.V_sum
bV, bp = cartesian_to_polar(beam_ind)
bE = bV * np.sin(rfstation.omega_rf[0, 0] * profile.bin_centers + bp - np.mean(np.angle(CL.V_SET)))
#bE = bE - V * np.sin(rfstation.omega_rf[0, 0] * profile.bin_centers)
print('Calc')

IMP_tot = rf_tracker_with_imp.totalInducedVoltage.induced_voltage

# CL Signals
plt.figure()
plt.plot(CL.rf_centers, np.abs(CL.V_ANT[-CL.n_coarse:]))
plt.plot(CL.rf_centers, np.abs(CL.V_SET[-CL.n_coarse:]))
plt.xlim((profile.bin_centers[0], profile.bin_centers[-1]))

plt.figure()
plt.title('V FB IN')
plt.plot(CL.rf_centers, np.abs(CL.V_FB_IN[-CL.n_coarse:]))
plt.xlim((profile.bin_centers[0], profile.bin_centers[-1]))

plt.figure()
plt.title('I TEST')
plt.plot(CL.rf_centers, np.abs(CL.I_TEST[-CL.n_coarse:]))
plt.xlim((profile.bin_centers[0], profile.bin_centers[-1]))

plt.figure()
plt.title('IGEN')
plt.plot(CL.rf_centers, np.abs(CL.I_GEN[-CL.n_coarse:]))
plt.xlim((profile.bin_centers[0], profile.bin_centers[-1]))

plt.figure()
plt.title('IBEAM')
plt.plot(CL.rf_centers, np.abs(CL.I_BEAM[-CL.n_coarse:]))
plt.xlim((profile.bin_centers[0], profile.bin_centers[-1]))

plt.figure()
plt.title('V ANT')
plt.plot(CL.rf_centers, np.abs(CL.V_ANT[-CL.n_coarse:]))
plt.xlim((profile.bin_centers[0], profile.bin_centers[-1]))


if not OLD:
    plt.figure()
    plt.title('I GEN FINE')
    plt.plot(profile.bin_centers, np.abs(CL.I_GEN_FINE[-profile.n_slices:]))
    plt.xlim((profile.bin_centers[0], profile.bin_centers[-1]))

    plt.figure()
    plt.title('I BEAM FINE')
    plt.plot(profile.bin_centers, np.abs(CL.I_BEAM_FINE[-profile.n_slices:]))
    plt.xlim((profile.bin_centers[0], profile.bin_centers[-1]))

    plt.figure()
    plt.title('V ANT FINE')
    plt.plot(profile.bin_centers, np.abs(CL.V_ANT_FINE[-profile.n_slices:]))
    plt.xlim((profile.bin_centers[0], profile.bin_centers[-1]))


# Physical signals
plt.figure()
plt.title('amplitude')
plt.plot(profile.bin_centers, bV)

plt.figure()
plt.title('phase')
plt.plot(profile.bin_centers, bp)

plt.figure()
plt.title('Impedance Model')
plt.plot(profile.bin_centers, IMP_tot)

plt.figure()
plt.title('CL Model')
plt.plot(profile.bin_centers, bE)
#plt.plot(profile.bin_centers, V * np.sin(rfstation.omega_rf[0, 0] * profile.bin_centers))
#plt.plot(profile.bin_centers, rf_tracker.rf_voltage)

plt.figure()
plt.title('Beam-Induced Voltage')
plt.plot(profile.bin_centers*1e9, bE/1e3, label='CL', marker='x', color='r')
plt.plot(profile.bin_centers*1e9, IMP_tot/1e3, label='IMP', color='b')
plt.plot(profile.bin_centers*1e9, 288 * 1e3 * profile.n_macroparticles / np.sum(profile.n_macroparticles)/1e3,
         label='profile', color='black')
plt.xlim((profile.bin_centers[0]*1e9, profile.bin_centers[-1]*1e9))
#plt.xlim((25, 40))
plt.xlabel(r'$\Delta t$ [ns]')
plt.ylabel(r'Induced Voltage [kV]')
plt.grid()
plt.legend()

plt.show()
