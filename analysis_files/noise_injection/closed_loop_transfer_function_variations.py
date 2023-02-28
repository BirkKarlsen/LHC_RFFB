'''
File to measure the closed loop response of the LHC RFFB.

Author: Birk Emil Karlsen-Bæck
'''

# Imports -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Homemade files
import utility_files.measurement_utilites as mut
import utility_files.signal_utilities as sut

from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, FitOptions, CutOptions
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation

# Cavity Controller
from blond.llrf.cavity_feedback import LHCRFFeedback, LHCCavityLoop
from blond.llrf.transfer_function import TransferFunction

plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })

# Options -------------------------------------------------------------------------------------------------------------
PLT_MODEL_TF = False
PLT_MEAS_TF = False
PLT_COMP = True

# Parameters ----------------------------------------------------------------------------------------------------------
# Machine and RF parameters
C = 26658.883                       # Machine circumference [m]
p_s = 450e9                         # Synchronous momentum [eV/c]
h = 35640                           # Harmonic number
V = 4e6                             # RF voltage [V]
dphi = 0                            # Phase modulation/offset
gamma_t = 53.8                      # Transition gamma
alpha = 1./gamma_t/gamma_t          # First order mom. comp. factor

# Bunch parameters
N_b = 1e9                           # Intensity
N_p = 50000                         # Macro-particles
tau_0 = 0.4e-9                      # Initial bunch length, 4 sigma [s]

# Simulation Objects --------------------------------------------------------------------------------------------------
ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=1)
rf = RFStation(ring, [h], [V], [dphi])

beam = Beam(ring, N_p, N_b)
profile = Profile(beam, CutOptions(n_slices=100),
                 FitOptions(fit_option='gaussian'))


phase = np.array([0, 0, 0])
gain = np.array([0, +3, -3])
names = ['Base', '+3 dB', '-3 dB']

n = 1
Hs = []
freqs = []

for i in range(len(names)):
    G_a = 6.8e-6
    G_a_in_dB = sut.linear_to_dB(G_a) + gain[i]
    G_a = sut.dB_to_linear(G_a_in_dB)
    print(G_a, phase[i])
    # Cavity Controller
    RFFB = LHCRFFeedback(open_loop=False, open_otfb=True, excitation=True,
                         G_a=G_a, G_d=10, tau_a=170e-6, tau_d=400e-6,
                         d_phi_ad=phase[i])

    CL = LHCCavityLoop(rf, profile, G_gen=1, f_c=rf.omega_rf[0,0]/(2*np.pi),
                       I_gen_offset=0, n_cav=8, Q_L=60000, R_over_Q=45,
                       tau_loop=650e-9,
                       n_pretrack=1, RFFB=RFFB)

    # Noise Injection -------------------------------------------------------------------------------------------------
    CL.track_no_beam_excitation(n_turns=100 * n)

    TF = TransferFunction(CL.V_EXC_IN, CL.V_EXC_OUT, T_s=CL.T_s, plot=PLT_MODEL_TF)
    TF.analyse(data_cut=0)

    print(names[i])
    Hs.append(TF.H_est)
    freqs.append(TF.f_est)

Hs = np.array(Hs, dtype=complex)
freqs = np.array(freqs)

f_s = 1e-6
plt.figure()
plt.title('Closed Loop Response')
colors = ['black', 'r', 'b', 'g', 'y']

for i in range(len(names)):
    plt.plot(freqs[i, :] * f_s, 20 * np.log10(np.abs(Hs[i, :])), label=names[i], color=colors[i])

plt.xlim((-1.5e6 * f_s, 1.5e6 * f_s))
plt.ylim((-25, 3))
plt.xlabel(r'Frequency [MHz]')
plt.ylabel(r'Gain [dB]')
plt.legend()



plt.show()