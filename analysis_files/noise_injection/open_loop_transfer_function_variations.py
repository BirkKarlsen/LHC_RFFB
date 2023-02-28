'''
File to measure the open loop response of the LHC RFFB.

Author: Birk Emil Karlsen-Bæck
'''

# Imports -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Homemade files
import utility_files.measurement_utilites as mut

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

n = 2
Q_Ls = [10000, 20000, 60000, 100000]
Hs = []
freqs = []

for i in range(len(Q_Ls)):
    # Cavity Controller
    print(Q_Ls[i])
    RFFB = LHCRFFeedback(open_loop=True, open_otfb=True, excitation=True,
                         G_a=6.79e-6, G_o=10,  G_d=10,
                         tau_a=170e-6, tau_d=400e-6, d_phi_ad=0)


    CL = LHCCavityLoop(rf, profile, G_gen=1, f_c=rf.omega_rf[0,0]/(2*np.pi),
                       I_gen_offset=0, n_cav=8, Q_L=Q_Ls[i], R_over_Q=45,
                       tau_loop=650e-9, n_pretrack=1, RFFB=RFFB)

    CL.track_no_beam_excitation(n_turns=100 * n)
    TF = TransferFunction(CL.V_EXC_IN, CL.V_EXC_OUT, T_s=CL.T_s, plot=False)
    TF.analyse(data_cut=0)

    Hs.append(TF.H_est)
    freqs.append(TF.f_est)


Hs = np.array(Hs, dtype=complex)
print(Hs.shape)
freqs = np.array(freqs)

f_s = 1e-3
plt.figure()
plt.title(f'Open Loop Response')

plt.plot(freqs[0, :] * f_s, 20 * np.log10(np.abs(Hs[0, :])), color='r', label='10k')
plt.plot(freqs[1, :] * f_s, 20 * np.log10(np.abs(Hs[1, :])), color='b', label='20k')
plt.plot(freqs[2, :] * f_s, 20 * np.log10(np.abs(Hs[2, :])), color='g', label='60k')
plt.plot(freqs[3, :] * f_s, 20 * np.log10(np.abs(Hs[3, :])), color='black', label='100k')

#plt.xlim((-0.04e6 * f_s, 0.04e6 * f_s))
#plt.ylim((15, 130))
plt.xlim((-0.04e6 * f_s, 0.04e6 * f_s))
plt.ylim((7.5, 60))

plt.xlabel(r'Frequency [kHz]')
plt.ylabel(r'Gain [dB]')
plt.legend()


plt.show()