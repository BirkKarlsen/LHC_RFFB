'''
File to measure the open loop response of the LHC RFFB.

Author: Birk Emil Karlsen-Bæck
'''

# Imports -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, FitOptions, CutOptions
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation

# Cavity Controller
from blond.llrf.cavity_feedback import LHCRFFeedback, LHCCavityLoop
from blond.llrf.transfer_function import TransferFunction

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

# Cavity Controller
RFFB = LHCRFFeedback(open_loop=True, open_otfb=True, G_a=6.8e-6 * 1.0,
                     excitation=True, G_d=10, d_phi_ad=45 * np.pi / 180)

CL = LHCCavityLoop(rf, profile, G_gen=1, f_c=rf.omega_rf[0,0]/(2*np.pi),
                   I_gen_offset=0, n_cav=8, Q_L=20000, R_over_Q=45,
                   tau_loop=650e-9, n_pretrack=1, RFFB=RFFB)

# Noise Injection -----------------------------------------------------------------------------------------------------

CL.track_no_beam_excitation(n_turns=100)


TF = TransferFunction(CL.V_EXC_IN, CL.V_EXC_OUT, T_s=CL.T_s, plot=True)
TF.analyse(data_cut=0)

plt.figure()
plt.title('Excitation Input')
plt.plot(CL.V_EXC_IN.real)
plt.plot(CL.V_EXC_IN.imag)

plt.figure()
plt.title('Excitation Output')
plt.plot(CL.V_EXC_OUT.real)
plt.plot(CL.V_EXC_OUT.imag)





plt.show()