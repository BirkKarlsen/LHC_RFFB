'''
File to test the RFFB without the OTFB and without Beam to check sanity checks of the basic model.

Author: Birk Emil Karlsen-Bæck
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Homemade files
import utility_files.measurement_utilites as mut
import utility_files.signal_utilities as sut
import utility_files.transfer_functions as tfu

from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, FitOptions, CutOptions
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation

# Cavity Controller
from blond.llrf.cavity_feedback import LHCRFFeedback, LHCCavityLoop
import blond.llrf.cavity_feedback_lhctmp as tmp
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
ANALYTIC_TF = False

# Parameters ----------------------------------------------------------------------------------------------------------
# Machine and RF parameters
C = 26658.883                       # Machine circumference [m]
p_s = 450e9                         # Synchronous momentum [eV/c]
h = 35640                           # Harmonic number
V = 1.5e6                             # RF voltage [V]
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
G_a = 6.8e-6
G_a_in_dB = sut.linear_to_dB(G_a) + 0
G_a = sut.dB_to_linear(G_a_in_dB)
G_d = 10
d_phi_ad = 0
G_gen = 1
tau_loop = 650e-9 * 1
tau_a = 170e-6
tau_d = 400e-6

RFFB = LHCRFFeedback(open_loop=True, open_otfb=False, G_a=G_a,
                     excitation=True, G_d=G_d, d_phi_ad=d_phi_ad)

CL = LHCCavityLoop(rf, profile, G_gen=G_gen, f_c=rf.omega_rf[0, 0]/(2 * np.pi),
                   I_gen_offset=0, n_cav=1, Q_L=60000, R_over_Q=45,
                   tau_loop=tau_loop,
                   n_pretrack=1, RFFB=RFFB, tau_otfb=1.2e-6)



LOAD = False
if not LOAD:
    print('Simulating...')
    CL.track_no_beam_excitation(n_turns=1000)

    TF = TransferFunction(CL.V_EXC_IN, CL.V_EXC_OUT, T_s=CL.T_s, plot=PLT_MODEL_TF)
    TF.analyse(data_cut=0)

    np.save('LHC_TF_with_OTFB.npy', np.array([TF.H_est, TF.f_est], dtype=complex))
    TF_H_est = TF.H_est
    TF_f_est = TF.f_est
else:
    data = np.load(f'LHC_TF_with_OTFB_{1000}.npy')
    TF_H_est = data[0, :]
    TF_f_est = data[1, :].real

if ANALYTIC_TF:
    H_fb = tfu.FeedbackTF(TF.f_est, g_d=(G_d), tau_d=tau_d, g_a=(G_a),
                          tau_a=tau_a, delta_phi=d_phi_ad)

    H_fb1 = tfu.H_a(TF.f_est, g_a=G_a, tau_a=tau_a) + tfu.H_d(TF.f_est, g_a=G_a, g_d=G_d, tau_d=tau_d, dphi_ad=d_phi_ad)

    H_cav = tfu.CavityTF(TF.f_est, rf.omega_rf[0, 0]/(2 * np.pi), 0, g_oo=1, r_over_q=45, q=60000)
    H_c = H_fb1 * H_cav
    H_cl = H_c / (1 - H_c)

    f_s = 1e-6

    plt.figure()
    plt.title('H fb')
    plt.plot(TF.f_est * f_s, 20 * np.log10(np.abs(H_fb)))
    plt.plot(TF.f_est * f_s, 20 * np.log10(np.abs(H_fb1)))
    plt.xlim((-1.5e6 * f_s, 1.5e6 * f_s))

    plt.figure()
    plt.title('H cav')
    plt.plot(TF.f_est * f_s, 20 * np.log10(np.abs(H_cav)))
    plt.xlim((-1.5e6 * f_s, 1.5e6 * f_s))

    plt.figure()
    plt.title('H closed loop')
    plt.plot(TF.f_est * f_s, 20 * np.log10(np.abs(H_cl)))
    plt.xlim((-1.5e6 * f_s, 1.5e6 * f_s))



if PLT_MODEL_TF:
    plt.figure()
    plt.title('Excitation Input')
    plt.plot(CL.V_EXC_IN.real)
    plt.plot(CL.V_EXC_IN.imag)

    plt.figure()
    plt.title('Excitation Output')
    plt.plot(CL.V_EXC_OUT.real)
    plt.plot(CL.V_EXC_OUT.imag)


if PLT_COMP:
    f_s = 1e-6

    plt.figure()
    plt.title('Closed Loop Response')
    plt.plot(TF_f_est * f_s, 20 * np.log10(np.abs(TF_H_est)), label='Sim', color='b')
    plt.legend()
    plt.xlabel(r'Frequency [MHz]')
    plt.ylabel(r'Gain [dB]')
    #plt.xlim((-1.5e6 * f_s, 1.5e6 * f_s))
    #plt.ylim((-20, 3))

    N_h = len(TF_H_est)
    N_harm = np.array([-102, -82, -62, -42, -22, -2, 2, 22, 42, 62, 82, 102])
    N_harm = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    f_rev = 1/rf.t_rev[0]
    N_sides = int(f_rev / (TF_f_est[1] - TF_f_est[0]))
    print(TF_f_est[N_h//2 - 1], TF_f_est[N_h//2], TF_f_est[N_h//2 + 1])
    plt.figure()
    plt.title('Open Loop Response - Nyquist')
    for i in range(len(N_harm)):
        ind_l = int(N_h//2 - N_sides//2 + N_harm[i] * N_sides)
        ind_r = int(N_h//2 + N_sides//2 + N_harm[i] * N_sides)
        plt.plot(TF_H_est.real[ind_l:ind_r],
                 TF_H_est.imag[ind_l:ind_r])
    plt.grid()
    #plt.xlim((-1, 1))
    #plt.ylim((-1, 1))



plt.show()