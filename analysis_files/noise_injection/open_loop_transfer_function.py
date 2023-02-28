'''
File to measure the open loop response of the LHC RFFB.

Author: Birk Emil Karlsen-Bæck
'''

# Imports -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Homemade files
import utility_files.measurement_utilites as mut
import utility_files.transfer_functions as tfu

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

# Cavity Controller
RFFB = LHCRFFeedback(open_loop=True, open_otfb=True, G_a=6.8e-6 * 1.0,
                     tau_a=0.00010762298111463399,
                     tau_d=0.0004087824351297405,
                     excitation=True, G_d=0.5, d_phi_ad=5,
                     G_o=10)

print(rf.omega_rf[0,0]/(2*np.pi))
print(-0.1367974192629025 * 2518300670.2580547)

CL = LHCCavityLoop(rf, profile, G_gen=2, f_c=rf.omega_rf[0,0]/(2*np.pi),
                   I_gen_offset=0, n_cav=8, Q_L=17082, R_over_Q=45,
                   tau_loop=7.163657773063989e-07 * 1, n_pretrack=1, RFFB=RFFB)

# Noise Injection -----------------------------------------------------------------------------------------------------

CL.track_no_beam_excitation(n_turns=200)


TF = TransferFunction(CL.V_EXC_IN, CL.V_EXC_OUT, T_s=CL.T_s, plot=False)
TF.analyse(data_cut=0)

plt.figure()
plt.title('Excitation Input')
plt.plot(CL.V_EXC_IN.real)
plt.plot(CL.V_EXC_IN.imag)

plt.figure()
plt.title('Excitation Output')
plt.plot(CL.V_EXC_OUT.real)
plt.plot(CL.V_EXC_OUT.imag)


# Closed-loop Measurements --------------------------------------------------------------------------------------------
cavID = np.array(['1B1.json'])#, '1B2.json', '2B1.json', '2B2.json', '3B1.json', '3B2.json', '4B1.json', '4B2.json',
                  #'5B1.json', '5B2.json', '6B1.json', '6B2.json', '7B1.json', '7B2.json', '8B1.json', '8B2.json'])

file_dir = '../../transfer_function_measurements/open_loop/'
measurement_length = 662

Hs = np.zeros((len(cavID), measurement_length), dtype=complex)
freqs = np.zeros((len(cavID), measurement_length))

for i in range(len(cavID)):
    H_i, H_phase_i, freq_i = mut.import_tf_measurement(file_dir, cavID[i])

    #H_i = mut.smooth_out_central_peak(H_i, freq_i, 1, 10)

    print(f'\nCavity {cavID[i][:-5]}')
    mut.get_tf_measurement_conditions(file_dir, cavID[i], open_loop=True)

    Hs[i, :] = H_i
    freqs[i, :] = freq_i

if PLT_MEAS_TF:
    plt.figure()
    plt.plot(freqs.T, np.abs(Hs.T))


# Model-Measurement Comparison ----------------------------------------------------------------------------------------

H_meas_avg = np.mean(Hs, axis=0)
H_meas_std = np.std(Hs, axis=0)
H_meas_avg = -H_meas_avg

H_a = tfu.H_a(TF.f_est, g_a=CL.G_a, tau_a=CL.tau_a)
H_d = tfu.H_d(TF.f_est, g_a=CL.G_a, g_d=CL.G_d, tau_d=CL.tau_d, dphi_ad=CL.d_phi_ad * 180/np.pi)
Z_cav = tfu.Z_cav(TF.f_est, df=CL.d_omega/(2 * np.pi), f_rf=rf.omega_rf[0, 0]/(2 * np.pi),
                  R_over_Q=CL.R_over_Q, Q_L=CL.Q_L)

H_open = 2 * tfu.H_open(TF.f_est, tau_loop=CL.tau_loop, H_a=H_a, H_d=H_d, Z_cav=Z_cav)

if PLT_COMP:
    f_s = 1e-6

    plt.figure()
    plt.title('Open Loop Response - Absolute')
    plt.plot(freq_i * f_s, 20 * np.log10(np.abs(H_meas_avg)), label='Meas', color='r')
    plt.plot(TF.f_est * f_s, 20 * np.log10(np.abs(TF.H_est)), label='Sim', color='b')
    plt.plot(TF.f_est * f_s, 20 * np.log10(np.abs(H_open)), label='An', color='black')
    plt.legend()
    plt.xlabel(r'Frequency [MHz]')
    plt.ylabel(r'Gain [dB]')
    plt.xlim((-0.4e6 * f_s, 0.4e6 * f_s))
    plt.ylim((-5, 30))

    n_points = len(TF.H_est) // 2
    move_ang = np.angle(TF.H_est, deg=True)[n_points] - np.unwrap(np.angle(H_open, deg=True))[n_points]
    move_ang2 = np.angle(TF.H_est, deg=True)[n_points] - np.unwrap(np.angle(H_meas_avg, deg=True))[len(H_meas_avg) // 2]

    plt.figure()
    plt.title('Open Loop Response - Phase')
    plt.plot(freq_i * f_s, np.angle(H_meas_avg, deg=True), label='Meas', color='r')
    plt.plot(TF.f_est * f_s, np.angle(TF.H_est, deg=True), label='Sim', color='b')
    plt.plot(TF.f_est * f_s, np.unwrap(np.angle(H_open, deg=True)) + move_ang, label='An', color='black')
    plt.legend()
    plt.xlabel(r'Frequency [MHz]')
    plt.ylabel(r'Phase [degrees]')
    plt.xlim((-0.4e6 * f_s, 0.4e6 * f_s))
    plt.ylim((-360, 360))

    plt.figure()
    plt.title('Open Loop Response - Real')
    plt.plot(freq_i * f_s, H_meas_avg.real, label='Meas', color='r')
    plt.plot(TF.f_est * f_s, TF.H_est.real, label='Sim', color='b')
    plt.plot(TF.f_est * f_s, H_open.real, label='An', color='black')
    plt.legend()
    plt.xlabel(r'Frequency [MHz]')
    plt.ylabel(r'Real [-]')
    plt.xlim((-0.4e6 * f_s, 0.4e6 * f_s))

    plt.figure()
    plt.title('Open Loop Response - Imaginary')
    plt.plot(freq_i * f_s, H_meas_avg.imag, label='Meas', color='r')
    plt.plot(TF.f_est * f_s, TF.H_est.imag, label='Sim', color='b')
    plt.plot(TF.f_est * f_s, H_open.imag, label='An', color='black')
    plt.legend()
    plt.xlabel(r'Frequency [MHz]')
    plt.ylabel(r'Imag [-]')
    plt.xlim((-0.4e6 * f_s, 0.4e6 * f_s))




plt.show()
