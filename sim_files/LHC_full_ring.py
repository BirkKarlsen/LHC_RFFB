'''
File to simulate the full ring and compare with I. Karpov paper:
"Consequences of longitudinal coupled-bunch instability mitigation on power requirements during the HL-LHC filling"

Author: Birk Emil Karlsen-Bæck
'''

# Imports -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from blond.beam.beam import Beam, Proton
from blond.beam.distributions import matched_from_line_density
from blond.beam.profile import Profile, CutOptions
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.llrf.cavity_feedback import LHCRFFeedback, LHCCavityLoop
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF

plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })

# Parameters ----------------------------------------------------------------------------------------------------------
# Accelerator parameters
C = 26658.883                       # Machine circumference [m]
p_s = 6.5e12                        # Synchronous momentum [eV/c]
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
delta_omega = -3480                 # Detuning caused by 12 bunches [Hz]
G_gen = 1
tau_o = 110e-6

show_turn = 0


# Beam parameters
N_p = 1.2e11                        # Bunch intensity [p/b]
N_p_tmp = 1.2e11                    # Bunch intensity [p/b]
mu = 1.5                            # Binomial exponent
N_buckets = h
bl = 1e-9

# Injected beam:
filling_scheme = np.array([12, 3 * 36, 4 * 36, 4 * 36, 36, 4 * 36, 4 * 36, 4 * 36, 4 * 36, 36, 4 * 36, 4 * 36, 4 * 36, 4 * 36, 36, 4 * 36, 4 * 36, 4 * 36, 4 * 36])
N_bunches = np.sum(filling_scheme)
bunch_spacing = 10
ps_batch_spacing = 9 * bunch_spacing
sps_batch_spacing = 39 * bunch_spacing

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
beam_single_bunch = Beam(ring, N_m, N_p)
tracker = RingAndRFTracker(rfstation, beam_single_bunch)
full_tracker = FullRingAndRF([tracker])
matched_from_line_density(beam_single_bunch, full_ring_and_RF=full_tracker, bunch_length=bl,
                          line_density_exponent=mu, line_density_type='binomial')

# Generate full beam with given filling scheme
print(f'Generating filling scheme with {N_bunches} bunches in total...\n')
beam = Beam(ring, N_m * N_bunches, N_p * N_bunches)
ddt = 0 * rfstation.t_rf[0, 0]
Delta_t = 0
l = 0
ps_max_batch_length = 36
for i in range(len(filling_scheme)):
    print(f'Generating batch {i + 1} of {filling_scheme[i]} bunches...')
    N_bunch_in_batch = filling_scheme[i]
    if N_bunch_in_batch / ps_max_batch_length < 1:
        for j in range(N_bunch_in_batch):
            beam.dE[l * N_m: (l + 1) * N_m] = beam_single_bunch.dE
            beam.dt[l * N_m: (l + 1) * N_m] = beam_single_bunch.dt + Delta_t
            l += 1
            if j != N_bunch_in_batch - 1:
                Delta_t += rfstation.t_rf[0, 0] * bunch_spacing
        Delta_t += rfstation.t_rf[0, 0] * sps_batch_spacing
    else:
        if N_bunch_in_batch % ps_max_batch_length == 0:
            N_ps_batches = N_bunch_in_batch // ps_max_batch_length
        else:
            N_ps_batches = N_bunch_in_batch // ps_max_batch_length + 1

        for k in range(N_ps_batches):
            if k == 0:
                N_bunches_in_ps_batch = N_bunch_in_batch - ps_max_batch_length * (N_ps_batches - 1)
            else:
                N_bunches_in_ps_batch = ps_max_batch_length

            for j in range(N_bunches_in_ps_batch):
                beam.dE[l * N_m: (l + 1) * N_m] = beam_single_bunch.dE
                beam.dt[l * N_m: (l + 1) * N_m] = beam_single_bunch.dt + Delta_t
                l += 1
                if j != N_bunches_in_ps_batch - 1:
                    Delta_t += rfstation.t_rf[0, 0] * bunch_spacing

            if k != N_ps_batches - 1:
                Delta_t += rfstation.t_rf[0, 0] * ps_batch_spacing
        Delta_t += rfstation.t_rf[0, 0] * sps_batch_spacing


# Beam Profile
profile = Profile(beam, CutOptions(cut_left=-0.5 * rfstation.t_rf[0, 0] + ddt,
                                   cut_right=(N_buckets + 0.5) * rfstation.t_rf[0, 0] + ddt,
                                   n_slices=(N_buckets + 1) * 2**7))
profile.track()
plt.figure()
plt.plot(profile.bin_centers, profile.n_macroparticles * beam.ratio)
plt.show()


# LHC Cavity Controller
print(f'\nPre-tracking Cavity Loop...\n')
RFFB = LHCRFFeedback(G_a=G_a, G_d=G_d, tau_d=tau_d, tau_a=tau_a, alpha=a_comb, tau_o=tau_o,
                     open_otfb=False, G_o=G_otfb, mu=-10, open_tuner=False)

CL = LHCCavityLoop(rfstation, profile, RFFB=RFFB,
                   f_c=rfstation.omega_rf[0, 0]/(2 * np.pi) + delta_omega,
                   Q_L=Q_L, tau_loop=tau_loop, n_pretrack=100, tau_otfb=tau_comp,
                   G_gen=G_gen)

CL.rf_beam_current()
I_rf_pk = np.max(np.absolute(CL.I_BEAM))
print(f'Peak RF beam current is {I_rf_pk:.3f} A')
d_f = LHCCavityLoop.half_detuning(I_rf_pk, 45, rfstation.omega_rf[0, 0]/(2*np.pi), V/8)
print(f'Optimal theoretical detuning is {d_f/1e3:.3f} kHz\n')


print(f'Simulating...\n')
detuning = np.zeros(N_t)
dt_print = 10
for i in tqdm(range(N_t)):
    CL.track()

    detuning[i] = CL.detuning * rfstation.omega_rf[0, 0] / (2 * np.pi)

    #if i%dt_print == 0:
    #    print(f'Turn {i}')


print(f'\nFinal detuning was {detuning[-1]/1e3:.3f} kHz which is {detuning[-1]/d_f * 100:.1f}% of theoretical value')

plt.figure()
plt.title('Detuning')
plt.plot(detuning/1e3, c='black')
plt.xlabel(r'Turns [-]')
plt.ylabel(r'$\Delta f$ [kHz]')
plt.xlim((0, 100))
plt.grid()

t = np.linspace(0, 2 * rfstation.t_rev[0], 2 * h//10) * 1e6

plt.figure()
plt.title('Power')
plt.plot(t, CL.generator_power()/1e3, c='black')
plt.ylim((0, 300))
plt.xlim((0, 100))
plt.xlabel(r'$\Delta t$ [$\mu$s]')
plt.ylabel(r'Power [kW]')
plt.grid()


plt.figure()
plt.title('Power phase')
plt.plot(t, np.angle(CL.I_GEN, deg=True), c='black')
plt.ylim((-100, 100))
plt.xlim((0, 100))
plt.xlabel(r'$\Delta t$ [$\mu$s]')
plt.ylabel(r'Phase [degrees]')
plt.grid()

plt.figure()
plt.title('Voltage amplitude')
plt.plot(t, np.abs(CL.V_ANT)/1e6, c='black')
plt.ylim((1.240, 1.250))
plt.xlim((0, 100))
plt.xlabel(r'$\Delta t$ [$\mu$s]')
plt.ylabel(r'$V_{rf}$ [MV]')
plt.grid()


plt.figure()
plt.title('Voltage phase')
plt.plot(t, np.angle(CL.V_ANT, deg=True), c='black')
plt.ylim((-1, 1))
plt.xlim((0, 100))
plt.xlabel(r'$\Delta t$ [$\mu$s]')
plt.ylabel(r'Phase [degrees]')
plt.grid()


plt.show()