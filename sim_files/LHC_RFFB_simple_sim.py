'''
First simulation with the LHC RFFB model.

Author: Birk Emil Karlsen-Bæck
'''

# Imports ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os

from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions import bigaussian
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF

from blond.llrf.cavity_feedback import LHCRFFeedback, LHCCavityLoop


# Parameters ------------------------------------------------------------------
C = 26658.883                               # Ring circumference [m]
gamma_t = 53.8                              # Transition Gamma [-]
alpha = 1 / gamma_t**2                      # Momentum compaction factor [-]
p_s = 450e9                                 # Synchronous momentum [eV]
h = 35640                                   # 400 MHz harmonic number [-]
V = 16e6                                    # 400 MHz RF voltage [V]
phi = 0                                     # 400 MHz phase [-]

# LHC RFFB parameters


# Beam parameters
bl = 1.2e-9                                     # Bunchlength [s]
N_p = int(1.15e11)                              # Number of protons in one bunch

# Simulation parameters
N_m = int(5e5)                                  # Number of macro-particles for tracking
N_t = 1000                                      # Number of turns to track


# Objects ---------------------------------------------------------------------

# LHC Ring
ring = Ring(C, alpha, p_s, Proton(), n_turns=N_t)

# RF station
rfstation = RFStation(ring, [h], [V], [phi])

# Beam
beam = Beam(ring, N_m, N_p)
bigaussian(ring, rfstation, beam, sigma_dt=bl/4)

# Profile
profile = Profile(beam, CutOptions = CutOptions(cut_left=rfstation.t_rf[0,0] * (1000 - 2.5),
    cut_right=rfstation.t_rf[0,0] * (1000 + 72 * 5 * 4 + 250 * 3 + 125),
    n_slices=int(round(2**7 * (2.5 + 72 * 5 * 4 + 250 * 3 + 125)))))

# LHC RFFB
RFFB = LHCRFFeedback()
lhccl = LHCCavityLoop(rfstation, profile, RFFB=RFFB)





