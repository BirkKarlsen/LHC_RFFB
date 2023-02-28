'''
File to check that the response of the FIR filter is the same as the LaPlace domain.

Author: Birk Emil Karlsen-Bæck
'''

# Import
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as spf

from blond.llrf.signal_processing import fir_filter_lhc_otfb_coeff

def H_LPF(f, N_taps=63, t_bb=10):
    taps = fir_filter_lhc_otfb_coeff(n_taps=N_taps)
    s = 1j * 2 * np.pi * f

    out = np.exp((N_taps - 1) * t_bb * s / 2)
    for i in range(N_taps):
        out += taps[i] * np.exp(-i * t_bb * s)

    return out


# Make signal
n_points = 3564
dt_factor = 1/10

t = np.linspace(1, dt_factor * n_points, n_points)
sig = np.zeros(n_points, dtype=complex)
sig[n_points//2:] = 1
taps = fir_filter_lhc_otfb_coeff()

time_out = np.zeros(sig.shape, dtype=complex)
for i in range(len(sig)):
    for k in range(len(taps)):
        time_out[i] += taps[k] * sig[i - k]


sig_fourier = spf.fftshift(spf.fft(sig))
f = spf.fftshift(spf.fftfreq(n_points, d=t[1]-t[0]))
H_lpf = H_LPF(f, t_bb=t[1] - t[0])


fourier_out = sig_fourier * H_lpf
fourier_result = spf.ifftshift(spf.ifft(fourier_out))



plt.figure()
plt.title('H_{LPF}')
plt.plot(f, H_lpf.real, c='r')
plt.plot(f, H_lpf.imag, c='b')

plt.figure()
plt.title('Signal')
plt.plot(t, sig.real, c='r')
plt.plot(t, sig.imag, c='b')

plt.figure()
plt.title('Signal Fourier Transformed')
plt.plot(f, sig_fourier.real, c='r')
plt.plot(f, sig_fourier.imag, c='b')

plt.figure()
plt.title('Result time-domain')
plt.plot(t, time_out.real, c='r')
plt.plot(t, time_out.imag, c='b')

plt.figure()
plt.title('Result Fourier-domain')
plt.plot(t, fourier_result.real, c='r')
plt.plot(t, fourier_result.imag, c='b')

plt.show()