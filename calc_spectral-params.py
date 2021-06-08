import matplotlib.pyplot as plt
from DualColorAnalysis import SpectralComponents
import numpy as np
import scipy.optimize
import scipy.special


def fct(x, data, r):
    """Generalize form of spectral anisotropy expression for fitting."""

    a, b, c, d = x
    fct_fit = a + b * scipy.special.erfinv(c * r + d)

    return np.nansum((fct_fit - data)**2)

filenames_optical_response = {
    'dichroic': 'spectral-response_dichroic.csv',
    'blue bandpass': 'spectral-response_blue-bandpass.csv',
    'red bandpass': 'spectral-response_red-bandpass.csv',
}

sigma = 10.2  # spectral width of emitter
p = SpectralComponents(filenames_optical_response)  # load optical response curves
[eta, wl, I_red, I_blue], wl_calib = p.wl_calib(sigma)  # calculate anisotropy curves

"fit general wavelength model function"
p0 = [620, 20, 1, 2e-2]  # initial estimates of fit parameters
opt = scipy.optimize.minimize(fct, p0, args=(wl_calib(eta), eta))
fct_opt = opt.x[0] + opt.x[1] * scipy.special.erfinv(opt.x[2] * eta + opt.x[3])
print(f'a: {opt.x[0]:.2f}, b: {opt.x[1]:.2f}, c: {opt.x[2]:.3f}, d: {opt.x[3]:.3f}')

"plot fit"
fig, ax = plt.subplots(1, 1, facecolor='none', figsize=(5, 3.5))
ax.plot(eta, wl_calib(eta), linestyle='-', label='Empirical')
ax.plot(eta, fct_opt, linestyle=':', label='Fit to empirical')
ax.plot(eta, p0[0] + np.sqrt(2) * p0[1] * scipy.special.erfinv(eta), linestyle='-', label='Ideal dichroic', color='k')
ax.grid(True, linestyle='--', color='k', alpha=0.45)
ax.legend()
ax.set_xlabel(r'$\eta$')
ax.set_ylabel('Wavelength [nm]')
fig.canvas.toolbar.zoom()
fig.tight_layout()
