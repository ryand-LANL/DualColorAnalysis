import numpy as np
from scipy.stats import gamma, poisson
from scipy.special import erf, erfinv
from scipy.interpolate import interp1d


class SpectralComponents:
    """Calibrate channel intensity ratios according to filter set."""

    def __init__(self, filenames_optical_response, wl_rng=[500, 700], res=128):
        """Initialize calibration."""

        "store filenames"
        self.filenames_optical_response = filenames_optical_response

        "import raw data"
        self.raw_data = dict()
        self.raw_data['dichroic'] = np.loadtxt(self.filenames_optical_response['dichroic'], skiprows=1, delimiter=',')
        self.raw_data['blue BP'] = np.loadtxt(self.filenames_optical_response['blue bandpass'], skiprows=1, delimiter=',')
        self.raw_data['red BP'] = np.loadtxt(self.filenames_optical_response['red bandpass'], skiprows=1, delimiter=',')

        "interpolate optics data"
        self.T = dict()
        self.T['dichroic'] = interp1d(self.raw_data['dichroic'][:, 0], self.raw_data['dichroic'][:, 1],
                                      kind='cubic', fill_value=0.0, bounds_error=False)
        self.T['blue BP'] = interp1d(self.raw_data['blue BP'][:, 0], self.raw_data['blue BP'][:, 1],
                                     kind='cubic', fill_value=0.0, bounds_error=False)
        self.T['red BP'] = interp1d(self.raw_data['red BP'][:, 0], self.raw_data['red BP'][:, 1],
                                    kind='cubic', fill_value=0.0, bounds_error=False)

        "common wavelength axis"
        self.wl = np.linspace(wl_rng[0], wl_rng[1], num=res)

    def wl_calib(self, sigma):
        r = np.zeros(self.wl.shape)
        I_R_norm = np.zeros(self.wl.shape)
        I_B_norm = np.zeros(self.wl.shape)

        for idx, l_0 in enumerate(self.wl):
            g = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(self.wl - l_0) ** 2 / (2 * sigma ** 2))
            I_R_norm[idx] = (self.T['dichroic'](self.wl) * g * self.T['red BP'](self.wl)).sum() * (self.wl[1] - self.wl[0])
            I_B_norm[idx] = ((1 - self.T['dichroic'](self.wl)) * g * self.T['blue BP'](self.wl)).sum() * (self.wl[1] - self.wl[0])
            r[idx] = I_R_norm[idx] - I_B_norm[idx]

        arg_min, arg_max = np.argmin(r), np.argmax(r)  # trim to single-valued region

        return [r, self.wl, I_R_norm, I_B_norm], interp1d(r[arg_min:arg_max], self.wl[arg_min:arg_max],
                                                          fill_value=np.nan, bounds_error=False)


class DualColorPdfSpectral:

    def __init__(self, y1_arr, y2_arr, calib):
        self.y1_arr = y1_arr
        self.y2_arr = y2_arr
        self.calib = calib
        self.a = calib['a']
        self.b = calib['b']
        self.c = calib['c']
        self.d = calib['d']

    def pdf_spectral(self, samples, mode='gamma'):

        f = np.zeros(self.y1_arr.shape)
        a, b, c, d = self.a, self.b, self.c, self.d
        x1b = (self.y2_arr / 2) * (1 + (erf((self.y1_arr - a) / b) - d) / c)
        x2b = (self.y2_arr / 2) * (1 - (erf((self.y1_arr - a) / b) - d) / c)

        for s in samples:
            x1, x2 = s
            y1 = a + b * erfinv(c * (x1 - x2) / (x1 + x2) + d)
            y2 = x1 + x2
            det = y2 * np.exp(-(y1 - a)**2 / b**2) / (b * c * np.sqrt(np.pi))

            if np.isfinite([x1, x2, det]).all():  # sanity check

                if mode == 'gamma':
                    f_tmp = det * gamma.pdf(x1b, x1 + 1) * gamma.pdf(x2b, x2 + 1)  # gamma approx.
                elif mode == 'gaussian':
                    f_tmp = det * np.exp(-(x1 - x1b) ** 2 / (2 * x1b)) * np.exp(-(x2 - x2b) ** 2 / (2 * x2b)) / (
                            2 * np.pi * np.sqrt(x1b * x2b))  # Gaussian approx.
                elif mode == 'poisson':
                    f_tmp = det * poisson.pmf(int(x1), x1b) * poisson.pmf(int(x2), x2b)
                else:
                    print("Mode must be `gamma', `gaussian', or `poisson'.")
                    raise

                f += f_tmp

        return f * (self.y1_arr[0, 1] - self.y1_arr[0, 0]) * (self.y2_arr[1, 0] - self.y2_arr[0, 0]) / len(samples)


class DualColorPdfRatio:

    def __init__(self, y1_arr, y2_arr, l_B, l_R, sigma_B, sigma_R, l_C=None, filenames_optical_response=None):
        self.y1_arr = y1_arr
        self.y2_arr = y2_arr
        self.l_C = l_C
        self.l_B = l_B
        self.l_R = l_R
        self.sigma_B = sigma_B
        self.sigma_R = sigma_R

        if filenames_optical_response:  # use real filter response data
            s = SpectralComponents(filenames_optical_response)
            g_B = (1 / np.sqrt(2 * np.pi * self.sigma_B ** 2)) * np.exp(-(s.wl - self.l_B) ** 2 / (2 * self.sigma_B ** 2))
            g_R = (1 / np.sqrt(2 * np.pi * self.sigma_R ** 2)) * np.exp(-(s.wl - self.l_R) ** 2 / (2 * self.sigma_R ** 2))

            int_B_upperhalf = (s.T['dichroic'](s.wl) * g_B * s.T['red BP'](s.wl)).sum() * (s.wl[1] - s.wl[0])
            int_B_lowerhalf = ((1 - s.T['dichroic'](s.wl)) * g_B * s.T['blue BP'](s.wl)).sum() * (s.wl[1] - s.wl[0])
            int_R_upperhalf = (s.T['dichroic'](s.wl) * g_R * s.T['red BP'](s.wl)).sum() * (s.wl[1] - s.wl[0])
            int_R_lowerhalf = ((1 - s.T['dichroic'](s.wl)) * g_R * s.T['blue BP'](s.wl)).sum() * (s.wl[1] - s.wl[0])

            self.C1 = -int_B_upperhalf + int_B_lowerhalf
            self.C2 = self.C1 + int_R_upperhalf - int_R_lowerhalf

        else:  # l_C must be specified
            if l_C:
                self.C1 = erf((self.l_C - self.l_B) / (np.sqrt(2) * self.sigma_B))
                self.C2 = self.C1 - erf((self.l_C - self.l_R) / (np.sqrt(2) * self.sigma_R))
            else:
                raise ValueError('l_C must be specified')

    def pdf_ratio(self, samples, mode='gamma'):

        f = np.zeros(self.y1_arr.shape)
        C1, C2 = self.C1, self.C2
        x1b = (self.y2_arr / 2) * (1 + self.y1_arr * C2 - C1)
        x2b = (self.y2_arr / 2) * (1 - self.y1_arr * C2 + C1)

        for s in samples:
            x1, x2 = s
            y2 = x1 + x2
            det = y2 * C2 / 2

            if np.isfinite([x1, x2, det]).all():  # sanity check

                if mode == 'gamma':
                    f_tmp = det * gamma.pdf(x1b, x1 + 1) * gamma.pdf(x2b, x2 + 1)  # gamma approx.
                elif mode == 'gaussian':
                    f_tmp = det * np.exp(-(x1 - x1b)**2 / (2 * x1b)) * np.exp(-(x2 - x2b)**2 / (2 * x2b)) / (2 * np.pi * np.sqrt(x1b * x2b))  # Gaussian approx.
                elif mode == 'poisson':
                        f_tmp = det * poisson.pmf(int(x1), x1b) * poisson.pmf(int(x2), x2b)
                else:
                    print("Mode must be `gamma', `gaussian', or `poisson'.")
                    raise

                f += f_tmp

        return f * (self.y1_arr[0, 1] - self.y1_arr[0, 0]) * (self.y2_arr[1, 0] - self.y2_arr[0, 0]) / len(samples)
