import numpy as np
from DualColorAnalysis import DualColorPdfSpectral
import matplotlib.pyplot as plt

samples = [  # [red channel intensity, blue channel intensity]
    [1.5, 3.0112781954887216],
    [8, 16.06015037593985],
    [20, 40.150375939849624],
    [10, 1.0344827586206897],
    [30, 3.103448275862069],
    [55, 5.689655172413793]
]

I_max = 75  # maximum intensity for PDF
res = 128  # resolution of PDF

calib = {  # params with sigma = 10.2 nm for emitter bandwidth, from calc_spectral-params.py
    'a': 620.02,
    'b': 15.86,
    'c': 1.058,
    'd': 1.6e-2,
}

"wavelength model"
wl_min, wl_max = 585, 655
y1_arr, y2_arr = np.meshgrid(np.linspace(585, 655, num=res), np.linspace(1, I_max, num=res))
p = DualColorPdfSpectral(y1_arr, y2_arr, calib)  # all calibration information, including bandwidth, contained in specification of [a, b, c, d] in class definition
pdf_spectral = p.pdf_spectral(samples)

"plot spectral-intensity distributions"
fig, ax = plt.subplots(1, 1, facecolor='none', figsize=(5, 4))
ax.pcolor(y2_arr, y1_arr, pdf_spectral, shading='auto')
ax.grid(True, linestyle='--', color='w', alpha=0.45)
ax.set_xlabel('Intensity [photons / ms]')
ax.set_ylabel('Wavelength [nm]')
fig.canvas.toolbar.zoom()
fig.tight_layout()
