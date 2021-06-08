import numpy as np
from DualColorAnalysis import DualColorPdfRatio
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

filenames_optical_response = {
    'dichroic': 'spectral-response_dichroic.csv',
    'blue bandpass': 'spectral-response_blue-bandpass.csv',
    'red bandpass': 'spectral-response_red-bandpass.csv',
}

"ratio model"
l_C = 624.0  # transition wavelength of dichroic (for simplified ideal model, use_filter=False)
l_B, sigma_B = 597.0, 10.2  # center wavelength and spectral width of blue emitter
l_R, sigma_R = 635.0, 10.2  # center wavelength and spectral width of red emitter
y1_arr, y2_arr = np.meshgrid(np.linspace(0, 1, num=res), np.linspace(1, I_max, num=res))
p = DualColorPdfRatio(y1_arr, y2_arr, l_B, l_R, sigma_B, sigma_R, l_C=None, filenames_optical_response=filenames_optical_response)  # use complete spectral response information
# p = DualColorPdfRatio(y1_arr, y2_arr, l_B, l_R, sigma_B, sigma_R, l_C=l_C)  # use ideal spectral response (hard transitions)
pdf_ratio = p.pdf_ratio(samples)

"plot ratio-intensity distributions"
fig, ax = plt.subplots(1, 1, facecolor='none', figsize=(5, 4))
ax.pcolor(y2_arr, y1_arr, pdf_ratio, shading='auto')
ax.grid(True, linestyle='--', color='w', alpha=0.45)
ax.set_xlabel('Intensity [photons / ms]')
ax.set_ylabel('Relative strength of red emitter [%]')
fig.canvas.toolbar.zoom()
fig.tight_layout()
