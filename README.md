For details of method, see publication:
Ryan, D. P. et al. A framework for quantitative analysis of spectral data in two channels. Appl. Phys. Lett. 117, 024101 (2020).
DOI: 10.1063/5.0013998

Essential files:
DualColorAnalysis.py: Loads spectral response functions and generate wavelength lookup table.
    Classes:
        SpectralComponents(): Returns optical transfer functions using response data of optical components. 
        DualColorPdfSpectral(): Class to generate a spectral-intensity distribution.
        DualColorPdfRatio():  Class to generate a ratio-intensity distribution

Optional files:
spectral-response_dichroic.csv: Transmission curve of dichroic beamsplitter.
spectral-response_blue-bandpass.csv: Transmission curve of bandpass filter on blue channel.
spectral-response_red-bandpass.csv: Transmission curve of bandpass filter on red channel.

Example files:
calc_spectral-params.py: Determine the calibration parameters from spectral response data.
generate_spectral-PDF.py: Generate a spectral-intensity distribution from samples. 
generate_ratio-PDF.py: Generate a ratio-intensity distribution from samples.
