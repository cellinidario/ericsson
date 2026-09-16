Mapping between physical blocks and surrogate model components:

Physical Block                     | Digital Surrogate Implementation
-------------------------------------------------------------------------
Transmitter Filter (Driver) / RX   | 1D Convolution (Conv1d) acting as FIR filter
DAC and ADC                        | Straight-Through Estimator (STE) quantizers
Mach-Zehnder Modulator (MZM)       | Cosine function with Extinction Ratio (ER) and chirp parameter
Optical Fiber (Chromatic Disp.)    | Implemented via Fast Fourier Transform (FFT) in frequency domain
Optical Noise (ASE)                | Additive noise loaded directly onto the optical field
Square-law Photodetector (PD)      | Quadratic activation function (absolute squared value)
