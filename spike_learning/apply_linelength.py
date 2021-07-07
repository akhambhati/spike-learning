#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" apply_linelength.py
Description: Calculate line-length feature from intracranial EEG.
"""
__author__ = "Ankit N. Khambhati"
__copyright__ = "Copyright 2021, Ankit N. Khambhati"
__credits__ = ["Ankit N. Khambhati"]
__license__ = ""
__version__ = "1.0.0"
__maintainer__ = "Ankit N. Khambhati"
__email__ = ""
__status__ = "Prototype"
    
import numpy as np
import pyfftw
import pyeisen
from zappy.sigproc import filters
import functools
print = functools.partial(print, flush=True)

N_KERN=1
MEM_FFT = True

def _reserve_fftw_mem(kernel_len, signal_len, n_kernel=N_KERN, threads=6):
    a = pyfftw.empty_aligned((kernel_len + signal_len, n_kernel), dtype=complex)
    fft = pyfftw.builders.fft(a, axis=0, threads=threads)
    ifft = pyfftw.builders.ifft(a, axis=0, threads=threads)

    return fft, ifft

def extract_ll(signal, Fs, ll_dur=np.array([0.04]), Fs_resample=None):
    """
    Extract the line-length of a signal for a range of window sizes.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_chan]
        Signal recorded from multiple electrodes that are to be
        low-pass filtered and then decimated.

    Fs: float
        Sampling frequency of the signal, before decimation.

    ll_dur: array(float)
        List of window sizes from which to calculate line-length. Window sizes
        should have units of seconds. 

    Fs_resample: float
        Sampling frequency of the downsampled signal after applying the line-length
        transform.

    Returns
    -------
    linelength: numpy.ndarray, shape: [n_sample x n_win x n_chan]
        Line-length of the original signal for the range of line-length window
        sizes provide by the `ll_dur` parameter.

    Fs: float
        Sampling frequency of the returned line-length signal. May be different
        than the specified Fs_resample parameter.
    """

    ### Create kernel family
    print('- Constructing line-length kernel')
    n_s, n_ch = signal.shape
    n_kr = len(ll_dur)
    Xp = np.zeros((n_kr, n_ch, n_s))

    for dur_ii, dur in enumerate(ll_dur):
        family = {'kernel': np.ones((N_KERN, int(dur * Fs))) / int(dur * Fs),
                  'linelength': {'duration': np.array([dur])}}

        ### Memory setup for convolution
        fft, ifft = [None, None]
        if MEM_FFT:
            print('- Reserving memory for kernel convolution')
            fft, ifft = _reserve_fftw_mem(
                    kernel_len=family['kernel'].shape[1],
                    signal_len=n_s)

        ### Create the linelength dictionary
        print('- Reserving memory for storing kernel coefficients')

        # Iterate over each channel and convolve
        print('- Iteratively convolving line-length kernel with each channel')
        for ch_ii in range(n_ch):
            print('    - {} of {}'.format(ch_ii+1, n_ch))

            # Subset the channel
            subset_sig = signal[:, [ch_ii]]

            # Absolute first-order difference
            subset_sig[:, 0] = np.concatenate(([0], np.abs(np.diff(subset_sig[:, 0]))))

            if (fft is not None) & (ifft is not None):
                out = pyeisen.convolve.fconv(
                        family['kernel'][:,:].T,
                        subset_sig.reshape(-1, 1),
                        fft=fft, ifft=ifft,
                        interp_nan=True)
            else:
                out = pyeisen.convolve.fconv(
                        family['kernel'][:,:].T,
                        subset_sig.reshape(-1, 1),
                        fft=fft, ifft=ifft,
                        interp_nan=True)

            Xp[dur_ii, ch_ii, :] = np.abs(out[:, 0, 0])

    Xp = Xp.transpose((2,0,1))

    # Downsample
    Fs_new = Fs
    if Fs_resample is not None:
        Xpp = []
        for nk in range(Xp.shape[1]):
            Xp_, Fs_new = filters.downsample(Xp[:, nk, :], Fs, Fs_resample)
            Xpp.append(Xp_)
        Xp = np.array(Xpp)
        Xp = Xp.transpose((1,0,2))

    return Xp, Fs_new
