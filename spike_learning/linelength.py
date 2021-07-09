#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" linelength.py
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


from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.signal as sp_sig
import scipy.stats as sp_stats
import matplotlib.pyplot as plt
import pyfftw
import pyeisen
from zappy.sigproc import filters
from zappy.vis import sigplot
import functools
print = functools.partial(print, flush=True)

N_KERN=1
MEM_FFT = True


def _reserve_fftw_mem(kernel_len, signal_len, n_kernel=N_KERN, threads=6):
    a = pyfftw.empty_aligned((kernel_len + signal_len, n_kernel), dtype=complex)
    fft = pyfftw.builders.fft(a, axis=0, threads=threads)
    ifft = pyfftw.builders.ifft(a, axis=0, threads=threads)

    return fft, ifft


def extract_LL(signal, Fs, ll_dur=np.array([0.04])):
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

    Returns
    -------
    linelength: numpy.ndarray, shape: [n_sample x n_win x n_chan]
        Line-length of the original signal for the range of line-length window
        sizes provide by the `ll_dur` parameter.
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

    return Xp


def extract_LL_peaks(sigLL, Fs):
    """
    Extract peaks in the line-length signal.

    Parameters
    ----------
    sigLL: numpy.ndarray, shape: [n_sample x n_chan]
        Line-length of the original signal.

    Fs: float
        Sampling frequency of the line-length feature.

    Returns
    -------
    LLpeak_dict: pandas DataFrame
        chan_ind - indicates channel index in the linelength feature matrix
        peak_ind - indicates positional index of the detected peak
        peak_height - indicates linelength value at the detected peak
        peak_width - indicates the half-maximum width on either side of the peak 
            (provided in units of seconds)
    """

    n_s, n_c = sigLL.shape

    LLpeak_dict = {'chan_ind': [],
                   'peak_ind': [],
                   'peak_height': [],
                   'peak_prom': [],
                   'peak_width': []}
    for c_i in tqdm(range(n_c)):
        pks = sp_sig.find_peaks(sigLL[:, c_i])[0]
        pks_hgt = sigLL[pks, c_i]
        pks_prom = sp_sig.peak_prominences(sigLL[:, c_i], pks)
        pks_wdt = sp_sig.peak_widths(sigLL[:, c_i], pks, prominence_data=pks_prom)[0]

        for p_i in range(len(pks)):
            LLpeak_dict['chan_ind'].append(c_i)
            LLpeak_dict['peak_ind'].append(pks[p_i])
            LLpeak_dict['peak_height'].append(pks_hgt[p_i])
            LLpeak_dict['peak_prom'].append(pks_prom[0][p_i])
            LLpeak_dict['peak_width'].append(pks_wdt[p_i] / Fs)
    LLpeak_dict = pd.DataFrame.from_dict(LLpeak_dict)

    return LLpeak_dict


def threshold_LL_peaks(LLpeak_dict):
    """
    Threshold prominence value of line-length peaks.

    Parameters
    ----------
    LLpeak_dict: pandas DataFrame
        Peak features derived from `extract_LL_peaks`.

    frac: float, 0 < frac < 1
        Fraction threshold above which to retain peaks. Can be conservative and
        allow for noisy observations.

    Returns
    -------
    LLpeak_dict: pandas DataFrame
        chan_ind - indicates channel index in the linelength feature matrix
        peak_ind - indicates positional index of the detected peak
        peak_height - indicates linelength value at the detected peak
        peak_width - indicates the half-maximum width on either side of the peak 
            (provided in units of seconds)
    """

    LLpeak_dict['peak_prom_log'] = np.log10(LLpeak_dict['peak_prom'])
    min_max_prom = LLpeak_dict.groupby(['chan_ind'])['peak_prom_log'].max().min()
    thr_val = min_max_prom

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111)
    ax.hist(LLpeak_dict['peak_prom_log'], 100, density=True, color='k');
    ax.axvline(x=thr_val, color='r')
    plt.show()

    return LLpeak_dict[LLpeak_dict['peak_prom_log'] > thr_val]


def plot_peak_detection(signal, sigLL, Fs, LLpeak_entry, dur):
    """
    Plot window around a given line-length peak detection.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_chan]
        Signal recorded from multiple electrodes that was fed into the line-length
        feature extractor.

    sigLL: numpy.ndarray, shape: [n_sample x n_chan]
        Line-length of the original signal.

    Fs: float
    Sampling frequency of the signal.

    LLpeak_entry: pandas Series
        Single entry in the Peak features DataFrame derived from `extract_LL_peaks`.

    dur: float
        Window duration in seconds to flank on either side of the Peak.
    """

    peak_ind = int(LLpeak_entry['peak_ind'])
    chan_ind = int(LLpeak_entry['chan_ind'])

    n_dur = int(dur*Fs)
    t_sig = np.arange(2*n_dur) / Fs - dur

    sl = slice(peak_ind-n_dur, peak_ind+n_dur)
    signal_win = signal[sl, chan_ind]
    sigLL_win = sigLL[sl, chan_ind]

    plt.figure(figsize=(6,6));
    ax = plt.subplot(111)
    ax.plot(t_sig, signal_win)
    ax.plot(t_sig, sigLL_win)
    ax.axvline(x=0)
    plt.show()


def segment_LL_events(LLpeak_dict, Fs, inter_event_interval=0.2):
    """
    Segment line-length peak detections into discrete events based on
    time spacing between peak detection. 

    Parameters
    ----------
    LLpeak_dict: pandas DataFrame
        Peak features derived from `extract_LL_peaks`.

    Fs: float
        Sampling frequency of the line-length feature.

    inter_event_interval: float
        Threshold duration between peak detections to segment individual events.
        Specified in units of seconds (default=0.2 seconds)

    Returns
    -------
    LLpeak_dict: pandas DataFrame
        chan_ind - indicates channel index in the linelength feature matrix
        peak_ind - indicates positional index of the detected peak
        peak_height - indicates linelength value at the detected peak
        peak_width - indicates the half-maximum width on either side of the peak 
            (provided in units of seconds)
        event_id - indicates which segmented event a given peak detection belongs to
    """

    n_inter_event_interval = int(Fs*inter_event_interval)

    sorted_peak_ind = LLpeak_dict['peak_ind'].sort_values()
    sorted_peak_ind_diff = sorted_peak_ind.diff()

    seg_ix = np.flatnonzero(sorted_peak_ind_diff > n_inter_event_interval)
    seg_ix = np.concatenate(([0], seg_ix, [len(sorted_peak_ind_diff)]))

    LLpeak_dict['Event_ID'] = np.nan
    for ii in range(len(seg_ix)-1):
        seg_indices = sorted_peak_ind_diff.iloc[seg_ix[ii]:seg_ix[ii+1]].index
        LLpeak_dict.loc[seg_indices, 'Event_ID'] = ii+1

    return LLpeak_dict


def plot_LL_event(signal, Fs, LLpeak_dict, event_id, win_dur):
    """
    Plot window around a given line-length peak detection.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_chan]
        Signal recorded from multiple electrodes that was fed into the line-length
        feature extractor.

    Fs: float
    Sampling frequency of the signal.

    LLpeak_dict: pandas Series
        Peak features DataFrame derived from `segment_LL_events`.

    event_id: int
        Event to select from the Peak features DataFrame.

    win_dur: float
        Window duration in seconds to flank on either side of the Peak event.
    """

    from copy import copy
    LLpeak_event = LLpeak_dict[LLpeak_dict['Event_ID'] == event_id]
    LLpeak_event_max = LLpeak_event.sort_values(by='peak_prom').iloc[-1]
    LLpeak_event_ind = LLpeak_event_max['peak_ind'].astype(int)

    n_win_dur = int(Fs*win_dur)

    win_start = LLpeak_event_ind-n_win_dur
    win_end = LLpeak_event_ind+n_win_dur
    sl = slice(win_start, win_end)
    signal_ev = signal[sl]

    ax = sigplot.plot_time_stacked(signal_ev, fs=Fs, wsize=signal_ev.shape[0]/Fs, color='k')

    for row in LLpeak_event.iterrows():
        ch_ind = row[1]['chan_ind'].astype(int)
        pk_ind = row[1]['peak_ind'].astype(int) - win_start
        pk_wdt = int(Fs*row[1]['peak_width']/2)
        if pk_ind >= 0:
            ll = copy(ax.lines[ch_ind])
            ll_dat = ll.get_ydata()
            ll_dat[:pk_ind-pk_wdt] = np.nan
            ll_dat[pk_ind+pk_wdt:] = np.nan
            ll.set_color('r')
            ll.set_ydata(ll_dat)
            ll.set_linewidth(ll.get_linewidth()*3)
            ax.lines.append(ll)
    plt.show()
