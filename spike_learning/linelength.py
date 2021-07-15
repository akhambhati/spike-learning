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
from sklearn import mixture
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
        peak_ind - indicates positional index of the detected peak
        peak_height - indicates linelength value at the detected peak
        peak_width - indicates the half-maximum width on either side of the peak 
            (provided in units of seconds)
    """

    sigLL = sigLL.sum(axis=1)
    n_s = sigLL.shape

    pks = sp_sig.find_peaks(sigLL)[0]
    pks_hgt = sigLL[pks]
    pks_prom = sp_sig.peak_prominences(sigLL, pks)
    pks_wdt = sp_sig.peak_widths(sigLL, pks, prominence_data=pks_prom)[0]

    LLpeak_dict = {'peak_ind': pks,
                   'peak_height': pks_hgt,
                   'peak_prom': pks_prom[0],
                   'peak_width': pks_wdt}
    LLpeak_dict = pd.DataFrame.from_dict(LLpeak_dict)

    return LLpeak_dict


def threshold_LL_peaks(LLpeak_dict, method='GMM', n_components=20, frac=0.9):
    """
    Threshold prominence value of line-length peaks.

    Parameters
    ----------
    LLpeak_dict: pandas DataFrame
        Peak features derived from `extract_LL_peaks`.

    method: ['GMM', None]
        GMM - fits the distribution of peak prominences, finds the best fit model,
        and returns the cluster with the greatest mean value. If only one cluster
        is found, then uses the fraction to set a threshold.
        None - uses frac to set a threshold.

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

    if method == 'GMM':
        GMM_models = []
        BIC = []
        n_comp = [*range(1, n_components+1)]
        print('Optimizing GMM...')
        for nc in tqdm(n_comp):
            GMM = mixture.GaussianMixture(n_components=nc, covariance_type='full')
            GMM.fit(LLpeak_dict['peak_prom_log'].values.reshape(-1,1))
            GMM_models.append(GMM)

            bic = GMM.bic(LLpeak_dict['peak_prom_log'].values.reshape(-1,1))
            BIC.append(bic)
        opt_bic = np.argmin(np.array(BIC))
        print('     {} components found.'.format(n_comp[opt_bic]))

        if n_comp[opt_bic] == 1:
            method = None
        else:
            GMM = GMM_models[opt_bic]
            comp_rank = np.argsort(GMM.means_[:,0])
            cl_assign = GMM.predict(LLpeak_dict['peak_prom_log'].values.reshape(-1,1))

            plt.figure(figsize=(6,6))
            ax = plt.subplot(111)
            _, bins, _ = ax.hist(LLpeak_dict['peak_prom_log'], 100, density=True,
                    color='k', histtype='stepfilled');

            g_noise = np.zeros(len(bins))
            for ci in comp_rank[:-1]:
                g_noise += GMM.weights_[ci]*sp_stats.norm.pdf(
                        bins, GMM.means_[ci], np.sqrt(GMM.covariances_[ci,0,0]))
            g_thres = GMM.weights_[comp_rank[-1]]*sp_stats.norm.pdf(
                        bins, GMM.means_[comp_rank[-1]], np.sqrt(GMM.covariances_[comp_rank[-1],0,0]))
            ax.plot(bins, g_noise, color='b')
            ax.plot(bins, g_thres, color='r')
            ax.set_ylabel('Density')
            ax.set_xlabel('Line-Length Peak Prominences')
            plt.show()

            return LLpeak_dict.iloc[cl_assign == comp_rank[-1]].reset_index(drop=True)

    if method is None:
        thr_val = LLpeak_dict['peak_prom_log'].quantile(frac)

        plt.figure(figsize=(6,6))
        ax = plt.subplot(111)
        _, bins, _ = ax.hist(LLpeak_dict['peak_prom_log'], 100, density=True,
                color='k', histtype='stepfilled');
        ax.axvline(x=thr_val, color='r')
        plt.show()

        return LLpeak_dict[LLpeak_dict['peak_prom_log'] > thr_val].reset_index(drop=True)


def calc_LL_IPI(LLpeak_dict):
    """
    Calculate the inter-peak-interval based on timestamps of the peaks.

    Parameters
    ----------
    LLpeak_dict: pandas DataFrame
        Peak features derived from `extract_LL_peaks`. Must have a 'timestamp'
        column.

    Returns
    -------
    LLpeak_dict: pandas DataFrame
        chan_ind - indicates channel index in the linelength feature matrix
        peak_ind - indicates positional index of the detected peak
        peak_height - indicates linelength value at the detected peak
        peak_width - indicates the half-maximum width on either side of the peak 
            (provided in units of seconds)
        timestamp - indicates the absolute time of the peak
        IPI- indicates the latency relative to the previous peak
    """

    LLpeak_dict = LLpeak_dict.sort_values(by='timestamp').reset_index(drop=True)
    LLpeak_dict.loc[1:, 'IPI'] = LLpeak_dict['timestamp'].diff().iloc[1:]
    if LLpeak_dict['IPI'].iloc[1:].apply(lambda x: type(x) == pd.Timedelta).all():
        LLpeak_dict.loc[1:, 'IPI'] = LLpeak_dict['IPI'].iloc[1:].apply(
                    lambda x: pd.Timedelta.total_seconds(x))

    LLpeak_dict = LLpeak_dict.dropna()
    LLpeak_dict['IPI'] = LLpeak_dict['IPI'].astype(np.float)

    return LLpeak_dict


def segment_LL_events(LLpeak_dict, method='GMM', n_components=20, IPI=0.333):
    """
    Segment the occurrence of line-length peaks into events.

    Parameters
    ----------
    LLpeak_dict: pandas DataFrame
        Peak features derived from `extract_LL_peaks`.

    method: ['GMM', None]
        GMM - fits the distribution of inter-peak-interval, finds the best fit model,
        and returns the cluster with the greatest mean value. If only one cluster
        is found, then uses the IPI to set a threshold.
        None - uses IPI to set a threshold.

    IPI: float
        Minimum time between peaks to be considered discrete events.

    Returns
    -------
    LLpeak_dict: pandas DataFrame
        chan_ind - indicates channel index in the linelength feature matrix
        peak_ind - indicates positional index of the detected peak
        peak_height - indicates linelength value at the detected peak
        peak_width - indicates the half-maximum width on either side of the peak 
            (provided in units of seconds)
        timestamp - indicates the absolute time of the peak
        IPI- indicates the latency relative to the previous peak
        event_id - indicates which segmented event a given peak detection belongs to
    """

    LLpeak_dict['IPI_log'] = np.log10(LLpeak_dict['IPI'])

    if method == 'GMM':
        GMM_models = []
        BIC = []
        n_comp = [*range(1, n_components+1)]
        print('Optimizing GMM...')
        for nc in tqdm(n_comp):
            GMM = mixture.GaussianMixture(n_components=nc, covariance_type='full')
            GMM.fit(LLpeak_dict['IPI_log'].values.reshape(-1,1))
            GMM_models.append(GMM)

            bic = GMM.bic(LLpeak_dict['IPI_log'].values.reshape(-1,1))
            BIC.append(bic)
        opt_bic = np.argmin(np.array(BIC))
        print('     {} components found.'.format(n_comp[opt_bic]))

        if n_comp[opt_bic] == 1:
            thr_val = np.log10(IPI)
        else:
            GMM = GMM_models[opt_bic]
            comp_rank = np.argsort(GMM.means_[:,0])
            cl_assign = GMM.predict(LLpeak_dict['IPI_log'].values.reshape(-1,1))

            plt.figure(figsize=(6,6))
            ax = plt.subplot(111)
            _, bins, _ = ax.hist(LLpeak_dict['IPI_log'], 100, density=True,
                    color='k', histtype='stepfilled');

            g_noise = np.zeros(len(bins))
            for ci in comp_rank[:-1]:
                g_noise += GMM.weights_[ci]*sp_stats.norm.pdf(
                        bins, GMM.means_[ci], np.sqrt(GMM.covariances_[ci,0,0]))
            g_thres = GMM.weights_[comp_rank[-1]]*sp_stats.norm.pdf(
                        bins, GMM.means_[comp_rank[-1]], np.sqrt(GMM.covariances_[comp_rank[-1],0,0]))
            ax.plot(bins, g_noise, color='b')
            ax.plot(bins, g_thres, color='r')
            ax.set_ylabel('Density')
            ax.set_xlabel('Inter-Peak Interval (log(s))')
            plt.show()

            thr_val = LLpeak_dict['IPI_log'].iloc[cl_assign == comp_rank[-1]].min()
    else:
        thr_val = np.log10(IPI)
    print('     Applying Inter-Peak Interval of {} sec'.format(10**thr_val))

    seg_ix = np.flatnonzero(LLpeak_dict['IPI_log'] > thr_val)
    seg_ix = np.concatenate((seg_ix, [len(LLpeak_dict)]))

    LLpeak_dict['Event_ID'] = np.nan
    for ii in range(len(seg_ix)-1):
        seg_indices = LLpeak_dict.iloc[seg_ix[ii]:seg_ix[ii+1]].index
        LLpeak_dict.loc[seg_indices, 'Event_ID'] = ii+1

    IEI = np.diff(np.sort(LLpeak_dict.groupby('Event_ID')['timestamp'].min().values))
    plt.hist(np.log10(IEI), 50);
    plt.xlabel('Time (log(s))')
    plt.title('Inter-Event Interval')
    plt.show()

    return LLpeak_dict


def plot_LL_event(signal, Fs, LLpeak_dict, event_id, win_dur, scale=3):
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

    ax = sigplot.plot_time_stacked(signal_ev, fs=Fs, wsize=signal_ev.shape[0]/Fs,
            color='k', scale=scale)

    plt.show()


def align_events(signal, Fs, LLpeak_dict, win_dur, clip_dur):
    """
    Generate a tensor of detections aligned to the fastest falling edge of the
    peak line-length detection.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_chan]
        Signal recorded from multiple electrodes that was fed into the line-length
        feature extractor.

    Fs: float
        Sampling frequency of the signal.

    LLpeak_dict: pandas Series
        Peak features DataFrame derived from `segment_LL_events`.

    win_dur: float
        Window duration in seconds to flank on either side of the Peak event.
    """

    n_win_dur = int(Fs*win_dur)
    n_clip_dur = int(Fs*clip_dur)

    events = []
    events_id = []
    for event_id, LLpeak_event in tqdm(LLpeak_dict.groupby(['Event_ID'])):
        LLpeak_event_max = LLpeak_event.sort_values(by='peak_prom').iloc[-1]
        LLpeak_event_ind = LLpeak_event_max['peak_ind'].astype(int)

        win_start = LLpeak_event_ind-n_win_dur
        win_end = LLpeak_event_ind+n_win_dur
        if (win_start < 0) or (win_end > signal.shape[0]):
            continue
        sl = slice(win_start, win_end)
        signal_ev = signal[sl]

        signal_align = []
        bad = False
        for ch_i in range(signal_ev.shape[1]):
            falling_ind = win_start+np.diff(signal_ev[:, ch_i]).argmin()
            win_start_align = falling_ind-n_clip_dur
            win_end_align = falling_ind+n_clip_dur
            if (win_start_align < 0) or (win_end_align > signal.shape[0]):
                bad = True
            sl = slice(win_start_align, win_end_align)
            signal_align.append(signal[sl, ch_i])
        signal_align = np.array(signal_align).T
        if bad: 
            continue

        events.append(signal_align)
        events_id.append(event_id)
    events = np.array(events)
    events_id = np.array(events_id)

    return events, events_id
