#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" event_learning.py
Description: Learn network representations of spiking events from intracranial EEG.
"""
__author__ = "Ankit N. Khambhati"
__copyright__ = "Copyright 2021, Ankit N. Khambhati"
__credits__ = ["Ankit N. Khambhati"]
__license__ = ""
__version__ = "1.0.0"
__maintainer__ = "Ankit N. Khambhati"
__email__ = ""
__status__ = "Prototype"


import os
import copy
import joblib
import h5py
import numpy as np
import pandas as pd
import torch
import scipy.stats as sp_stats

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, silhouette_samples
import hashlib
import json

from precog.operations.standardizers import NNMFLineLengthStandardizer
from .torch_seqnmf import SeqNMF, SeqNMFTrainer, parallel_model_update_HW
from .utils import dict_hash

eps = np.finfo(float).resolution


def expand_h5(h5_obj, skip_key=[], axis=0):
    for key in h5_obj:
        if key in skip_key:
            continue

        if isinstance(h5_obj[key], h5py.Group):
            expand_h5(h5_obj[key], skip_key=skip_key, axis=axis)

        if isinstance(h5_obj[key], h5py.Dataset):
            if h5_obj[key].maxshape[axis] is None:
                shape = list(h5_obj[key].shape)
                shape[axis] += 1
                shape = tuple(shape)
                h5_obj[key].resize(shape)


def refresh_h5(h5_obj):
    for key in h5_obj:
        if hasattr(h5_obj[key].id, 'refresh'):
            h5_obj[key].id.refresh()
        else:
            refresh_h5(h5_obj[key])


# MOTIF_SHIFT_SCALER_THRESOLD


class MotifLearning():

    def __init__(self, base_path, mode='r', overwrite=False):
        self.base_path = base_path
        self.model_init = False

        if mode == 'r':
            model_exists = self.setup_path(overwrite=False)
        elif mode == 'a':
            model_exists = self.setup_path(overwrite)
        else:
            raise Exception('Model load mode does not exist.')

        if model_exists:
            self.read_model(mode)
            self.model_init = True
        else:
            print('Model does not exist. Must initialize.')

    def setup_path(self, overwrite):
        self.h5_path = '{}.h5'.format(self.base_path)
        self.obj_path = '{}.obj'.format(self.base_path)

        if overwrite:
            try:
                os.remove(self.h5_path)
            except:
                pass

            try:
                os.remove(self.obj_path)
            except:
                pass

        return os.path.exists(self.h5_path) & os.path.exists(self.obj_path)

    def read_model(self, mode):
        self.h5 = h5py.File(self.h5_path, mode, libver='latest', swmr=True)
        obj = joblib.load(self.obj_path)
        self.params = obj['params']
        self.seqnmf_dict = obj['seqnmf_dict']
        self.events_dict = obj['events_dict']

    def cache_model(self):
        self.h5.flush()
        joblib.dump(
                {'params': self.params,
                 'seqnmf_dict': self.seqnmf_dict,
                 'events_dict': self.events_dict},
                self.obj_path)
        return None

    def update_h5_motifsearch(self):
        for model_i, model in enumerate(self.seqnmf_dict['seqnmf_model']):
            for motif_id, hash_id in enumerate(model.hashes):
                hash_ix = np.flatnonzero(
                        self.h5['MotifSearch/motif_hash'][0, :] ==
                        bytes(hash_id, 'utf-8'))
                if len(hash_ix) == 0:
                    expand_h5(
                        self.h5, skip_key=['LineLength', 'timestamp'], axis=1)
                    hash_ix = -1
                    self.h5['MotifSearch/motif_hash'][0, hash_ix] = hash_id
                else:
                    hash_ix = hash_ix[0]

                self.h5['MotifSearch/motif_coef'][-1, hash_ix] = \
                        model.cnmf.W.detach().numpy()[:, motif_id, :]
                self.h5['MotifSearch/exprs_coef_0'][-1, hash_ix] = \
                        model.H0[0, motif_id, :]
                self.h5['MotifSearch/exprs_coef_1'][-1, hash_ix] = \
                        model.H1[0, motif_id, :]
                self.h5['MotifSearch/exprs_coef_2'][-1, hash_ix] = \
                        model.H.detach().numpy()[0, motif_id, :]
                self.h5['Diagnostic/RMSE'][-1, model_i, motif_id, 0] = \
                        model.RMSE_marginal[motif_id]

    def init_model(self, params):
        """
        params 
            -> ecog_params
                -> channel_table
                -> n_channel
                -> sampling_frequency
                -> stream_window_size
                -> stream_window_shift
            -> linelength_params
                -> 'squared_estimator
            -> seqnmf_params
                -> model
                    -> convolutional_window_size
                    -> n_motif
                    -> penalties
                        ->
                -> trainer
                    -> motif_lr
                    -> event_lr
                    -> motif_iter
                    -> event_iter
                    -> beta
        """
        if self.model_init:
            print('Initialized model may not be re-initialized.')
            return None

        ###
        self.params = params

        ###
        fs = self.params['ecog_params']['sampling_frequency']
        n_chan = self.params['ecog_params']['n_channel']
        stream_winsize = int(self.params['ecog_params']['stream_window_size'] * fs)
        seqnmf_winconv = int(
                self.params['seqnmf_params']['convolutional_window_size'] * fs)
        n_evwin = stream_winsize - seqnmf_winconv
        motif_lr = self.params['seqnmf_params']['trainer']['max_motif_lr']
        event_lr = self.params['seqnmf_params']['trainer']['max_event_lr']
        motif_lr_decay = self.params['seqnmf_params']['trainer']['max_motif_lr_decay']

        forget_factor = 1-np.exp(-(1/fs)/self.params['linelength_params']['ewma_norm_window'])
        rescale_threshold = self.params['linelength_params']['rescale_threshold']
        self.LL_Standardizer = NNMFLineLengthStandardizer(
                forget_factor=forget_factor, threshold=rescale_threshold)

        self.burn_in = True

        ###
        seqnmf_dict = {'seqnmf_model': [],
                       'seqnmf_trainer': [],
                       'ensemble_model': [],
                       'ensemble_trainer': []}
        ensemble_rank = 0
        for ms_param in self.params['seqnmf_params']['MotifSearch']:
            seqnmf_dict['seqnmf_model'].append(
                    SeqNMF(
                        n_chan=n_chan,
                        n_sample=stream_winsize-1,
                        n_convwin=seqnmf_winconv,
                        rank=ms_param['model']['n_motif'],
                        coef_normalization=ms_param['model']['coef_normalization'],
                        feat_normalization=ms_param['model']['feat_normalization'],
                        feat_recentering=ms_param['model']['feat_recentering'],
                        motif_competitions=ms_param['model']['motif_competitions'],
                        penalties=ms_param['model']['penalties']
                        ))
            seqnmf_dict['seqnmf_trainer'].append(
                    SeqNMFTrainer(
                        seqnmf_dict['seqnmf_model'][-1],
                        max_motif_lr=motif_lr,
                        max_event_lr=event_lr,
                        max_motif_lr_decay=motif_lr_decay, 
                        beta=self.params['seqnmf_params']['trainer']['beta']
                        ))
            ensemble_rank += ms_param['model']['n_motif']

        ##
        seqnmf_dict['ensemble_model'].append(
                SeqNMF(
                    n_chan=n_chan,
                    n_sample=stream_winsize-1,
                    n_convwin=seqnmf_winconv,
                    rank=ensemble_rank,
                    coef_normalization=self.params['seqnmf_params']['MotifEnsemble']['model']['coef_normalization'],
                    feat_normalization=self.params['seqnmf_params']['MotifEnsemble']['model']['feat_normalization'],
                    feat_recentering=self.params['seqnmf_params']['MotifEnsemble']['model']['feat_recentering'],
                    motif_competitions=self.params['seqnmf_params']['MotifEnsemble']['model']['motif_competitions'],
                    penalties=self.params['seqnmf_params']['MotifEnsemble']['model']['penalties']
                    ))
        seqnmf_dict['ensemble_trainer'].append(
                SeqNMFTrainer(
                    seqnmf_dict['ensemble_model'][-1],
                    max_motif_lr=self.params['seqnmf_params']['trainer']['max_motif_lr'],
                    max_event_lr=self.params['seqnmf_params']['trainer']['max_event_lr'],
                    max_motif_lr_decay=self.params['seqnmf_params']['trainer']['max_motif_lr_decay'],
                    beta=self.params['seqnmf_params']['trainer']['beta']
                    ))

        ###
        h5 = h5py.File(self.h5_path, 'w', libver='latest')

        #
        h5.create_group("LineLength")
        h5.create_group("MotifSearch")
        h5.create_group("Diagnostic")

        #
        h5.create_dataset('timestamp', dtype="f8",
                shape=(0, 2), maxshape=(None, 2))

        h5.create_dataset('LineLength/channel_SIG', dtype="f8",
                shape=(1, stream_winsize-1, n_chan), maxshape=(1, stream_winsize-1, n_chan))


        #
        h5.create_dataset('MotifSearch/motif_hash',
                dtype="S32",
                shape=(1, 0),
                maxshape=(1, None))
        h5.create_dataset('MotifSearch/motif_coef',
                dtype="f8",
                shape=(0, 0, n_chan, seqnmf_winconv),
                maxshape=(None, None, n_chan, seqnmf_winconv))
        h5.create_dataset('MotifSearch/exprs_coef_0',
                dtype="f8",
                shape=(0, 0, n_evwin),
                maxshape=(None, None, n_evwin))
        h5.create_dataset('MotifSearch/exprs_coef_1',
                dtype="f8",
                shape=(0, 0, n_evwin),
                maxshape=(None, None, n_evwin))
        h5.create_dataset('MotifSearch/exprs_coef_2',
                dtype="f8",
                shape=(0, 0, n_evwin),
                maxshape=(None, None, n_evwin))

        #
        h5.create_dataset('Diagnostic/RMSE',
                dtype="f8",
                shape=(0, len(self.params['seqnmf_params']['MotifSearch']), self.params['seqnmf_params']['MotifSearch'][0]['model']['n_motif'], 1),
                maxshape=(None, len(self.params['seqnmf_params']['MotifSearch']), self.params['seqnmf_params']['MotifSearch'][0]['model']['n_motif'], 1))

        h5.swmr_mode = True
        self.h5 = h5
        self.seqnmf_dict = seqnmf_dict
        self.events_dict = {'current': None, 'average': None, 'raster': None}
        self.model_init = True

        self.cache_model()

    def measure_linelength(self, signal):
        if self.burn_in: 
            out_ll = self.LL_Standardizer.operations['line_length'].evaluate(data=signal)
            init_mean, init_var = (out_ll[:].mean(axis=0), out_ll[:].var(axis=0))
            self.LL_Standardizer.operations['shift_scale'].previous_mean = init_mean
            self.LL_Standardizer.operations['shift_scale'].previous_variance = init_var
            self.burn_in = False
            return None
        else:
            LL_ZV_BOUNDED = self.LL_Standardizer.evaluate(data=signal)
            return LL_ZV_BOUNDED

    def transfer_to_ensemble(self):
        ensemble_id = 0
        for model in self.seqnmf_dict['seqnmf_model']:
            self.seqnmf_dict['ensemble_model'][-1].cnmf.W.data[:, ensemble_id:ensemble_id+model.rank, :] = \
                    model.cnmf.W.data[:, :, :]
            self.seqnmf_dict['ensemble_model'][-1].cnmf.H.data[:, ensemble_id:ensemble_id+model.rank, :] = \
                    model.cnmf.H.data[:, :, :]
            self.seqnmf_dict['ensemble_model'][-1].H0[:, ensemble_id:ensemble_id+model.rank, :] = \
                    model.H0[:, :, :]
            self.seqnmf_dict['ensemble_model'][-1].H1[:, ensemble_id:ensemble_id+model.rank, :] = \
                    model.H1[:, :, :]
            ensemble_id += model.rank

    def transfer_from_ensemble(self):
        ensemble_id = 0
        for model in self.seqnmf_dict['seqnmf_model']:
            model.cnmf.W.data[:, :, :] = \
                    self.seqnmf_dict['ensemble_model'][-1].cnmf.W.data[:, ensemble_id:ensemble_id+model.rank, :]
            model.cnmf.H.data[:, :, :] = \
                    self.seqnmf_dict['ensemble_model'][-1].cnmf.H.data[:, ensemble_id:ensemble_id+model.rank, :]
            model.H0[:, :, :] = \
                    self.seqnmf_dict['ensemble_model'][-1].H0[:, ensemble_id:ensemble_id+model.rank, :]
            model.H1[:, :, :] = \
                    self.seqnmf_dict['ensemble_model'][-1].H1[:, ensemble_id:ensemble_id+model.rank, :]
            ensemble_id += model.rank

    def update_ensemble_tags(self, X):
        # Update cross-factor overlap matrix (ortho-x)
        X = torch.from_numpy(X.T).unsqueeze(0).float()
        self.seqnmf_dict['ensemble_model'][0].orthoX_overlap(X)
        oX = self.seqnmf_dict['ensemble_model'][0].oX.clip(min=0, max=1)
        oX_D = (1-oX)

        # Cluster motifs based on overlapping representation
        sscore = -np.inf
        for nc in range(2, self.seqnmf_dict['ensemble_model'][0].rank):
            km = SpectralClustering(n_clusters=nc, affinity='precomputed')
            clust = km.fit_predict(oX)
            sscore_new = silhouette_score(oX_D, clust, metric='precomputed')
            if sscore_new > sscore:
                sscore = sscore_new
                clusters = clust.copy()
        self.seqnmf_dict['ensemble_model'][0].ensemble_tags = clusters

    def update_model(self, signal, pool=None):
        # Compute line-length
        LL = self.measure_linelength(signal['data'])
        if LL is None:
            return True

        if np.isfinite(LL).all():
            if (LL > (2*eps)).any():

                # SeqNMF Update Models
                (self.seqnmf_dict['seqnmf_model'],
                 self.seqnmf_dict['seqnmf_trainer']) = \
                     parallel_model_update_HW(
                             self.seqnmf_dict['seqnmf_trainer'],
                             LL,
                             n_iter_H=self.params['seqnmf_params']['trainer']['event_iter'],
                             n_iter_W=self.params['seqnmf_params']['trainer']['motif_iter'],
                             pool=pool)

                self.transfer_to_ensemble()

                #self.update_ensemble_tags(LL)

                """
                event_times = self.seqnmf_dict['ensemble_model'][0].get_event_times()
                self.update_event_raster(event_times, signal['timestamp vector'][0])
                self.clip_and_update_events(
                        signal['data'], event_times,
                        self.seqnmf_dict['ensemble_model'][0].W.shape[-1]*2)
                """

        # Update h5
        expand_h5(self.h5, skip_key=[], axis=0)
        self.h5['timestamp'][-1, 0] = signal['timestamp vector'][0]
        self.h5['timestamp'][-1, 1] = signal['timestamp vector'][-1]
        self.h5['LineLength/channel_SIG'][-1, :, :] = LL
        self.update_h5_motifsearch()
        self.cache_model()
        return True

    def update_event_raster(self, event_times, start_ts):
        if self.events_dict['raster'] is None:
            self.events_dict['raster'] = [np.empty((0,2))]*len(event_times)

        for m_ii, m_ev in enumerate(event_times):
            m_ev2 = np.array(m_ev).T
            m_ev2 = m_ev2[np.argsort(m_ev2[:,0])]
            m_ev2[:,0] /= self.params['ecog_params']['sampling_frequency']
            m_ev2[:,0] += start_ts
            self.events_dict['raster'][m_ii] = np.append(
                    self.events_dict['raster'][m_ii], m_ev2, axis=0)

    def clip_and_update_events(self, X, event_times, event_win):
        clip_len = int(event_win*2)
        if self.events_dict['average'] is None:
            self.events_dict['average'] = [{
                'N': 0,
                'mean': np.zeros((clip_len, X.shape[1])),
                'M2': np.zeros((clip_len, X.shape[1]))}
                for r in range(len(event_times))]

        motifs = []
        for r, event in enumerate(event_times):
            event_ix = event[0]
            event_val = event[1]
            clips = []
            for ev_ix, ev_val in zip(event_ix, event_val):
                clip = slice(max(ev_ix - event_win, 0), 
                             min(ev_ix + event_win, X.shape[0]))
                Xc = (X[clip] - X[clip].mean(axis=0)) / X[clip].std(axis=0)
                if (ev_ix-event_win) < 0:
                    Xc = np.concatenate((
                        np.zeros((clip_len-Xc.shape[0], Xc.shape[1])), Xc),
                        axis=0)
                if (ev_ix+event_win) > X.shape[0]:
                    Xc = np.concatenate((
                        Xc, np.zeros((clip_len-Xc.shape[0], Xc.shape[1]))),
                        axis=0)
                Xc *= ev_val
                Xc = np.nan_to_num(Xc)

                self.events_dict['average'][r]['N'] += 1
                dX = Xc - self.events_dict['average'][r]['mean']
                self.events_dict['average'][r]['mean'] += (
                        dX / self.events_dict['average'][r]['N'])
                dX2 = Xc - self.events_dict['average'][r]['mean']
                self.events_dict['average'][r]['M2'] += (dX * dX2)

                clips.append(Xc)
            motifs.append(clips)
        self.events_dict['current'] = motifs
