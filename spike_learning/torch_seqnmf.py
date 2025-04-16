#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" event_learning.py
Description: Learn network representations of spiking events from intracranial EEG.
"""
__author__ = "Ankit N. Khambhati"
__copyright__ = "Copyright 2022, Ankit N. Khambhati"
__credits__ = ["Ankit N. Khambhati"]
__license__ = ""
__version__ = "1.0.0"
__maintainer__ = "Ankit N. Khambhati"
__email__ = ""
__status__ = "Prototype"


import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import scipy.signal as sp_sig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchnmf.nmf import NMFD
from torchnmf.trainer import AdaptiveMu
import scipy.stats as sp_stats
from scipy.optimize import linear_sum_assignment
from time import time_ns

from .utils import dict_hash

torch.set_flush_denormal(True)
eps = 1e-16

import matplotlib.pyplot as plt


def roll_by_gather(mat, dim, shifts: torch.LongTensor):
    # assumes 2D array
    n_rows, n_cols = mat.shape

    if dim==0:
        arange1 = torch.arange(n_rows).view((n_rows, 1)).repeat((1, n_cols))
        arange2 = (arange1 - shifts) % n_rows
        return torch.gather(mat, 0, arange2)
    elif dim==1:
        arange1 = torch.arange(n_cols).view(( 1,n_cols)).repeat((n_rows,1))
        arange2 = (arange1 - shifts) % n_cols
        return torch.gather(mat, 1, arange2)


def pool_wrap(fn_dict):
    return fn_dict['fn'](**fn_dict['args'])


def parallel_model_update_HW(individual_trainers, signal, reinit_H, reinit_W, event_segmented_idx, pool=None):

    if pool is None:
        for trainer in individual_trainers:
            trainer.model_online_update_and_filter(signal, reinit_H, reinit_W, event_segmented_idx)
    else:
        fn_dict = [
                {'fn': trainer.model_online_update_and_filter,
                    'args': {
                        'signal': signal,
                        'reinit_H': reinit_H,
                        'reinit_W': reinit_W,
                        'event_segmented_idx': event_segmented_idx}}
                for trainer in individual_trainers]
        individual_trainers = pool.map(pool_wrap, fn_dict)
    individual_models = [mtrain.seqnmf_model
            for mtrain in individual_trainers]
    return individual_models, individual_trainers


class SeqNMF(nn.Module):
    def __init__(self,
            n_chan,
            n_sample,
            n_convwin,
            rank,
            motif_postprocess_params,
            coef_postprocess_params):

        super().__init__()
        self.n_chan = n_chan
        self.n_sample = n_sample
        self.n_convwin = n_convwin
        self.motif_postprocess_params = motif_postprocess_params
        self.coef_postprocess_params = coef_postprocess_params
        self.rank = 0
        self.cnmf = None
        
        self.add_rank(rank)

    def add_rank(self, n_add):
        new_rank = self.rank + n_add

        with torch.no_grad():
            nmfd = NMFD((1, self.n_chan, self.n_sample),
                    rank=new_rank, T=self.n_convwin)
            nmfd.W[...] = self.init_motifs(motifs=nmfd.W)
            nmfd.H[...] = self.init_coefs(coefs=nmfd.H)

            if self.cnmf is not None:
                nmfd.W[:,:self.rank,:] = self.cnmf.W.detach()[:,:,:]
                nmfd.H[:,:self.rank,:] = self.cnmf.H.detach()[:,:,:]

            self.rank = new_rank
            self.cnmf = nmfd
            self.W = self.cnmf.W
            self.H = self.cnmf.H
            self.postprocess_motifs()

    def forward(self):
        WxH = self.cnmf()
        return WxH

    def loss(self, X, beta):
        X = torch.from_numpy(X.T).unsqueeze(0).float()

        io_dict = {}
        for pn, p in self.cnmf.named_parameters():
            if id(p) not in io_dict:
                io_dict[id(p)] = list()
            penalty = self.penalty(pn, X)
            io_dict[id(p)].append((X, self(), beta, penalty, torch.ones_like(X)))
        return io_dict

    def penalty(self, par_name, X):
        if par_name == 'W':
            pen = torch.zeros_like(self.W)
        else:
            pen = torch.zeros_like(self.H)
        return pen

    def init_coefs(self, coefs=None):
        if coefs is None:
            self.cnmf.H[...] = torch.rand(self.cnmf.H.shape).abs()
        else:
            return torch.rand(coefs.shape).abs()

    def init_motifs(self, motifs=None):
        if motifs is None:
            self.cnmf.W[...] = torch.rand(self.cnmf.W.shape).abs()
        else:
            return torch.rand(motifs.shape).abs()

    def _trim_coefs(self):
        half_cw = int(self.n_convwin // 2)
        hann_win = sp_sig.windows.hann(self.n_convwin)[:half_cw]
        for r in range(self.H.shape[1]):
            H = self.H[0, r, :].detach().numpy().copy()
            H[:half_cw] = H[:half_cw] * hann_win
            H[-half_cw:] = H[-half_cw:] * hann_win[::-1]
            self.H[0, r, :] = torch.Tensor(H)

    def _sparsify_coefs(self, event_segmented_idx=None):

        if event_segmented_idx is None:
            return None

        # Step 1. Find motifs with greatest overlap with the signal 
        H2 = np.zeros_like(self.cnmf.H)
        for ev_idx in event_segmented_idx:
            r = np.argmax([olap[ev_idx] for olap in self.signal_overlap])
            H2[0, r, ev_idx] = self.cnmf.H[0, r, ev_idx]
        
        # Step 2. Update tensor
        self.cnmf.H[0,:,:] = torch.Tensor(H2 + eps)

    def postprocess_coefs(self, event_segmented_idx):
        if self.coef_postprocess_params['trim'] is not None:
            self._trim_coefs()
        if self.coef_postprocess_params['sparsify'] is not None:
            self._sparsify_coefs(event_segmented_idx)

    def _constrain_motifs(self):
        self.cnmf.W[:, :self.motif_postprocess_params['constraints'].shape[1], :] = \
                self.motif_postprocess_params['constraints']

    def _norm_motifs(self):
        for r in range(self.cnmf.rank):
            if self.motif_postprocess_params['normalize'] == 'l1':
                self.cnmf.W[:,r,:] /= self.cnmf.W[:,r,:].sum()
            elif self.motif_postprocess_params['normalize'] == 'l2':
                self.cnmf.W[:,r,:] /= np.sqrt((self.cnmf.W[:,r,:]**2).sum())
            elif self.motif_postprocess_params['normalize'] == 'max':
                self.cnmf.W[:,r,:] /= self.cnmf.W[:,r,:].max()
            else:
                self.cnmf.W[:,r,:] = self.cnmf.W[:,r,:]
        
        torch.nan_to_num_(self.cnmf.W)
        self.cnmf.W[...] += eps

    def _recenter_motifs(self):
        midpt = int(self.W.shape[-1] // 2)

        for r in range(self.cnmf.rank):
            if self.motif_postprocess_params['recenter'] == 'cofm':
                cofm = self.cnmf.W[:, r, :].numpy().mean(axis=0)
                cofm = ((cofm / cofm.sum()) * np.arange(len(cofm))).sum()
                shift = 0 if np.isnan(midpt-cofm) else int(midpt-cofm)
            elif self.motif_postprocess_params['recenter'] == 'max':
                cofm = self.cnmf.W[:, r, :].numpy().max(axis=0).argmax()
                shift = 0 if np.isnan(midpt-cofm) else int(midpt-cofm)
            else:
                shift = 0

            self.cnmf.W[:, r, :] = torch.roll(
                    self.cnmf.W[:, r, :], shift, dims=1)

    def postprocess_motifs(self):
        if self.motif_postprocess_params['constraints'] is not None:
            self._constrain_motifs()
        if self.motif_postprocess_params['normalize'] is not None:
            self._norm_motifs()
        if self.motif_postprocess_params['recenter'] is not None:
            self._recenter_motifs()

    def marginal_recons(self, signal):
        X = torch.from_numpy(signal.T).unsqueeze(0).float()
        self.signal = signal[:,0]
        self.signal_recons = []
        for r in range(self.rank):
            WxH = F.conv1d(self.H.detach()[:,[r],:],
                           self.W.detach()[:,[r],:], padding=self.W.shape[-1]-1)
            self.signal_recons.append(WxH.detach().numpy()[0].T[:,0])
        self.signal_recons = np.array(self.signal_recons)
        self.signal_resid = self.signal - self.signal_recons.sum(axis=0)
        self.signal_overlap = sp_sig.fftconvolve(
                signal.T, sp_stats.zscore(self.W.detach().numpy()[0], axis=1),
                axes=1, mode='valid')

class SeqNMFTrainer():
    def __init__(self,
            seqnmf_model,
            max_motif_lr,
            max_coef_lr,
            motif_iter,
            coef_iter,
            beta):

        self.max_motif_lr = max_motif_lr
        self.max_coef_lr = max_coef_lr
        self.motif_iter = motif_iter
        self.coef_iter = coef_iter
        self.beta = beta

        self.seqnmf_model = None
        self.motif_trainer = None
        self.coef_trainer = None

        self.relink_model(seqnmf_model)

    def relink_model(self, seqnmf_model):
        ####
        motif_trainer = AdaptiveMu(
                params=[seqnmf_model.cnmf.W],
                theta=[self.max_motif_lr*torch.ones_like(seqnmf_model.cnmf.W)]
        )
        if ((self.motif_trainer is not None) &
            (self.seqnmf_model is not None)):
            key = [*self.motif_trainer.state.keys()][0] 
            state_dict = self.motif_trainer.state[key]

            new_state_dict = motif_trainer.state[seqnmf_model.cnmf.W]
            new_state_dict['step'] = state_dict['step']

            new_state_dict['neg'] = torch.zeros_like(seqnmf_model.cnmf.W,
                    memory_format=torch.preserve_format)
            new_state_dict['neg'][:, :state_dict['neg'].shape[1], :] = \
                    state_dict['neg'][...]

            new_state_dict['pos'] = torch.zeros_like(seqnmf_model.cnmf.W,
                    memory_format=torch.preserve_format)
            new_state_dict['pos'][:, :state_dict['pos'].shape[1], :] = \
                    state_dict['pos'][...]

        ####
        coef_trainer = AdaptiveMu(
                params=[seqnmf_model.cnmf.H],
                theta=[self.max_coef_lr*torch.ones_like(seqnmf_model.cnmf.H)]
        )
        if ((self.coef_trainer is not None) &
            (self.seqnmf_model is not None)):
            key = [*self.coef_trainer.state.keys()][0] 
            state_dict = self.coef_trainer.state[key]

            new_state_dict = coef_trainer.state[seqnmf_model.cnmf.W]
            new_state_dict['step'] = state_dict['step']

            new_state_dict['neg'] = torch.zeros_like(seqnmf_model.cnmf.W,
                    memory_format=torch.preserve_format)
            new_state_dict['neg'][:, :state_dict['neg'].shape[1], :] = \
                    state_dict['neg'][...]

            new_state_dict['pos'] = torch.zeros_like(seqnmf_model.cnmf.W,
                    memory_format=torch.preserve_format)
            new_state_dict['pos'][:, :state_dict['pos'].shape[1], :] = \
                    state_dict['pos'][...]

        self.seqnmf_model = seqnmf_model
        self.motif_trainer = motif_trainer
        self.coef_trainer = coef_trainer

    def model_update_H(self, signal, reinit=True, n_iter=1):

        with torch.no_grad():
            if reinit:
                self.seqnmf_model.init_coefs()

        for i in range(n_iter):
            def closure():
                self.coef_trainer.zero_grad()
                return self.seqnmf_model.loss(
                        signal,
                        self.beta)
            self.coef_trainer.step(closure) 

    def model_update_W(self, signal, reinit=True, n_iter=1):

        with torch.no_grad():
            if reinit:
                self.seqnmf_model.init_motifs()

        for i in range(n_iter):
            def closure():
                self.motif_trainer.zero_grad()
                return self.seqnmf_model.loss(
                        signal,
                        self.beta)
            self.motif_trainer.step(closure)

    def model_online_update_and_filter(self, signal, reinit_H, reinit_W, event_segmented_idx):
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        if self.seqnmf_model.rank == 0:
            return self

        # Update Coefficients
        self.model_update_H(signal, reinit=reinit_H, n_iter=self.coef_iter)

        # Sparse Event Segmentation
        with torch.no_grad():
            self.seqnmf_model.marginal_recons(signal)
            self.seqnmf_model.postprocess_coefs(event_segmented_idx)

        # Update Motifs 
        self.model_update_W(signal, reinit=reinit_W, n_iter=self.motif_iter)
        with torch.no_grad():
            self.seqnmf_model.postprocess_motifs()
        
        return self
