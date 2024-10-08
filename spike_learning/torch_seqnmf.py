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
from precog.operations.shiftrescalers import RunningShiftScaler
from time import time_ns

from .utils import dict_hash

torch.set_flush_denormal(True)
eps = 1e-16


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


def parallel_model_update_HW(individual_trainers, signal,
        n_iter_H, n_iter_W, pool=None):

    if pool is None:
        for trainer in individual_trainers:
            trainer.model_online_update_and_filter(signal, n_iter_H, n_iter_W)
    else:
        fn_dict = [
                {'fn': trainer.model_online_update_and_filter,
                 'args': {
                     'signal': signal,
                     'n_iter_H': n_iter_H,
                     'n_iter_W': n_iter_W
                     }
                 }
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
            feat_normalization,
            coef_normalization,
            feat_recentering,
            motif_competitions,
            penalties=[{}]):

        super().__init__()
        self.n_chan = n_chan
        self.n_sample = n_sample
        self.n_convwin = n_convwin
        self.rank = rank
        self.coef_normalization = coef_normalization
        self.feat_normalization = feat_normalization
        self.feat_recentering = feat_recentering
        self.penalties = penalties

        self.one_minus_eye = torch.Tensor(motif_competitions)
        self.ortho = torch.Tensor(np.ones(2*n_convwin-1).reshape(1, 1, 1, -1))

        self.runshiftscale = RunningShiftScaler(
                forget_factor=1-np.exp(-(1/1024.0)/60),
                shift_rescale='shift_rescale_zscore')
        self.runshiftscale_burn = True

        self.motif_groups = [np.array([0]),
                             np.array([1,2,3,4,5]),
                             np.array([6,7,8,9,10])]

        with torch.no_grad():
            self.cnmf = NMFD((1, n_chan, n_sample), rank=rank, T=n_convwin)
            self.reinit_coefs()
            self.reinit_feats()
            self.renorm_feats()
            self.recenter_model()
            self.update_hash()

    def forward(self):
        WxH = self.cnmf()
        return WxH

    def loss(self, X, beta, skip_penalty=False):

        X = X.copy()
        X = torch.from_numpy(X.T).unsqueeze(0).float()

        io_dict = {}
        for pn, p in self.cnmf.named_parameters():
            if id(p) not in io_dict:
                io_dict[id(p)] = list()
            penalty = self.penalty(pn, X)
            if skip_penalty:
                penalty[...] = 0
            io_dict[id(p)].append((X, self(), beta, penalty, torch.ones_like(X)))

        return io_dict

    def penalty(self, par_name, X):

        if par_name == 'W':
            pen = torch.zeros_like(self.W)
        else:
            pen = torch.zeros_like(self.H)

        if ('l1W' in self.penalties) and par_name == 'W':
            pen += self.penalties['l1W']

        if ('l1H' in self.penalties) and par_name == 'H':
            pen += self.penalties['l1H']

        if ('orthoH' in self.penalties) and par_name == 'H':
            HS = F.conv2d(
                    self.H.unsqueeze(0),
                    self.ortho,
                    padding='same')[0, 0]
            pen += (self.penalties['orthoH'] * 
                    self.one_minus_eye.mm(HS).unsqueeze(0))

        if ('orthoW' in self.penalties) and par_name == 'W':
            Wf = self.W.detach().sum(axis=-1).mm(self.one_minus_eye.T)
            pen += (self.penalties['orthoW'] *
                    Wf.unsqueeze(2).repeat(1, 1, self.W.shape[-1]))

        if ('orthoX' in self.penalties) and par_name == 'W':
            HS = F.conv2d(
                    self.H.unsqueeze(0),
                    self.ortho,
                    padding='same')[0, 0]
            HS = HS.T

            HS = HS.mm(self.one_minus_eye.T)
            for i in range(self.n_convwin):
                i_end = self.n_sample-self.n_convwin+1+i
                XSH = X.detach()[0, :, i:i_end].mm(HS)
                pen[:, :, (self.n_convwin-1)-i] += self.penalties['orthoX']*XSH

        if ('orthoX' in self.penalties) and par_name == 'H':
            WxX = torch.conv_transpose1d(X, self.W, padding=self.W.shape[-1]-1)
            WxXS = F.conv2d(
                    WxX.detach().unsqueeze(0),
                    self.ortho,
                    padding='same')[0, 0]
            pen += (self.penalties['orthoX'] * 
                    self.one_minus_eye.mm(WxXS).unsqueeze(0))

        return pen

    def reinit_coefs(self):
        self.cnmf.H[...] = torch.rand(self.cnmf.H.shape).abs()
        self.H = self.cnmf.H
        self.H0 = self.H.detach().numpy().copy()
        self.H1 = self.H.detach().numpy().copy()
        self.R2 = np.nan*np.zeros(self.rank)
        self.R2_event = np.nan*np.zeros(self.rank)
        self.R2_seq = np.nan*np.zeros(self.rank)

    def reinit_feats(self):
        self.cnmf.W[...] = torch.rand(self.cnmf.W.shape).abs()
        self.cnmf.W[:,self.motif_groups[0],:] = 1
        self.W = self.cnmf.W
        self.W_o = self.cnmf.W.detach().numpy().copy()
        self.W_R2_delta = np.zeros(self.rank)

    def update_hash(self, motif_ids=None):
        if not hasattr(self, 'hashes'):
            self.hashes = ['']*self.rank

        if motif_ids is None:
            motif_ids = range(self.rank)

        for r in motif_ids:
            as_dict = dict(enumerate(
                self.cnmf.W[:,r,:].detach().numpy().astype(float).flatten()))
            self.hashes[r] = str(time_ns()) #dict_hash(as_dict)

    def trim_coefs(self):
        half_cw = int(self.n_convwin // 2)
        hann_win = sp_sig.hann(self.n_convwin)[:half_cw]
        for r in range(self.H.shape[1]):
            H = self.H[0, r, :].detach().numpy().copy()
            H[:half_cw] = H[:half_cw] * hann_win
            H[-half_cw:] = H[-half_cw:] * hann_win[::-1]
            self.H[0, r, :] = torch.Tensor(H)

    def renorm_coefs(self):

        # Step 0. Motif expression and total motif power per time sample
        H = self.H.detach().numpy().copy()[0]
        Hpow = H.sum(axis=0)

        # Step 1. Keep "ones motif" the same. No refinement.

        # Step 2. Apply tanh refinement to the "background motifs", redistribute weights.
        if self.coef_normalization['step2'] == 'learn_background':
            if self.runshiftscale_burn:
                init_mean, init_var = (H[self.motif_groups[1]].mean(axis=-1), H[self.motif_groups[1]].var(axis=-1))
                self.runshiftscale.previous_mean = init_mean
                self.runshiftscale.previous_variance = init_var
                self.runshiftscale_burn = False
            else:
                if self.coef_normalization['step2a'] is None:
                    stdv_fac = 1
                else:
                    stdv_fac = float(self.coef_normalization['step2a'])

                tan_thresh = (self.runshiftscale.previous_mean +
                        np.sqrt(self.runshiftscale.previous_variance)*stdv_fac)
                self.runshiftscale.evaluate(data=H[self.motif_groups[1]].T)
                H_squashed = (tan_thresh * np.tanh(H[self.motif_groups[1]].T / tan_thresh)).T
                H_reduced = (H[self.motif_groups[1]] - H_squashed).sum(axis=0)
                H[self.motif_groups[1]] = H_squashed

                if self.coef_normalization['step2b'] == 'excess_to_max_sparse':
                    # implement vectorized version of this, no need to for-loop. 
                    for ix in range(H_reduced.shape[-1]):
                        max_motif = self.motif_groups[2][np.argmax(H[self.motif_groups[2], ix])]
                        H[max_motif, ix] += H_reduced[ix]

        # Step 3. Apply sparse/threshold refinement to the "sparse motifs", redistribute weights.
        # implement a sparse version of Step 3 that does not perform cross-motif thresholding (treat each motif independently)
        if self.coef_normalization['step3'] == 'nongreedy_sparse':
            HH = H.copy()
            if self.coef_normalization['step3a'] is None:
                H[self.motif_groups[2]] = 0
            else:
                H[self.motif_groups[2]] *= self.coef_normalization['step3a']

            if self.coef_normalization['step3b'] is None:
                power_thresh = 0.5
            else:
                power_thresh = float(self.coef_normalization['step3b'])

            for r in self.motif_groups[2]:
                local_valley_ix = np.flatnonzero(~((HH[r][1:-1] > HH[r][:-2]) &
                                                   (HH[r][1:-1] > HH[r][2:]))) + 1
                HH[r][local_valley_ix] = 0
                while HH[r].sum() > 0:
                    col_idx = np.flatnonzero(HH[r] == HH[r].max())
                    if len(col_idx) == 0:
                        break
                    col_slice = slice(
                            max(int(col_idx[0] - self.n_convwin), 0),
                            min(int(col_idx[0] + self.n_convwin), HH.shape[1]))

                    if (HH[r, col_idx[0]] / (Hpow[col_idx[0]] - HH[0, col_idx[0]])) > power_thresh:
                        H[r, col_idx[0]] = HH[r, col_idx[0]]
                    HH[r, col_slice] = 0

        # Step 4. Garbage Collect
        if self.coef_normalization['step4'] == 'excess_to_ones':
            H_resid_pow = Hpow - H.sum(axis=0)
            H[self.motif_groups[0]] += H_resid_pow

        # Step 5. Update tensor
        self.H[0,:,:] = torch.Tensor(H + eps)

    def renorm_feats(self):
        self.cnmf.W[:,0,:] = 1
        for r in range(self.cnmf.rank):
            if 'log' in self.feat_normalization:
                self.cnmf.W[:,r,:] = np.log(self.cnmf.W[:,r,:] + 1)

            if self.feat_normalization == 'l1':
                self.cnmf.W[:,r,:] /= self.cnmf.W[:,r,:].sum()
            elif self.feat_normalization == 'l2':
                self.cnmf.W[:,r,:] /= np.sqrt((self.cnmf.W[:,r,:]**2).sum())
            elif self.feat_normalization == 'max':
                self.cnmf.W[:,r,:] /= self.cnmf.W[:,r,:].max()
            else:
                self.cnmf.W[:,r,:] = self.cnmf.W[:,r,:]

        torch.nan_to_num_(self.cnmf.W)
        self.cnmf.W[...] += eps

    def recenter_model(self):
        midpt = int(self.W.shape[-1] // 2)

        for r in range(self.cnmf.rank):
            if self.feat_recentering == 'cofm':
                cofm = self.cnmf.W[:, r, :].numpy().mean(axis=0)
                cofm = ((cofm / cofm.sum()) * np.arange(len(cofm))).sum()
                shift = 0 if np.isnan(midpt-cofm) else int(midpt-cofm)
            elif self.feat_recentering == 'max':
                cofm = self.cnmf.W[:, r, :].numpy().max(axis=0).argmax()
                shift = 0 if np.isnan(midpt-cofm) else int(midpt-cofm)
            else:
                shift = 0

            self.cnmf.W[:, r, :] = torch.roll(
                    self.cnmf.W[:, r, :], shift, dims=1)

    def orthoX_overlap(self, X):
        WxX = torch.conv_transpose1d(X, self.W,
                padding=self.W.shape[-1]-1).detach().numpy()[0]
        HS = F.conv2d(
                    self.H.unsqueeze(0),
                    self.ortho,
                    padding='same').detach().numpy()[0,0]
        self.oX = (WxX @ HS.T)

    def get_event_times(self):
        cwin_half = int(self.n_convwin // 2)

        event_raster = []
        for r in range(self.rank):
            s = self.H[0,r,:].detach().numpy()
            s_ind = np.flatnonzero(s > eps)
            s_val = s[s_ind]
            s_ind += cwin_half

            s_ind_new = []
            s_val_new = []
            while len(s_ind) > 0:
                amax = s_ind[np.argmax(s_val)]
                maxv = np.max(s_val)

                if amax in s_ind_new:
                    break

                ix_drop = ((s_ind > (amax - cwin_half)) &
                           (s_ind < (amax + cwin_half)))
                s_ind = np.delete(s_ind, ix_drop)
                s_val = np.delete(s_val, ix_drop)

                s_ind_new.append(amax)
                s_val_new.append(maxv)
            event_raster.append((s_ind_new, s_val_new))
        return event_raster

    def calc_rmse(self, signal):
        WxH = self.forward()
        sdiff1 = (signal - WxH.detach().numpy()[0].T)
        self.RMSE = np.sqrt(np.mean(sdiff1**2))
        self.error_signal = (signal, WxH.detach().numpy()[0].T)

    def calc_marginal_rmse(self, signal):
        self.RMSE_marginal = []
        for r in range(self.rank):
            WxH = F.conv1d(self.H.detach()[:,[r],:],
                           self.W.detach()[:,[r],:], padding=self.W.shape[-1]-1)
            sdiff1 = (signal - WxH.detach().numpy()[0].T)
            self.RMSE_marginal.append(np.sqrt(np.mean(sdiff1**2)))


class SeqNMFTrainer():
    def __init__(self,
            seqnmf_model,
            max_motif_lr,
            max_event_lr,
            max_motif_lr_decay,
            beta):

        self.seqnmf_model = seqnmf_model
        self.max_motif_lr = max_motif_lr*torch.ones_like(seqnmf_model.cnmf.W)
        self.max_event_lr = max_event_lr*torch.ones_like(seqnmf_model.cnmf.H)
        self.max_motif_lr_decay = max_motif_lr_decay*torch.ones_like(seqnmf_model.cnmf.W)

        self.motif_trainer = AdaptiveMu(
                params=[seqnmf_model.cnmf.W],
                theta=[self.max_motif_lr])
        self.event_trainer = AdaptiveMu(
                params=[seqnmf_model.cnmf.H],
                theta=[self.max_event_lr])
        self.beta = beta

    def adapt_motif_lr(self):
        self.motif_trainer.param_groups[0]['theta'][0] = (
                self.max_motif_lr*(1-self.seqnmf_model.W_R2_delta))

    def adapt_event_lr(self):
        self.event_trainer.param_groups[0]['theta'][0] = (
                self.max_event_lr*(1-self.seqnmf_model.W_R2_delta))

    def model_update_H(self, signal, reinit=True, n_iter=1, verbose=True, skip_penalty=False):

        with torch.no_grad():
            if reinit:
                self.seqnmf_model.reinit_coefs()

        for i in range(n_iter):
            def closure():
                self.event_trainer.zero_grad()
                return self.seqnmf_model.loss(
                        signal,
                        self.beta,
                        skip_penalty)
            self.event_trainer.step(closure) 

    def model_update_W(self, signal, reinit=True, n_iter=1, verbose=True, skip_penalty=False):

        with torch.no_grad():
            if reinit:
                self.seqnmf_model.reinit_feats()

        for i in range(n_iter):
            def closure():
                self.motif_trainer.zero_grad()
                return self.seqnmf_model.loss(
                        signal,
                        self.beta,
                        skip_penalty)
            self.motif_trainer.step(closure)

    def model_online_update_and_filter(self, signal, n_iter_H, n_iter_W):
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'

        self.model_update_H(signal, reinit=True, n_iter=n_iter_H)
        with torch.no_grad():
            self.seqnmf_model.trim_coefs()
            self.seqnmf_model.calc_marginal_rmse(signal)
            self.seqnmf_model.H0 = self.seqnmf_model.H.detach().numpy().copy()
            self.seqnmf_model.renorm_coefs()
            self.seqnmf_model.H1 = self.seqnmf_model.H.detach().numpy().copy()

        self.model_update_W(signal, reinit=False, n_iter=n_iter_W)
        with torch.no_grad():
            self.seqnmf_model.recenter_model()
            self.seqnmf_model.renorm_feats()

        self.max_motif_lr *= self.max_motif_lr_decay

        return self
