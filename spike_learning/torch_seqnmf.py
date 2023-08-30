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
from oasis.functions import GetSn, estimate_time_constant, deconvolve
import scipy.stats as sp_stats
from scipy.optimize import linear_sum_assignment

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


def oasis_tau_to_g(t_d, t_r, fs):
    if (t_d is None) and (t_r is None):
        return (t_d, t_r,)
    if (t_d is None) or (t_r is None):
        raise Exception('Must either specify both as None or neither as None')
    if (t_r >= t_d):
        raise Exception('Rise time constant must be faster/smaller than decay time constant')
    if (t_d == 0):
        raise Exception('Decay time constant must be greater than zero or set to None')

    g_t_d = np.exp(-1/(t_d*fs))
    g_t_r = np.exp(-1/(t_r*fs)) if t_r > 0 else 0
    g1 =  g_t_d + g_t_r
    g2 = -g_t_d * g_t_r

    if abs(g2) <= eps:
        return (g1,)
    else:
        return (g1, g2,)


def oasis_g_to_tau(g1, g2, fs):
    tau_r = (np.log(-1/(2*g2) * (g1 + np.sqrt(g1**2 + 4*g2)))*fs)**(-1)
    tau_d = (np.log(-1/(2*g2) * (g1 - np.sqrt(g1**2 + 4*g2)))*fs)**(-1)
    return tau_r, tau_d


def motif_compare_spatiotemp(W0, W1):
    W1 = W1[:, ::-1]

    W_conv = sp_sig.fftconvolve(
            W0, W1, mode='same', axes=1) / W0.shape[1]
    ch_energy = W_conv.max(axis=1)
    ch_shift = W_conv.argmax(axis=1)
    wt_mean_shift = int(round((ch_energy/ch_energy.sum()*ch_shift).sum()))

    motif_shift = (int(round(wt_mean_shift)) - W_conv.shape[1] // 2)

    W1 = np.roll(W1, -motif_shift, axis=1)[:, ::-1]

    W_R2 = (1 - 
            (((W0 / np.linalg.norm(W0)) - 
              (W1 / np.linalg.norm(W1)))**2).sum())

    return W_R2


def motif_compare_spatial(W0, W1):

    WW_R2 = (1 - 
            (((W0.mean(axis=1) / np.linalg.norm(W0.mean(axis=1))) - 
              (W1.mean(axis=1) / np.linalg.norm(W1.mean(axis=1))))**2).sum())

    return WW_R2


def motif_compare_temporal(H0, H1):
    H1 = H1[::-1]
    H_conv = sp_sig.fftconvolve(H0, H1, mode='same')
    motif_shift = int(H_conv.argmax() - len(H_conv) // 2)
    H1 = np.roll(H1, -motif_shift)[::-1]

    H_R2 = (1 - 
            (((H0 / np.linalg.norm(H0)) - 
              (H1 / np.linalg.norm(H1)))**2).sum())

    return H_R2


def motif_fit(W, H, X):
    tot_pow = (X.detach().numpy()**2).sum()
    Xh = F.conv1d(H, W,
                  padding=W.shape[2]-1).detach().numpy()
    Xh_rss = ((X.detach().numpy()-Xh)**2).sum()
    R2 = np.nan_to_num((tot_pow - Xh_rss) / tot_pow)
    return R2


def motif_score(W, H, X, w_perm_ix, h_perm_ix):
    R2_full = min(max(motif_fit(W, H, X), 0), 1)

    H_perm = torch.Tensor(np.roll(H[:, [0], :].detach().numpy(),
        h_perm_ix, axis=-1))

    W_perm = np.array([np.roll(W[ch, [0], :].detach().numpy(), w_perm_ix[ch], axis=-1)
        for ch in range(W.shape[0])])
    W_perm = torch.Tensor(W_perm)
    midpt = int(W.shape[-1] // 2)
    cofm = W_perm[:, 0, :].numpy().mean(axis=0)
    cofm = ((cofm / cofm.sum()) * np.arange(len(cofm))).sum()
    shift = 0 if np.isnan(midpt-cofm) else int(midpt-cofm)
    W_perm[:, 0, :] = torch.roll(
            W_perm[:, 0, :], shift, dims=1)

    R2_event_precision = min(max(motif_fit(W, H_perm, X), 0), 1)
    R2_motif_precision = min(max(motif_fit(W_perm, H, X), 0), 1)

    R2_event_precision = max(min((R2_full - R2_event_precision) / R2_full, 0), 1)
    R2_motif_precision = max(min((R2_full - R2_motif_precision) / R2_full, 0), 1)

    return R2_full, R2_event_precision, R2_motif_precision


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


def parallel_model_update_H(individual_trainers, signal,
        n_iter_H, pool=None):

    if pool is None:
        for trainer in individual_trainers:
            trainer.model_online_filter(signal, n_iter_H)
    else:
        fn_dict = [
                {'fn': trainer.model_online_filter,
                 'args': {
                     'signal': signal,
                     'n_iter_H': n_iter_H,
                     }
                 }
                for trainer in individual_trainers]
        individual_trainers = pool.map(pool_wrap, fn_dict)
    individual_models = [mtrain.seqnmf_model
            for mtrain in individual_trainers]
    return individual_models, individual_trainers


def parallel_model_update_W(individual_trainers, signal,
        n_iter_W, pool=None):

    if pool is None:
        for trainer in individual_trainers:
            trainer.model_online_update(signal, n_iter_W)
    else:
        fn_dict = [
                {'fn': trainer.model_online_update,
                 'args': {
                     'signal': signal,
                     'n_iter_W': n_iter_W,
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
            input_rescale,
            feat_normalization,
            feat_recentering,
            motif_noise_additive,
            motif_noise_jitter,
            oasis_g1g2,
            oasis_g_optimize,
            penalties=[{}]):

        super().__init__()
        self.n_chan = n_chan
        self.n_sample = n_sample
        self.n_convwin = n_convwin
        self.rank = rank
        self.input_rescale = input_rescale
        self.feat_normalization = feat_normalization
        self.feat_recentering = feat_recentering
        self.penalties = penalties
        self.motif_noise_additive = np.array(motif_noise_additive)
        self.motif_noise_jitter = np.array(motif_noise_jitter)
        self.oasis_g1g2 = np.array(rank * [oasis_g1g2])
        self.oasis_g1g2_opt = oasis_g_optimize

        self.one_minus_eye = torch.Tensor(1 - np.eye(self.rank))
        self.ortho = torch.Tensor(np.ones(2*n_convwin-1).reshape(1, 1, 1, -1))

        self.Hsn = np.zeros(rank)
        self.Hb = np.zeros(rank)

        with torch.no_grad():
            self.cnmf = NMFD((1, n_chan, n_sample), rank=rank, T=n_convwin)
            self.reinit_coefs()
            self.reinit_feats()
            self.renorm_feats()
            self.recenter_model()
            self.update_hash()
            self.update_feat_cache()

    def forward(self):
        WxH = self.cnmf()
        return WxH

    def loss(self, X, beta, skip_penalty=False):

        X = X.copy()
        if self.input_rescale is not None:
            X = (2 / np.pi) * np.arctan(
                    ((X + self.input_rescale[0]) /
                        self.input_rescale[1])**(self.input_rescale[2]))
        X = (X / X.sum(axis=0)) + eps
        X = torch.from_numpy(X.T).unsqueeze(0).float()

        io_dict = {}
        for pn, p in self.cnmf.named_parameters():
            if id(p) not in io_dict:
                io_dict[id(p)] = list()
            penalty = self.penalty(pn, X)
            if skip_penalty:
                penalty[...] = 0
            io_dict[id(p)].append((X, self(), beta, penalty))

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
            Wf = self.W.detach().sum(axis=-1).mm(self.one_minus_eye)
            pen += (self.penalties['orthoW'] *
                    Wf.unsqueeze(2).repeat(1, 1, self.W.shape[-1]))

        if ('orthoX' in self.penalties) and par_name == 'W':
            HS = F.conv2d(
                    self.H.unsqueeze(0),
                    self.ortho,
                    padding='same')[0, 0]
            HS = HS.T

            HS = HS.mm(self.one_minus_eye)
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

    def noise_injection(self):

        with torch.no_grad():
            if (self.motif_noise_additive > 0).any():
                for r in range(self.rank):
                    noise_add = torch.rand(self.cnmf.W.shape).abs()
                    motif_noise_additive = (noise_add[:, r, :] /
                            noise_add[:, r, :].sum())
                    motif_orig = (self.cnmf.W[:, r, :] /
                            self.cnmf.W[:, r, :].sum())

                    self.cnmf.W[:, r, :] = (
                        (1 - self.motif_noise_additive[r]) * motif_orig +
                        (self.motif_noise_additive[r]) * motif_noise_additive)

            if (self.motif_noise_jitter > 0).any():
                jit_win = self.n_convwin // 2
                for r in range(self.rank):
                    noise_jit = (np.random.uniform(-0.5, 0.5, self.n_chan) *
                            self.motif_noise_jitter[r] * jit_win).astype(int)
                    for ch in range(self.cnmf.W.shape[0]):
                        self.cnmf.W.data[ch,r,:] = torch.roll(
                                self.cnmf.W[ch,r,:], 
                                noise_jit[ch])

            self.renorm_feats()

    def reinit_coefs(self):
        self.cnmf.H[...] = torch.rand(self.cnmf.H.shape).abs()
        self.H = self.cnmf.H
        self.Hraw = torch.Tensor(np.zeros(self.cnmf.H.shape))
        self.Hspk = torch.Tensor(np.zeros(self.cnmf.H.shape))
        self.R2 = np.nan*np.zeros(self.rank)
        self.R2_event = np.nan*np.zeros(self.rank)
        self.R2_seq = np.nan*np.zeros(self.rank)

    def reinit_feats(self):
        self.cnmf.W[...] = torch.rand(self.cnmf.W.shape).abs()
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
            self.hashes[r] = dict_hash(as_dict)

    def trim_coefs(self):
        half_cw = int(self.n_convwin // 2)
        hann_win = sp_sig.hann(self.n_convwin)[:half_cw]
        for r in range(self.H.shape[1]):
            H = self.H[0, r, :].detach().numpy().copy()
            H[:half_cw] = H[:half_cw] * hann_win
            H[-half_cw:] = H[-half_cw:] * hann_win[::-1]
            self.H[0, r, :] = torch.Tensor(H)

    def deconv_coefs(self):
        for r in range(self.H.shape[1]):
            H = self.H[0, r, :].detach().numpy().copy()
            self.Hraw[0, r, :] = torch.Tensor(H.copy())
            try:
                # Update spike deconvolution model
                c, s, b, g, lam = deconvolve(
                        H, g=self.oasis_g1g2[r], penalty=0,
                        b=None, b_nonneg=True,
                        optimize_g=self.H.shape[-1] if self.oasis_g1g2_opt else 0)

                self.oasis_g1g2[r] = g
                self.Hsn[r] = lam
                self.Hb[r] = b
            except Exception as E:
                print(E)
                c = np.zeros(len(self.H[0, r, :]))
                s = np.zeros(len(c))

            self.H[0, r, :] = torch.Tensor(c.clip(min=0) + eps)
            self.Hspk[0, r, :] = torch.Tensor(s.clip(min=0))

    def renorm_feats(self):
        for r in range(self.cnmf.rank):
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

    def update_feat_cache(self):
        W_R2_delta = []
        for r in range(self.cnmf.rank):
            W_R2 = motif_compare_spatiotemp(
                    self.cnmf.W.detach().numpy()[:, r, :],
                    self.W_o[:, r, :])

            W_R2_delta.append(W_R2)

        self.W_R2_delta = np.array(W_R2_delta)
        self.W_o[...] = self.cnmf.W.detach().numpy().copy()

    def update_motif_scores(self, X, W_perm_ix, H_perm_ix):
        if not hasattr(self, 'mscores_welford'):
            self.mscores_welford = {
                    'R2': {
                        'N': np.zeros(self.rank),
                        'mean': np.zeros(self.rank),
                        'M2': np.zeros(self.rank)},
                    'event_precision': {
                        'N': np.zeros(self.rank),
                        'mean': np.zeros(self.rank),
                        'M2': np.zeros(self.rank)},
                    'motif_precision': {
                        'N': np.zeros(self.rank),
                        'mean': np.zeros(self.rank),
                        'M2': np.zeros(self.rank)}
                    }

        self.R2 = np.nan*np.zeros(self.rank)
        self.pscore = np.nan*np.zeros(self.rank)

        for r in range(self.rank):
            (self.R2[r], self.pscore[r]) = motif_precision_score(
                    self.W[:, [r], :],
                    self.H[:, [r], :],
                    X, W_perm_ix, H_perm_ix)
        self.R2 = self.R2.clip(min=0, max=1)
        self.pscore = self.pscore.clip(min=0)

    def orthoX_overlap(self, X):
        WxX = torch.conv_transpose1d(X, self.W,
                padding=self.W.shape[-1]-1).detach().numpy()[0].clip(min=0)
        H = self.H.detach().numpy()[0].clip(min=0)
        oX = np.sqrt((WxX)*H)

        if not hasattr(self, 'oX_welford'):
            self.oX_welford = {
                    'N': 0,
                    'mean': np.zeros(self.rank),
                    'M2': np.zeros((self.rank, self.rank))}

        for s in range(oX.shape[1]):
            self.oX_welford['N'] += 1
            dX = oX[:, s] - self.oX_welford['mean']
            self.oX_welford['mean'] += (dX / self.oX_welford['N'])
            dX2 = oX[:, s] - self.oX_welford['mean']
            self.oX_welford['M2'] += np.outer(dX, dX2)

        cov = self.oX_welford['M2'] / self.oX_welford['N']
        sxy = np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
        cxy = cov / sxy
        self.oX = cxy

    def get_event_times(self):
        cwin_half = int(self.n_convwin // 2)

        event_raster = []
        for r in range(self.rank):
            s = self.Hspk[0,r,:].detach().numpy()
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


class SeqNMFTrainer():
    def __init__(self,
            seqnmf_model,
            max_motif_lr,
            max_event_lr,
            beta):

        self.seqnmf_model = seqnmf_model
        self.max_motif_lr = max_motif_lr*np.ones(seqnmf_model.rank)
        self.max_event_lr = max_event_lr*np.ones(seqnmf_model.rank)

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
            else:
                self.seqnmf_model.H.data[...] = self.seqnmf_model.Hraw.data[...]

        for i in range(n_iter):
            def closure():
                self.event_trainer.zero_grad()
                return self.seqnmf_model.loss(
                        signal,
                        self.beta,
                        skip_penalty)
            self.event_trainer.step(closure) 

        with torch.no_grad():
            self.seqnmf_model.trim_coefs()
            self.seqnmf_model.deconv_coefs()

    def model_update_W(self, signal, reinit=True, n_iter=1, verbose=True, skip_penalty=False):

        if reinit:
            with torch.no_grad():
                self.seqnmf_model.reinit_feats()

        for i in range(n_iter):
            def closure():
                self.motif_trainer.zero_grad()
                return self.seqnmf_model.loss(
                        signal,
                        self.beta,
                        skip_penalty)
            self.motif_trainer.step(closure)

        with torch.no_grad():
            self.seqnmf_model.recenter_model()
            self.seqnmf_model.renorm_feats()

    def model_online_filter(self, signal, n_iter_H):
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'

        self.model_update_H(signal, reinit=True, n_iter=n_iter_H)
        return self

    def model_online_update(self, signal, n_iter_W):
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'

        self.seqnmf_model.noise_injection()
        self.model_update_W(signal, reinit=False, n_iter=n_iter_W)
        self.seqnmf_model.update_feat_cache()
        #self.adapt_motif_lr()
        return self

    def model_online_update_and_filter(self, signal, n_iter_H, n_iter_W):
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'

        self.seqnmf_model.noise_injection()
        self.model_update_H(signal, reinit=True, n_iter=n_iter_H)
        self.model_update_W(signal, reinit=False, n_iter=n_iter_W)
        self.model_update_H(signal, reinit=False, n_iter=1)
        self.seqnmf_model.update_feat_cache()
        #self.adapt_motif_lr()
        return self
