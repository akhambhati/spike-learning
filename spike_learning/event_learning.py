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


import copy
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
import tensortools as tt


eps = np.finfo(np.float).resolution
calc_l2err = lambda T, Th: (
        np.linalg.norm(T - Th) /
        np.linalg.norm(T))

calc_cost = lambda T, Th, beta: tt.ncp_nnlds.calc_cost(T, Th, beta)


def nested_folds(n_sample, n_outer_fold, n_inner_fold):

    sample_ix = np.arange(n_sample)

    obs_per_outer_fold = int(np.floor(n_sample  / n_outer_fold))
    outer_sample_len = (obs_per_outer_fold * n_outer_fold)

    obs_per_inner_fold = int(np.floor(obs_per_outer_fold * (n_outer_fold-1) / n_inner_fold))
    inner_sample_len = obs_per_inner_fold * n_inner_fold

    sample_ix = sample_ix[:outer_sample_len]
    outer_bins = sample_ix.reshape(-1, obs_per_outer_fold)

    folds = {}
    for outfold in range(n_outer_fold):
        outfold_name = 'outer_fold-{}'.format(outfold)
        folds[outfold_name] = {}

        test_fold = {outfold}
        train_fold = set([*range(n_outer_fold)]) - test_fold
        folds[outfold_name]['train'] = outer_bins[list(train_fold)].reshape(-1)
        folds[outfold_name]['test'] = outer_bins[list(test_fold)].reshape(-1)

        inner_bins = folds[outfold_name]['train'][:inner_sample_len].reshape(-1, obs_per_inner_fold)

        for infold in range(n_inner_fold):
            infold_name = 'inner_fold-{}'.format(infold)
            folds[outfold_name][infold_name] = {}

            test_fold = {infold}
            train_fold = set([*range(n_inner_fold)]) - test_fold
            folds[outfold_name][infold_name]['train'] = inner_bins[list(train_fold)].reshape(-1)
            folds[outfold_name][infold_name]['test'] = inner_bins[list(test_fold)].reshape(-1)

    return folds


def gen_minibatches(n_sample, mb_size, mb_shift, perm_sample=False):

    sample_ix = np.arange(n_sample)
    if perm_sample:
        sample_ix = np.random.permutation(sample_ix)

    batches = []
    start_ix = 0
    while (start_ix + mb_size) <= n_sample:
        batches.append(sample_ix[start_ix:(start_ix+mb_size)])
        start_ix += mb_shift

        if ((start_ix + mb_size) > n_sample) & (start_ix < n_sample):
            batches.append(sample_ix[start_ix:])

    return np.array(batches)


def minibatch_setup(tensor, exog_input, rank, beta,
        lds_beta, lag_state, lag_exog, anneal_wt, lds_burn,
        mb_size, mb_shift, mb_tol):
    """
    Adjust the negative values in a tensor so they are strictly non-negative.

    Parameters
    ----------

    Returns
    -------
    """

    n_dim = len(tensor.shape)
    n_obs = tensor.shape[0]
    if mb_size is None:
        mb_size = n_obs
        mb_shift = n_obs

    batches = gen_minibatches(n_obs, mb_size, mb_shift, perm_sample=False)

    print('--- Mini-Batch Setup ---')
    print(' :: # of observations - {}'.format(n_obs))
    print(' :: mini-batch size   - {}'.format(mb_size))
    print(' :: # of mini-batches - {}'.format(batches.shape[0]))
    print('========================\n')

    tensor_bdummy = np.zeros_like(tensor)[:mb_size]
    if lag_state > 0:
        LDS_dict = {
            'axis': 0,
            'beta': lds_beta,
            'lag_state': lag_state,
            'lag_exog': lag_exog,
            'anneal_wt': anneal_wt,
            'init': 'rand'}
        exog_bdummy = np.zeros_like(exog_input)[:mb_size]
    else:
        LDS_dict = None
        exog_bdummy = None
    REG_dict = None

    mdl = tt.ncp_nnlds.init_model(
        X=tensor_bdummy,
        rank=rank,
        REG_dict=REG_dict,
        LDS_dict=LDS_dict,
        exog_input=exog_bdummy)

    mdl.model_param['NTF']['beta'] = beta

    mdl.model_param['minibatch'] = {}

    mdl.model_param['minibatch']['tensor'] = tensor
    mdl.model_param['minibatch']['exog_input'] = exog_input

    KTensor = []
    for fac in mdl.model_param['NTF']['W'].factors:
        KTensor.append(fac)
    KTensor[0] = np.random.uniform(size=(n_obs, rank))
    mdl.model_param['minibatch']['WF'] = tt.KTensor(KTensor)

    mdl.model_param['NTF']['W'].factors[0] = mdl.model_param['minibatch']['WF'].factors[0][:mb_size].copy()
    for f_i in range(1, n_dim):
        mdl.model_param['NTF']['W'].factors[f_i] = \
            mdl.model_param['minibatch']['WF'].factors[f_i].copy()

    mdl.model_param['minibatch']['training'] = {
            'beta_cost': [calc_cost(mdl.model_param['minibatch']['tensor'],
                                    mdl.model_param['minibatch']['WF'].full(),
                                    mdl.model_param['NTF']['beta'])],
            'l2_cost': [calc_l2err(mdl.model_param['minibatch']['tensor'],
                                mdl.model_param['minibatch']['WF'].full())],
            'total_epochs': 0,
            'mb_size': mb_size,
            'mb_tol': mb_tol,
            'mbatches': batches,
            'lds_burn': lds_burn if LDS_dict is not None else 0}

    """
    pop_T, ev_T, ch_T = reference_tensor(tensor)
    mdl.model_param['minibatch']['training']['beta_cost_pop'] = calc_cost(
            tensor, pop_T, mdl.model_param['NTF']['beta'])
    mdl.model_param['minibatch']['training']['beta_cost_event_avg'] = calc_cost(
            tensor, ev_T, mdl.model_param['NTF']['beta'])
    mdl.model_param['minibatch']['training']['beta_cost_chan_avg'] = calc_cost(
            tensor, ch_T, mdl.model_param['NTF']['beta'])

    mdl.model_param['minibatch']['training']['l2_cost_pop'] = calc_l2err(
            tensor, pop_T)
    mdl.model_param['minibatch']['training']['l2_cost_event_avg'] = calc_l2err(
            tensor, ev_T)
    mdl.model_param['minibatch']['training']['l2_cost_chan_avg'] = calc_l2err(
            tensor, ch_T)
    """
    return mdl


def minibatch_update(mdl, epochs, mode):
    """
    Adjust the negative values in a tensor so they are strictly non-negative.

    Parameters
    ----------

    Returns
    -------
    """

    n_dim = mdl.model_param['minibatch']['WF'].ndim

    for ep_i in tqdm(range(epochs), position=0, leave=True):
        batches = mdl.model_param['minibatch']['training']['mbatches'][
            np.random.permutation(
                mdl.model_param['minibatch']['training']['mbatches'].shape[0])]

        ###
        if mode == 'train':
            fixed_axes = []
            if (mdl.model_param['minibatch']['training']['total_epochs'] >=
                mdl.model_param['minibatch']['training']['lds_burn'] + 1):
                update_lds_system = True
                update_lds_state = True
            elif (mdl.model_param['minibatch']['training']['total_epochs'] >=
                    mdl.model_param['minibatch']['training']['lds_burn']):
                update_lds_system = True
                update_lds_state = False
            else:
                update_lds_system = False
                update_lds_state = False

        elif mode == 'filter':
            fixed_axes = [1]
            update_lds_system = False
            update_lds_state = True

        elif mode == 'forecast':
            fixed_axes = [1]
            update_lds_system = False
            update_lds_state = True

        ###
        for bbb in tqdm(batches, position=0, leave=True):
            tensor_batch = mdl.model_param['minibatch']['tensor'][bbb]
            if mdl.model_param['LDS'] is not None:
                exog_batch = mdl.model_param['minibatch']['exog_input'][bbb]
            else:
                exog_batch = None

            if len(bbb) != mdl.model_param['NTF']['W'].factors[0].shape[0]:
                KTensor = []
                for fac in mdl.model_param['NTF']['W'].factors:
                    KTensor.append(fac)
                KTensor[0] = mdl.model_param['minibatch']['WF'].factors[0][bbb, :].copy()
                mdl.model_param['NTF']['W'] = tt.KTensor(KTensor)
            else:
                mdl.model_param['NTF']['W'].factors[0] = \
                    mdl.model_param['minibatch']['WF'].factors[0][bbb, :].copy()

            mdl = tt.ncp_nnlds.model_update(
                tensor_batch,
                mdl,
                fixed_axes=fixed_axes,
                exog_input=exog_batch,
                update_lds_state=update_lds_state,
                update_lds_system=update_lds_system,
                fit_dict={
                    'max_iter': 1, 
                    'tol': -np.inf,
                    'verbose': False})
            mdl.model_param['NTF']['W'].rebalance()

            if mode == 'filter':
                mdl.model_param['minibatch']['WF'].factors[0][bbb[mdl.model_param['LDS']['lag_state']:], :] = \
                    mdl.model_param['NTF']['W'].factors[0][mdl.model_param['LDS']['lag_state']:].copy()
            else:
                mdl.model_param['minibatch']['WF'].factors[0][bbb, :] = \
                    mdl.model_param['NTF']['W'].factors[0].copy()

        for f_i in range(1, n_dim):
            mdl.model_param['minibatch']['WF'].factors[f_i] = \
                mdl.model_param['NTF']['W'].factors[f_i].copy()

        mdl.model_param['minibatch']['WF'].rebalance()
        Th = mdl.model_param['minibatch']['WF'].full()

        mdl.model_param['minibatch']['training']['l2_cost'].append(
            calc_l2err(mdl.model_param['minibatch']['tensor'], Th))
        mdl.model_param['minibatch']['training']['beta_cost'].append(
            calc_cost(mdl.model_param['minibatch']['tensor'], Th,
                      mdl.model_param['NTF']['beta']))
        cost = mdl.model_param['minibatch']['training']['beta_cost']

        if not np.isfinite(cost).any():
            break

        last_cost = cost[-1]
        delta_cost = cost[-1]-cost[-2]
        total_cost = cost[-1]-cost[0]

        print('{} : DELTA: {} : TOTAL: {}'.format(
            last_cost, delta_cost, total_cost))

        mdl.model_param['minibatch']['training']['total_epochs'] += 1

        if np.abs(delta_cost) < mdl.model_param['minibatch']['training']['mb_tol']:
            break

    return mdl


def minibatch_forecast(mdl):
    Wn_filt = mdl.model_param['LDS']['AB'].filter_state(
            mdl.model_param['minibatch']['WF'][mdl.model_param['LDS']['axis']],
            mdl.model_param['minibatch']['exog_input'])[0]
    Wn_filt = np.concatenate((np.nan*np.ones((mdl.model_param['LDS']['lag_state'], mdl.model_param['rank'])), Wn_filt), axis=0)

    return Wn_filt


def minibatch_xval(tensor, n_fold, mb_params):
    """
    Adjust the negative values in a tensor so they are strictly non-negative.

    Parameters
    ----------

    Returns
    -------
    """

    n_obs, n_samp, n_chan = tensor.shape

    obs_per_fold = int(np.ceil(n_obs / n_fold))
    pad_obs = (obs_per_fold * n_fold) - n_obs

    """
    obs_shuf_ix = np.concatenate((
        np.random.permutation(n_obs),
        np.random.permutation(n_obs)[:pad_obs]))
    """
    obs_shuf_ix = np.concatenate((
        np.arange(n_obs),
        np.arange(n_obs)[::-1][:pad_obs]))
    n_obs_shuf = len(obs_shuf_ix)

    folds = obs_shuf_ix.reshape(-1, obs_per_fold)

    print('--- Cross-Validation Setup ---')
    print(' :: # of observations - {}'.format(n_obs))
    print(' :: fold size         - {}'.format(obs_per_fold))
    print(' :: # of folds        - {}'.format(n_fold))
    print('========================\n')


    xval_dict = {'fold': [],

                 'train_fold_ix': [],
                 'train_model': [],
                 'train_beta_cost': [],
                 'train_beta_cost_pop': [],
                 'train_beta_cost_event_avg': [],
                 'train_beta_cost_chan_avg': [],
                 'train_l2_cost': [],
                 'train_l2_cost_pop': [],
                 'train_l2_cost_event_avg': [],
                 'train_l2_cost_chan_avg': [],

                 'test_fold_ix': [],
                 'test_model': [],
                 'test_beta_cost': [],
                 'test_beta_cost_pop': [],
                 'test_beta_cost_event_avg': [],
                 'test_beta_cost_chan_avg': [],
                 'test_l2_cost': [],
                 'test_l2_cost_pop': [],
                 'test_l2_cost_event_avg': [],
                 'test_l2_cost_chan_avg': []}
    for key in mb_params:
        xval_dict[key] = []

    for fold in tqdm(range(n_fold), position=0, leave=True):
        test_fold = {fold}
        train_fold = set([*range(n_fold)]) - test_fold

        print('*******TRAINING*******')
        tensor_train = tensor[folds[list(train_fold)].reshape(-1)]
        mdl_train = minibatch_setup(
                tensor=tensor_train,
                rank=mb_params['rank'], 
                beta=mb_params['beta'],
                l1_alpha=mb_params['l1_alpha'],
                lag_order=mb_params['lag_order'],
                mb_size=mb_params['mb_size'],
                mb_epochs=mb_params['mb_epochs'],
                mb_tol=mb_params['mb_tol'],
                mb_iter=1)
        mdl_train = minibatch_train(mdl_train, fixed_axes=[])

        print('*******TESTING*******')
        tensor_test = tensor[folds[list(test_fold)].reshape(-1)]
        mdl_test = minibatch_setup(
                tensor=tensor_test,
                rank=mb_params['rank'], 
                beta=mb_params['beta'],
                l1_alpha=mb_params['l1_alpha'],
                lag_order=mb_params['lag_order'],
                mb_size=tensor_test.shape[0],
                mb_epochs=1,
                mb_tol=1e-9,
                mb_iter=1)
        mdl_test.model_param['NTF']['W'].factors[1] = mdl_train.model_param['minibatch']['WF'].factors[1].copy()
        mdl_test.model_param['NTF']['W'].factors[2] = mdl_train.model_param['minibatch']['WF'].factors[2].copy()
        mdl_test.model_param['LDS'] = mdl_train.model_param['LDS']
        mdl_test.model_param['REG'] = mdl_train.model_param['REG']
        normT = np.linalg.norm(tensor_test)
        mdl_test.model_param['minibatch']['WF'].rescale(normT)
        mdl_test.model_param['NTF']['W'].rescale(normT)
        mdl_test = minibatch_train(mdl_test, fixed_axes=[1,2])

        for key in mb_params:
                    xval_dict[key].append(mb_params[key])
        xval_dict['fold'].append(fold)

        xval_dict['train_model'].append(mdl_train)
        xval_dict['train_fold_ix'].append(folds[list(train_fold)].reshape(-1))
        xval_dict['train_beta_cost'].append(mdl_train.model_param['minibatch']['training']['beta_cost'][-1])
        xval_dict['train_beta_cost_pop'].append(mdl_train.model_param['minibatch']['training']['beta_cost_pop'])
        xval_dict['train_beta_cost_event_avg'].append(mdl_train.model_param['minibatch']['training']['beta_cost_event_avg'])
        xval_dict['train_beta_cost_chan_avg'].append(mdl_train.model_param['minibatch']['training']['beta_cost_chan_avg'])
        xval_dict['train_l2_cost'].append(mdl_train.model_param['minibatch']['training']['l2_cost'][-1])
        xval_dict['train_l2_cost_pop'].append(mdl_train.model_param['minibatch']['training']['l2_cost_pop'])
        xval_dict['train_l2_cost_event_avg'].append(mdl_train.model_param['minibatch']['training']['l2_cost_event_avg'])
        xval_dict['train_l2_cost_chan_avg'].append(mdl_train.model_param['minibatch']['training']['l2_cost_chan_avg'])

        xval_dict['test_model'].append(mdl_test)
        xval_dict['test_fold_ix'].append(folds[list(test_fold)].reshape(-1))
        xval_dict['test_beta_cost'].append(mdl_test.model_param['minibatch']['training']['beta_cost'][-1])
        xval_dict['test_beta_cost_pop'].append(mdl_test.model_param['minibatch']['training']['beta_cost_pop'])
        xval_dict['test_beta_cost_event_avg'].append(mdl_test.model_param['minibatch']['training']['beta_cost_event_avg'])
        xval_dict['test_beta_cost_chan_avg'].append(mdl_test.model_param['minibatch']['training']['beta_cost_chan_avg'])
        xval_dict['test_l2_cost'].append(mdl_test.model_param['minibatch']['training']['l2_cost'][-1])
        xval_dict['test_l2_cost_pop'].append(mdl_test.model_param['minibatch']['training']['l2_cost_pop'])
        xval_dict['test_l2_cost_event_avg'].append(mdl_test.model_param['minibatch']['training']['l2_cost_event_avg'])
        xval_dict['test_l2_cost_chan_avg'].append(mdl_test.model_param['minibatch']['training']['l2_cost_chan_avg'])

        mdl_train = clear_model_cache(mdl_train)
        mdl_test = clear_model_cache(mdl_test)

    xval_dict = pd.DataFrame.from_dict(xval_dict)
    return xval_dict


def clear_model_cache(mdl):
    mdl.model_param['minibatch']['tensor'] = None
    mdl.model_param['minibatch']['training'] = None
    return mdl
