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
import tensortools as tt


eps = np.finfo(np.float).resolution
calc_cost = lambda x: (
        np.linalg.norm(x.model_param['minibatch']['tensor'] - x.model_param['minibatch']['WF'].full()) /
        np.linalg.norm(x.model_param['minibatch']['tensor']))

def adjust_neg(tensor, method='exponential'):
    """
    Adjust the negative values in a tensor so they are strictly non-negative.

    Parameters
    ----------
    tensor: numpy.ndarray
        Arbitrary tensor containing negative and positive valued data

    method: ['exponential', 'shift', 'absolute']
        exponential - exponential transform of the values (i.e. e^X)
        shift - subtract minimum value from the tensor
        absolute - absolute value of the tensor

    Returns
    -------
    tensor: numpy.ndarray
        Arbitrary tensor with strictly non-negative values.
    """

    if (tensor >= 0).all():
        return tensor

    if method == 'exponential':
        tensor = np.exp(tensor / np.abs(tensor).max())
    elif method == 'shift':
        tensor = tensor - tensor.min()
    elif method == 'absolute':
        tensor = np.abs(tensor)
    else:
        tensor = tensor

    assert (tensor >= 0).all()
    return tensor


def minibatch_setup(tensor, rank, beta, LDS, mb_size, mb_epochs, mb_tol, mb_iter):
    """
    Adjust the negative values in a tensor so they are strictly non-negative.

    Parameters
    ----------

    Returns
    -------
    """

    n_obs, n_samp, n_chan = tensor.shape

    n_batches = int(np.ceil(n_obs / mb_size))
    pad_obs = (n_batches * mb_size) - n_obs

    obs_shuf_ix = np.concatenate((
        np.random.permutation(n_obs),
        np.random.permutation(n_obs)[:pad_obs]))
    n_obs_shuf = len(obs_shuf_ix)

    batches = obs_shuf_ix.reshape(-1, mb_size)

    print('--- Mini-Batch Setup ---')
    print(' :: # of observations - {}'.format(n_obs))
    print(' :: mini-batch size   - {}'.format(mb_size))
    print(' :: # of mini-batches - {}'.format(batches.shape[0]))
    print('========================\n')

    tensor_bdummy = np.zeros_like(tensor)[:mb_size]
    if LDS:
        mdl = tt.ncp_nnlds.init_model(
            X=tensor_bdummy, 
            rank=rank,
            REG_dict=None,
            LDS_dict={
                'axis': 1,
                'beta': beta,
                'lag_state': 1,
                'lag_exog': 1,
                'init': 'rand'},
            exog_input=np.zeros((tensor_bdummy.shape[1], 1)))
    else:
        mdl = tt.ncp_nnlds.init_model(
            X=tensor_bdummy, 
            rank=rank,
            REG_dict=None, #{'axis': 2, 'l1_ratio': 0.5, 'alpha': 1e3},
            LDS_dict=None, 
            exog_input=None)

    mdl.model_param['NTF']['beta'] = beta

    mdl.model_param['minibatch'] = {}

    mdl.model_param['minibatch']['tensor'] = tensor

    mdl.model_param['minibatch']['WF'] = tt.KTensor(
            [np.random.uniform(size=(n_obs, rank)),
             mdl.model_param['NTF']['W'].factors[1],
             mdl.model_param['NTF']['W'].factors[2]])
    normT = np.linalg.norm(tensor)
    mdl.model_param['minibatch']['WF'].rescale(normT)

    mdl.model_param['NTF']['W'].factors[0] = mdl.model_param['minibatch']['WF'].factors[0][:mb_size].copy()
    mdl.model_param['NTF']['W'].factors[1] = mdl.model_param['minibatch']['WF'].factors[1].copy()
    mdl.model_param['NTF']['W'].factors[2] = mdl.model_param['minibatch']['WF'].factors[2].copy()

    mdl.model_param['minibatch']['training'] = {
            'cost': [calc_cost(mdl)],
            'mb_iter': mb_iter,
            'mb_epochs': mb_epochs,
            'mb_size': mb_size,
            'mb_tol': mb_tol,
            'mbatches': batches,
            'obs_shuf_ix': obs_shuf_ix}

    return mdl


def minibatch_train(mdl, fixed_axes=[]):
    """
    Adjust the negative values in a tensor so they are strictly non-negative.

    Parameters
    ----------

    Returns
    -------
    """

    for ep_i in tqdm(range(mdl.model_param['minibatch']['training']['mb_epochs']), position=0, leave=True):
        batches = mdl.model_param['minibatch']['training']['mbatches'][
            np.random.permutation(
                mdl.model_param['minibatch']['training']['mbatches'].shape[0])]

        for bbb in tqdm(batches, position=0, leave=True):
            tensor_batch = mdl.model_param['minibatch']['tensor'][bbb]

            mdl.model_param['NTF']['W'].factors[0][:,:] = \
                mdl.model_param['minibatch']['WF'].factors[0][bbb, :].copy()

            mdl = tt.ncp_nnlds.model_update(
                tensor_batch,
                mdl,
                fixed_axes=fixed_axes,
                mask=np.ones_like(tensor_batch).astype(bool),
                exog_input=(np.zeros((tensor_batch.shape[1], 1)) 
                    if mdl.model_param['LDS'] is not None else None),
                fit_dict={
                    'max_iter': mdl.model_param['minibatch']['training']['mb_iter'],
                    'tol': mdl.model_param['minibatch']['training']['mb_tol'],
                    'verbose': False})

            mdl.model_param['minibatch']['WF'].factors[0][bbb, :] = \
                mdl.model_param['NTF']['W'].factors[0][:,:].copy()

        mdl.model_param['minibatch']['WF'].factors[1] = \
            mdl.model_param['NTF']['W'].factors[1][:,:].copy()

        mdl.model_param['minibatch']['WF'].factors[2] = \
            mdl.model_param['NTF']['W'].factors[2][:,:].copy()

        mdl.model_param['minibatch']['WF'].rescale(
            np.linalg.norm(mdl.model_param['minibatch']['tensor']))

        mdl.model_param['minibatch']['training']['cost'].append(
            calc_cost(mdl))
        cost = mdl.model_param['minibatch']['training']['cost']

        if not np.isfinite(cost).any():
            break

        last_cost = cost[-1]
        delta_cost = np.log(cost[-1]/cost[-2])
        total_cost = np.log(cost[-1]/cost[0])

        print('{} : DELTA: {} : TOTAL: {}'.format(
            last_cost, delta_cost, total_cost))

        if np.abs(delta_cost) < mdl.model_param['minibatch']['training']['mb_tol']:
            break

    return mdl


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

    obs_shuf_ix = np.concatenate((
        np.random.permutation(n_obs),
        np.random.permutation(n_obs)[:pad_obs]))
    n_obs_shuf = len(obs_shuf_ix)

    folds = obs_shuf_ix.reshape(-1, obs_per_fold)

    print('--- Cross-Validation Setup ---')
    print(' :: # of observations - {}'.format(n_obs))
    print(' :: fold size         - {}'.format(obs_per_fold))
    print(' :: # of folds        - {}'.format(n_fold))
    print('========================\n')


    xval_dict = {'fold': [],
                 'train_model': [],
                 'test_model': [],
                 'train_cost_abs': [],
                 'test_cost_abs': [],
                 'train_cost_rel': [],
                 'test_cost_rel': []}
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
                LDS=mb_params['LDS'],
                beta=mb_params['beta'],
                mb_size=mb_params['mb_size'],
                mb_epochs=5,
                mb_tol=1e-3,
                mb_iter=1)
        mdl_train = minibatch_train(mdl_train, fixed_axes=[])

        print('*******TESTING*******')
        tensor_test = tensor[folds[list(test_fold)].reshape(-1)]
        mdl_test = minibatch_setup(
                tensor=tensor_test,
                rank=mb_params['rank'], 
                LDS=mb_params['LDS'],
                beta=mb_params['beta'],
                mb_size=tensor_test.shape[0],
                mb_epochs=1,
                mb_tol=1e-3,
                mb_iter=1)
        mdl_test.model_param['NTF']['W'].factors[1] = mdl_train.model_param['NTF']['W'].factors[1].copy()
        mdl_test.model_param['NTF']['W'].factors[2] = mdl_train.model_param['NTF']['W'].factors[2].copy()
        mdl_test.model_param['LDS'] = mdl_train.model_param['LDS']
        mdl_test.model_param['REG'] = mdl_train.model_param['REG']
        mdl_test = minibatch_train(mdl_test, fixed_axes=[1,2])

        cost_train = mdl_train.model_param['minibatch']['training']['cost'].copy()
        cost_test = mdl_test.model_param['minibatch']['training']['cost'].copy()

        mdl_train = clear_model_cache(mdl_train)
        mdl_test = clear_model_cache(mdl_test)

        xval_dict['fold'].append(fold)
        xval_dict['train_model'].append(mdl_train)
        xval_dict['test_model'].append(mdl_test)
        xval_dict['train_cost_abs'].append(cost_train[-1])
        xval_dict['test_cost_abs'].append(cost_test[-1])
        xval_dict['train_cost_rel'].append(np.log(cost_train[-1] / cost_train[0]))
        xval_dict['test_cost_rel'].append(np.log(cost_test[-1] / cost_test[0]))
        for key in mb_params:
            xval_dict[key].append(mb_params[key])

    xval_dict = pd.DataFrame.from_dict(xval_dict)
    return xval_dict


def clear_model_cache(mdl):
    mdl.model_param['minibatch']['tensor'] = None
    mdl.model_param['minibatch']['training'] = None
    return mdl
