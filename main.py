import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spikennet.learn import dnn_validate
from spikennet.utils.dataset import ExpData
from spikennet.utils.logger import get_logger
from spikennet.utils.prepare import gen_folds

parser = argparse.ArgumentParser(description='Start model fit.')
parser.add_argument('-model',  type=str, default='GB', help="Model")
args = parser.parse_args()

if __name__ == '__main__':
    KEY_INDEX = 1

    logger = get_logger()

    exp_data = ExpData('data_132591818490899344_.txt')
    exp_data.prep()

    keys = exp_data.keys
    cols = exp_data.columns

    data = exp_data.get_data(KEY_INDEX)
    folds, width, split = gen_folds(data, n_folds=25)
    time = np.linspace(0, width, width)

    (tr_res, vl_res, mse_res, mae_res,
     r2_res, norms_W_1, norms_W_2) = dnn_validate(folds)

    for i, fold in enumerate(folds):
        tr_target = fold[0][0]
        tr_control = fold[0][1]

        vl_target = fold[1][0]
        vl_control = fold[1][1]

        tr_est = tr_res[i]
        vl_pred = vl_res[i]

        fig, ax = plt.subplots(3, figsize=(18, 12))

        ax[0].plot(time[:split], tr_target[:, 0])
        ax[0].plot(time[split:], vl_target[:, 0])
        ax[0].plot(time[:split], tr_est[:, 0])
        ax[0].plot(time[split:], vl_pred[:, 0])
        ax[0].axvline(x=split, c='grey', linestyle='--')
        ax[0].legend([
              'train', 'valid',
              'train predict', 'valid predict'
             ])

        ax[1].plot(time[:split], tr_control)
        ax[1].plot(time[split:], vl_control)
        ax[1].axvline(x=split, c='grey', linestyle='--')
        ax[1].legend(['train control', 'valid control'])

        ax[2].plot(time[:split-1], norms_W_1[i])
        ax[2].plot(time[:split-1], norms_W_2[i])
        ax[2].plot(time[split:], np.ones((width-split)) * norms_W_1[i][-1])
        ax[2].plot(time[split:], np.ones((width-split)) * norms_W_2[i][-1])
        ax[2].axvline(x=split, c='grey', linestyle='--')
        ax[2].legend(['train frobenius norm W1', 'train frobenius norm W2',
                      'valid frobenius norm W1', 'valid frobenius norm W2'])

        fig.savefig('./report/fold_{}.png'.format(i))

    print('Train shape={}, valid shape={}'.format(tr_control.shape, vl_control.shape))
    print("""
    DNN for {} folds:
        MSE train: mean={:2.6f}, std={:2.6f} valid: mean={:2.6f}, std={:2.6f}
        MAE train: mean={:2.6f}, std={:2.6f} valid: mean={:2.6f}, std={:2.6f}
        R2  train: mean={:2.6f}, std={:2.6f} valid: mean={:2.6f}, std={:2.6f}
    """.format( 5,
            np.mean(mse_res[:, 0]), np.std(mse_res[:, 0]), np.mean(mse_res[:, 1]), np.std(mse_res[:, 1]),
            np.mean(mae_res[:, 0]), np.std(mae_res[:, 0]), np.mean(mae_res[:, 1]), np.std(mae_res[:, 1]),
            np.mean(r2_res[:, 0]), np.std(r2_res[:, 0]), np.mean(r2_res[:, 1]), np.std(r2_res[:, 1])
    ))