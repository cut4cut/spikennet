import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spikennet.learn import dnn_validate
from spikennet.models import SpikeDNNet, IzhikevichAF, SigmoidAF
from spikennet.utils.dataset import ExpData
from spikennet.utils.logger import get_logger
from spikennet.utils.prepare import gen_folds
from spikennet.utils.plot import plot_experiment, plot_article

parser = argparse.ArgumentParser(description='Start model fit.')
parser.add_argument('-model',  type=str, default='GB', help="Model")
args = parser.parse_args()

if __name__ == '__main__':
    KEY_INDEX = 1

    logger = get_logger()

    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)

    exp_data = ExpData('data_132591818490899344_.txt')
    exp_data.prep()

    keys = exp_data.keys
    cols = exp_data.columns

    data = exp_data.get_data(KEY_INDEX)
    folds, width, split = gen_folds(data, n_folds=2)
    time = np.linspace(0, width, width)

    act = IzhikevichAF() # IzhikevichAF()

    dnn = SpikeDNNet(act.map, act.map, 2)
    # dnn = SigmaDNNet(2)

    k_pnts = 1
    (tr_res, vl_res, mse_res, mae_res, smae_res,
     norms_W_1, norms_W_2, weights_W_1, weights_W_2) = dnn_validate(dnn,
                                                                    folds[:1],
                                                                    n_epochs=1,
                                                                    k_points=k_pnts)

    print("""
        Activation: {}, count epochs: {}, MA data-points: {}
         MSE train: mean={:2.6f}, std={:2.6f} valid: mean={:2.6f}, std={:2.6f}
         MAE train: mean={:2.6f}, std={:2.6f} valid: mean={:2.6f}, std={:2.6f}
        sMAE train: mean={:2.6f}, std={:2.6f} valid: mean={:2.6f}, std={:2.6f}
    """.format('Sigmoid', 1, k_pnts,
               np.mean(mse_res[:, 0]), np.std(mse_res[:, 0]),
               np.mean(mse_res[:, 1]), np.std(mse_res[:, 1]),
               np.mean(mae_res[:, 0]), np.std(mae_res[:, 0]),
               np.mean(mae_res[:, 1]), np.std(mae_res[:, 1]),
               np.mean(smae_res[:, 0]), np.std(smae_res[:, 0]),
               np.mean(smae_res[:, 1]), np.std(smae_res[:, 1])
        )
    )

    if True:
        for i, fold in enumerate(folds[:1]):

            error = np.abs(fold[0][0][:, 0] - tr_res[i][:, 0])
            wdiff = [np.diff(weights_W_1[i], axis=0)[:, :, :1].reshape(-1, 2),
                     np.diff(weights_W_2[i], axis=0)[:, :, :1].reshape(-1, 2)]

            if False:
                plot_experiment(i, time, split, width,
                                tr_target=fold[0][0],
                                tr_control=fold[0][1],
                                vl_target=fold[1][0],
                                vl_control=fold[1][1],
                                tr_est=tr_res[i],
                                vl_pred=vl_res[i],
                                norms_W_1=norms_W_1,
                                norms_W_2=norms_W_2)

            plot_article(i, time, split,
                         tr_target=fold[0][0],
                         tr_control=fold[0][1],
                         tr_est=tr_res[i],
                         error=error,
                         weaights_dyn=wdiff)
