import logging
import argparse
import textwrap
import numpy as np
import matplotlib.pyplot as plt

from spikennet.learn import dnn_validate
from spikennet.models import SpikeDNNet, IzhikevichAF, SigmoidAF
from spikennet.utils.dataset import ExpData
from spikennet.utils.logger import get_logger
from spikennet.utils.prepare import gen_folds, prep_files
from spikennet.utils.plot import plot_experiment, plot_article

DIM = 2
K_PNTS = 1
N_EPOCHS = 1
M_FOLDS = 1
KEY_INDEX = 1

parser = argparse.ArgumentParser(
    prog='main.py',
    formatter_class=argparse.RawDescriptionHelpFormatter
)

parser.add_argument('-a',   dest='act_func', type=str, default='izhikevich', help='use izhikevich or sigmoidal activation function')
parser.add_argument('-d',   dest='dataset', type=str, default='winter', help='set dataset for learning')
parser.add_argument('-p',   dest='plot', type=bool, default=False, help='save figs of experimental data')

args = parser.parse_args()

if __name__ == '__main__':

    logger = get_logger()

    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)

    act_izh = IzhikevichAF(dim=DIM)
    act_sig = SigmoidAF()

    mdls_config = {
        'izhikevich': {
            'act_func_1': act_izh.map,
            'act_func_2': act_izh.map,
            'dim': DIM,
            'mat_A': 20 * np.diag([-1, -2]),
            'mat_P': 1575.9 * np.diag([60, 40]),
            'mat_K_1': 0.15 * np.diag([10, 1]),
            'mat_K_2': 0.15 * np.diag([1, 1]),
            'mat_W_1': 20 * np.ones((DIM, DIM)),
            'mat_W_2': 20 * np.ones((DIM, DIM))
        },
        'sigmoidal': {
            'act_func_1': act_sig.map,
            'act_func_2': act_sig.map,
            'dim': DIM,
            'mat_A': 20 * np.diag([-2, -2]),
            'mat_P': 1575.9 * np.diag([60, 40]),
            'mat_K_1': 0.0001 * np.diag([20, 10]),
            'mat_K_2': 0.0001 * np.diag([20, 10]),
            'mat_W_1': 0.1 * np.ones((DIM, DIM)),
            'mat_W_2': 20 * np.ones((DIM, DIM))
            }
    }

    dnn = SpikeDNNet(**mdls_config[args.act_func])

    if args.dataset == 'winter':
        exp_data = ExpData('data_132591818490899344_.txt')
        exp_data.prep()

        keys = exp_data.keys
        cols = exp_data.columns

        data = exp_data.get_data(KEY_INDEX)
    else:
        data = prep_files(args.dataset) # flights, optokinetics

    if type(data) == list:
        folds = []
        for df in data[:]:
            fold, width, split = gen_folds(df, n_folds=M_FOLDS)
            folds.append(fold[0])
    else: 
        folds, width, split = gen_folds(data, n_folds=M_FOLDS)
        
    time = np.linspace(0, width, width)
    (tr_res, vl_res, mse_res, mae_res, smae_res,
    norms_W_1, norms_W_2, weights_W_1, weights_W_2) = dnn_validate(dnn,
                                                                   folds,
                                                                   n_epochs=N_EPOCHS,
                                                                   k_points=K_PNTS)
    print("\n    Count of experiment: {}\n".format(len(folds)))

    print("""
        Activation: {}, count epochs: {}, MA data-points: {}

        MSE train: mean={:2.6f}, std={:2.6f} valid: mean={:2.6f}, std={:2.6f}
        MAE train: mean={:2.6f}, std={:2.6f} valid: mean={:2.6f}, std={:2.6f}
        sMAE train: mean={:2.6f}, std={:2.6f} valid: mean={:2.6f}, std={:2.6f}
    """.format(args.act_func, N_EPOCHS, K_PNTS,
            np.mean(mse_res[:, 0]), np.std(mse_res[:, 0]),
            np.mean(mse_res[:, 1]), np.std(mse_res[:, 1]),
            np.mean(mae_res[:, 0]), np.std(mae_res[:, 0]),
            np.mean(mae_res[:, 1]), np.std(mae_res[:, 1]),
            np.mean(smae_res[:, 0]), np.std(smae_res[:, 0]),
            np.mean(smae_res[:, 1]), np.std(smae_res[:, 1])
        )
    )

    print("""
        Activation: {}, count epochs: {}, MA data-points: {}

        MSE train: {}

        MAE train: {}

        sMAE train: {}
    """.format('Sigmoid', N_EPOCHS, K_PNTS,
            mse_res[:, 0],
            mae_res[:, 0],
            smae_res[:, 0]
        )
    )

    if args.plot:
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