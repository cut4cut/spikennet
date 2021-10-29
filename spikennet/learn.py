import numpy as np

from typing import Union
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .models import SpikeDNNet


def dnn_validate(dnn: SpikeDNNet,
                 folds: list,
                 n_epochs: int = 1,
                 k_points: int = 2) -> tuple:
    """Learn model on folds and
        save metrics of learning.
    """
    mse_res = np.ones((len(folds), 2))
    mae_res = np.ones((len(folds), 2))
    smae_res = np.ones((len(folds), 2))
    weights_W_1 = []
    weights_W_2 = []
    norms_W_1 = []
    norms_W_2 = []
    tr_res = {}
    vl_res = {}

    for i, fold in enumerate(folds):
        tr_target = fold[0][0]
        tr_control = fold[0][1]

        vl_target = fold[1][0]
        vl_control = fold[1][1]

        snn = dnn
        target_est = snn.fit(tr_target, tr_control) #,
                            # n_epochs=n_epochs, k_points=k_points)

        vl_pred = snn.predict(target_est[-1][0], vl_control)

        mse_res[i][0] = mean_squared_error(tr_target[:, 0], target_est[:, 0])
        mse_res[i][1] = mean_squared_error(vl_target[:, 0], vl_pred[:, 0])

        mae_res[i][0] = mean_absolute_error(tr_target[:, 0], target_est[:, 0])
        mae_res[i][1] = mean_absolute_error(vl_target[:, 0], vl_pred[:, 0])

        smae_res[i][0] = mae_res[i][0] / np.mean(tr_target[:, 0])
        smae_res[i][1] = mae_res[i][1] / np.mean(vl_target[:, 0])

        tr_res[i] = target_est
        vl_res[i] = vl_pred

        weights_W_1.append(snn.array_hist_W_1)
        weights_W_2.append(snn.array_hist_W_2)

        norms_W_1.append(np.linalg.norm(snn.array_hist_W_1[:-1], axis=2)[:, 0])
        norms_W_2.append(np.linalg.norm(snn.array_hist_W_2[:-1], axis=2)[:, 0])

    return (tr_res, vl_res, mse_res, mae_res, smae_res,
            norms_W_1, norms_W_2, weights_W_1, weights_W_2)

def dnn_validate_alter_models(model,
                 folds: list,
                 n_epochs: int = 1,
                 k_points: int = 2) -> tuple:
    """Learn model on folds and
        save metrics of learning.
    """
    mse_res = np.ones((len(folds), 2))
    mae_res = np.ones((len(folds), 2))
    smae_res = np.ones((len(folds), 2))
    weights_W_1 = []
    weights_W_2 = []
    norms_W_1 = []
    norms_W_2 = []
    tr_res = {}
    vl_res = {}

    for i, fold in enumerate(folds):
        tr_target = fold[0][0]
        tr_control = fold[0][1]

        vl_target = fold[1][0]
        vl_control = fold[1][1]

        model.fit(tr_target[:,0], tr_control.reshape(-1, 1)) #,
                            # n_epochs=n_epochs, k_points=k_points)

        target_est = model.predict(tr_control)

        mse_res[i][0] = mean_squared_error(tr_target[:, 0], target_est[:, 0])
        mse_res[i][1] = mean_squared_error(vl_target[:, 0], vl_target[:, 0])

        mae_res[i][0] = mean_absolute_error(tr_target[:, 0], target_est[:, 0])
        mae_res[i][1] = mean_absolute_error(vl_target[:, 0], vl_target[:, 0])

        smae_res[i][0] = mae_res[i][0] / np.mean(tr_target[:, 0])
        smae_res[i][1] = mae_res[i][1] / np.mean(vl_target[:, 0])

        tr_res[i] = target_est
        vl_res[i] = vl_target

    return (tr_res, vl_res, mse_res, mae_res, smae_res,
            norms_W_1, norms_W_2, weights_W_1, weights_W_2)