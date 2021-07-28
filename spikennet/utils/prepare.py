import os
import re
import logging

import numpy as np
import pandas as pd

from scipy import fftpack
from numpy.fft import rfft, irfft, rfftfreq


logger = logging.getLogger('spikennet.main')


def dff_transform(data: np.ndarray,
                  freq_bounds: tuple = (0, 0.99)) -> np.ndarray:
    length, ncols = data.shape
    data_tranform = np.empty_like(data)
    frequencies = rfftfreq(length, d=1e-2)
    min_f, max_f = freq_bounds

    for i in range(ncols):
        fourier = rfft(data[:, i])
        ft_threshed = fourier.copy()
        ft_threshed[(min_f >= frequencies)] = 0
        ft_threshed[(max_f <= frequencies)] = 0
        ifourier = irfft(ft_threshed, length)
        data_tranform[:, i] = ifourier.copy()

    return data_tranform


def gen_folds(data: pd.DataFrame,
              n_folds: int = 5,
              freq_bounds: tuple = (0, 5.99),
              tr_perc: float = 0.75,
              t_cols: tuple = (2, 4),
              win_bounds: tuple = (20, 20),
              transform_flg: bool = False):

    folds = []
    from_col, to_col = t_cols
    r_bound, l_bound = win_bounds
    width = int(len(data)/n_folds)
    split = int(tr_perc * width)
    start = 0
    end = width
    for i in range(n_folds):
        fold = []

        if transform_flg:
            targets = dff_transform(
                            data.iloc[start:end, from_col:to_col].values,
                            freq_bounds)
        else:
            targets = data.iloc[start:end, from_col:to_col].values

        targets = data.iloc[start:end, from_col:to_col].values
        controls = data.iloc[start:end, 5].values

        datasets = [(targets[r_bound:split],
                     controls[r_bound:split]),
                    (targets[split:width-l_bound],
                     controls[split:width-l_bound])]

        for target, control in datasets:
            fold.append((target, control))
        folds.append(fold)
        start = end
        end += width

    return (folds, width-(r_bound+l_bound), split-r_bound)


def convert(s: str) -> np.array:
    s = re.sub('\n', ' ', s)
    s = re.sub(',', '.', s)
    return np.array([float(i) for i in s.split(';')], dtype=np.float32)


def load_txt(file_name: str, size: int = 9) -> np.array:
    data = np.ones(size)
    with open(file_name, 'r') as file:
        cnt = 0
        for line in file:
            try:
                row = convert(line)
                data = np.vstack((data, row))
            except ValueError:
                cnt += 1
                if cnt > 1:
                    break
    return data[1:]


def make_learn_data(folder: str) -> None:
    rdata_folder = './data/raw/' + folder
    pdata_folder = './data/prep/' + folder

    files = [f for f in os.listdir(rdata_folder)
             if os.path.isfile(os.path.join(rdata_folder, f)) and '.' in f]

    try:
        os.mkdir(pdata_folder)
        logger.info('Directory was created')
    except FileExistsError:
        logger.info('Directory already exist')

    for f in files:
        file_path = '{}/{}'.format(rdata_folder, f)
        data = load_txt(file_path)
