import numpy as np

from scipy.interpolate import CubicSpline
from transforms3d.axangles import axangle2mat


class DataGenerator(object):

    def __init__(self):
        self.original_data = None
        self.synthetic_data = None

    @staticmethod
    def fft(data: np.array) -> np.array:

        r"""Apllies data augmentation by FFT as described in the paper
        `Time Series Data Augmentation for Deep Learning: A Survey
        <https://arxiv.org/abs/2002.12478>."""

        data = data.T.copy()
        (N, M) = data.shape
        tmp_data = np.fft.rfft(data, axis=1)
        len_ = tmp_data.shape[1]

        phases = np.random.uniform(low=0, high=2*np.pi, size=(N, len_))
        tmp_data *= np.exp(1j*phases)

        return np.real(np.fft.irfft(tmp_data, n=M, axis=1)).T

    @staticmethod
    def rand_curve(data: np.array,
                   sigma: float = 0.2,
                   loc: float = 0.0,
                   knot: int = 4) -> np.array:

        r"""Apllies data augmentation by generation random curves as described
        in the paper `Data Augmentation of Wearable Sensor Data
        for Parkinson's Disease Monitoring using Convolutional Neural Networks
        <https://arxiv.org/abs/1706.00527>
        <https://github.com/terryum/\\
            Data-Augmentation-For-Wearable-Sensor-Data>`."""

        data = data.T.copy()
        (N, M) = data.shape
        x_range = np.arange(N)
        tmp_data = np.empty_like(data)
        xx = np.arange(0, N, (N-1)/(knot+1))
        yy = np.random.normal(loc=loc, scale=sigma, size=(xx.shape[0], M))

        for i, el in enumerate(range(M)):
            cs = CubicSpline(xx, yy[:, i])
            tmp_data[:, i] = cs(x_range)

        return tmp_data.T

    @staticmethod
    def permutate(data: np.array,
                  nPerm: int = 2,
                  minSegLength: int = 20) -> np.array:

        r"""Apllies data augmentation by permutation as described in the paper
        `Data Augmentation of Wearable Sensor Data for Parkinson's Disease
        Monitoring using Convolutional Neural Networks
        <https://arxiv.org/abs/1706.00527>
        <https://github.com/terryum/\\
            Data-Augmentation-For-Wearable-Sensor-Data>`."""

        (N, M) = data.shape
        tmp_data = np.zeros((N, M))
        idx = np.random.permutation(nPerm)
        bWhile = True
        pp = 0

        while bWhile:
            segs = np.zeros(nPerm+1, dtype=int)
            segs[1:-1] = np.sort(
                            np.random.randint(minSegLength,
                                              N - minSegLength,
                                              nPerm-1)
                            )
            segs[-1] = N

            if np.min(segs[1:]-segs[0:-1]) > minSegLength:
                bWhile = False

        for ii in range(nPerm):
            x_temp = data[segs[idx[ii]]:segs[idx[ii]+1], :]
            tmp_data[pp:pp+len(x_temp), :] = x_temp
            pp += len(x_temp)

        return tmp_data

    @staticmethod
    def time_inverse(data: np.array) -> np.array:

        return data[::-1].copy()
