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

        (N, M) = data.shape
        tmp_data = np.fft.rfft(data, axis=1)
        len_ = tmp_data.shape[1]

        phases = np.random.uniform(low=0, high=2*np.pi, size=(N, len_))
        tmp_data *= np.exp(1j*phases)

        return np.real(np.fft.irfft(tmp_data, n=M, axis=1))

    @staticmethod
    def rand_curve(data: np.array,
                   sigma: float = 0.2,
                   loc: float = 0.0,
                   knot: int = 4) -> np.array:

        r"""Apllies data augmentation by generation random curves as described
        in the paper `Data Augmentation of Wearable Sensor Data
        for Parkinson's Disease Monitoring using Convolutional Neural Networks
        <https://arxiv.org/abs/1706.00527>
        <https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data>`."""

        (N, M) = data.shape
        x_range = np.arange(N)
        tmp_data = np.empty_like(data)
        xx = np.arange(0, N, (N-1)/(knot+1))
        yy = np.random.normal(loc=loc, scale=sigma, size=(xx.shape[0], M))

        for i, el in enumerate(range(M)):
            cs = CubicSpline(xx, yy[:, i])
            tmp_data[:, i] = cs(x_range)

        return tmp_data

    @staticmethod
    def __axangle2mat(axis, angle):
        (x, y) = axis

        n = np.sqrt(x*x + y*y)
        x = x/n
        y = y/n
        z = 0

        c = np.cos(angle); s = np.sin(angle); C = 1-c
        xs = x*s;   ys = y*s;   zs = z*s
        xC = x*C;   yC = y*C;   zC = z*C
        xyC = x*yC; yzC = y*zC; zxC = z*xC

        return np.array([[ x*xC+c,   xyC-zs],
                         [ xyC+zs,   y*yC+c] ])

    #@staticmethod
    def rotate(self, data: np.array) -> np.array:

        r"""Apllies data augmentation by rotation as described in the paper
        `Data Augmentation of Wearable Sensor Data for Parkinson's Disease
        Monitoring using Convolutional Neural Networks
        <https://arxiv.org/abs/1706.00527>
        <https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data>`."""

        (N, M) = data.shape

        axis = np.random.uniform(low=-1, high=1, size=M)
        angle = np.random.uniform(low=-np.pi, high=np.pi)

        if M == 2:
            tmp = np.matmul(data, self.__axangle2mat(axis, angle))
        else:
            tmp = np.matmul(data, axangle2mat(axis, angle))

        return tmp

    @staticmethod
    def permutate(data: np.array,
                  nPerm: int = 2,
                  minSegLength: int = 20) -> np.array:

        r"""Apllies data augmentation by permutation as described in the paper
        `Data Augmentation of Wearable Sensor Data for Parkinson's Disease
        Monitoring using Convolutional Neural Networks
        <https://arxiv.org/abs/1706.00527>
        <https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data>`."""

        (N, M) = data.shape
        X_new = np.zeros((N, M))
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
            X_new[pp:pp+len(x_temp), :] = x_temp
            pp += len(x_temp)

        return X_new
