from abc import ABC

import numpy as np


class ActFunc(ABC):
    def __init__(self):
        pass

    def map(self, input: np.ndarray) -> np.ndarray:
        return input


class IzhikevichAF(ActFunc):

    def __init__(self,
                 izh_border: float = 0.18,
                 param_a: float = 0.00002,
                 param_b: float = 0.035,
                 param_c: float = -0.055,
                 param_d: float = 0.05,
                 param_e: float = -0.065):

        self.izh_border = izh_border
        self.control = np.ones(2) * param_b * param_e
        self.state = np.ones(2) * param_e

        self.param_a = param_a
        self.param_b = param_b
        self.param_c = param_c
        self.param_d = param_d
        self.param_e = param_e

    def map(self,
            input: np.ndarray,
            step: float = 0.01) -> np.ndarray:

        _state = self.state + step * (
                            0.04 * self.state @ self.state
                            + 5 * self.state + 140 - self.control
                            + np.ones(2) * input
                        )

        self.control = self.control + step * (
                                        self.param_a * (
                                            self.param_b * self.state
                                            - self.control
                                            )
                                        )

        if np.all(_state > self.izh_border):
            self.state = np.ones(2) * self.param_c
            self.control = np.ones(2) * self.param_d
        else:
            self.state = _state

        return self.state


class SigmoidAF(ActFunc):

    def __init__(self,
                 param_a: float = 1.,
                 param_b: float = 1.,
                 param_c: float = 0.02,
                 param_d: float = -0.02):

        self.param_a = param_a
        self.param_b = param_b
        self.param_c = param_c
        self.param_d = param_d

    def map(self, input: np.ndarray) -> np.ndarray:
        state = self.param_a / (self.param_b
                                + self.param_c * np.exp(self.param_d * input))
        return state


class SpikeDNNet(object):

    def __init__(self,
                 act_func_1: ActFunc.map,
                 act_func_2: ActFunc.map,
                 dim: int = 2,
                 mat_A: np.ndarray = 20 * np.diag([-1, -2]),
                 mat_P: np.ndarray = 1575.9 * np.diag([60, 40]),
                 mat_K_1: np.ndarray = 0.15 * np.diag([10, 1]),
                 mat_K_2: np.ndarray = 0.15 * np.diag([1, 1]),
                 mat_W_1: np.ndarray = None,
                 mat_W_2: np.ndarray = None):

        self.mat_dim = dim

        self.mat_A = mat_A
        self.mat_P = mat_P

        self.mat_K_1 = mat_K_1
        self.mat_K_2 = mat_K_2

        self.afunc_1 = act_func_1
        self.afunc_2 = act_func_2

        self.mat_W_1 = None
        self.mat_W_2 = None

        self.init_mat_W_1 = mat_W_1 or 20 * np.ones((dim, dim))
        self.init_mat_W_2 = mat_W_2 or 20 * np.ones((dim, dim))

        self.array_hist_W_1 = None
        self.array_hist_W_2 = None

    @staticmethod
    def moving_average(x: np.ndarray, w: int = 2) -> np.ndarray:
        return np.convolve(x, np.ones(w), 'valid') / w

    def smooth(self, x: np.ndarray, w: int = 2) -> np.ndarray:
        l, m, n = x.shape
        new_x = np.ones((l-w+1, m, n))

        for i in range(m):
            for j in range(n):
                new_x[:, i, j] = self.moving_average(x[:, i, j], w)

        return new_x

    def fit(self,
            vec_x: np.ndarray,
            vec_u: np.ndarray,
            step: float = 0.01,
            n_epochs: int = 3,
            k_points: int = 2) -> np.ndarray:

        n = self.mat_dim
        nt = len(vec_u)
        vec_est = 0.1 * np.ones((nt, n))

        self.mat_W_1 = self.init_mat_W_1
        self.mat_W_2 = self.init_mat_W_2

        self.array_hist_W_1 = np.ones((nt, n, n))
        self.array_hist_W_2 = np.ones((nt, n, n))

        for e in range(1, n_epochs+1):
            if e % 2 == 0:
                vec_x = vec_x[::-1]
                vec_u = vec_u[::-1]

                self.mat_W_1 = self.smoothed_W_1[-1].copy()
                self.mat_W_2 = self.smoothed_W_2[-1].copy()

            elif e > 1:
                vec_x = vec_x[::-1]
                vec_u = vec_u[::-1]

                self.mat_W_1 = self.smoothed_W_1[-1].copy()
                self.mat_W_2 = self.smoothed_W_2[-1].copy()

            for i in range(nt-1):

                delta = vec_est - vec_x

                vec_est[i+1] = vec_est[i] + step * (
                                        self.mat_A@vec_est[i]
                                        + self.mat_W_1@self.afunc_1(vec_est[i])
                                        + self.mat_W_2@self.afunc_2(vec_est[i])
                                        * vec_u[i]
                                    )

                self.mat_W_1 = self.mat_W_1 - step * (
                                                self.mat_K_1
                                                @ self.mat_P
                                                @ delta[i]
                                                @ self.afunc_1(vec_est[i])
                                            )

                self.mat_W_2 = self.mat_W_2 - step * (
                                                self.mat_K_2
                                                @ self.mat_P
                                                @ delta[i]
                                                @ self.afunc_2(vec_est[i])
                                                * vec_u[i]
                                            )

                self.array_hist_W_1[i] = self.mat_W_1.copy()
                self.array_hist_W_2[i] = self.mat_W_2.copy()

            self.smoothed_W_1 = self.smooth(self.array_hist_W_1, k_points)
            self.smoothed_W_2 = self.smooth(self.array_hist_W_2, k_points)

        return vec_est

    def predict(self,
                init_state: np.ndarray,
                vec_u: np.ndarray,
                step: float = 0.01) -> np.ndarray:

        n = self.mat_dim
        nt = len(vec_u)
        vec_est = init_state * np.ones((nt, n))

        mat_W_1 = self.smoothed_W_1[-1]
        mat_W_2 = self.smoothed_W_2[-1]

        for i in range(nt-1):

            vec_est[i+1] = vec_est[i] + step * (
                                    self.mat_A@vec_est[i]
                                    + mat_W_1@self.afunc_1(vec_est[i])
                                    + mat_W_2@self.afunc_2(vec_est[i])
                                    * vec_u[i]
                                )

        return vec_est


class SigmaDNNet(object):

    def __init__(self,
                 dim: int = 2,
                 mat_A: np.ndarray = 20 * np.diag([-2, -2]),
                 mat_P: np.ndarray = 1575.9 * np.diag([60, 40]),
                 mat_K_1: np.ndarray = 0.0001 * np.diag([20, 10]),
                 mat_K_2: np.ndarray = 0.0001 * np.diag([20, 10]),
                 mat_W_1: np.ndarray = None,
                 mat_W_2: np.ndarray = None,
                 param_a: float = 0.00002,
                 param_b: float = 0.035,
                 param_c: float = -0.055,
                 param_d: float = 0.05,
                 param_e: float = -0.065):

        self.mat_dim = dim

        self.mat_A = mat_A
        self.mat_P = mat_P

        self.mat_K_1 = mat_K_1
        self.mat_K_2 = mat_K_2

        self.mat_W_1 = None
        self.mat_W_2 = None

        self.init_mat_W_1 = mat_W_1 or 0.1 * np.ones((dim, dim))
        self.init_mat_W_2 = mat_W_2 or 20 * np.ones((dim, dim))

        self.array_hist_W_1 = None
        self.array_hist_W_2 = None

        self.param_a = param_a
        self.param_b = param_b
        self.param_c = param_c
        self.param_d = param_d
        self.param_e = param_e

    @staticmethod
    def moving_average(x: np.ndarray, w: int = 2) -> np.ndarray:
        return np.convolve(x, np.ones(w), 'valid') / w

    def smooth(self, x: np.ndarray, w: int = 2) -> np.ndarray:
        l, m, n = x.shape
        new_x = np.ones((l-w+1, m, n))

        for i in range(m):
            for j in range(n):
                new_x[:, i, j] = self.moving_average(x[:, i, j], w)

        return new_x

    def sigmoida(self, vec: np.ndarray) -> np.ndarray:
        return 1 / (1 + 0.2 * np.exp(-0.2 * vec))

    def act_func(self, type: str = 'sigmoida'):
        maper = {'sigmoida': self.sigmoida}
        return maper[type]

    def fit(self,
            vec_x: np.ndarray,
            vec_u: np.ndarray,
            step: float = 0.01,
            n_epochs: int = 3,
            k_points: int = 2) -> np.ndarray:

        n = self.mat_dim
        nt = len(vec_u)
        vec_est = 0.1 * np.ones((nt, n))
        act = self.sigmoida

        self.mat_W_1 = self.init_mat_W_1
        self.mat_W_2 = self.init_mat_W_2

        self.array_hist_W_1 = np.ones((nt, n, n))
        self.array_hist_W_2 = np.ones((nt, n, n))

        for e in range(1, n_epochs+1):
            if e % 2 == 0:
                vec_x = vec_x[::-1]
                vec_u = vec_u[::-1]

                self.mat_W_1 = self.smoothed_W_1[-1].copy()
                self.mat_W_2 = self.smoothed_W_2[-1].copy()

            elif e > 1:
                vec_x = vec_x[::-1]
                vec_u = vec_u[::-1]

                self.mat_W_1 = self.smoothed_W_1[-1].copy()
                self.mat_W_2 = self.smoothed_W_2[-1].copy()

            for i in range(nt-1):
                delta = vec_est - vec_x

                vec_est[i+1] = vec_est[i] + step * (
                                            self.mat_A@vec_est[i]
                                            + self.mat_W_1@act(vec_est[i])
                                            + self.mat_W_2@act(vec_est[i])
                                            * vec_u[i]
                                        )

                self.mat_W_1 = self.mat_W_1 - step * (
                                                self.mat_K_1
                                                @ self.mat_P
                                                @ delta[i]
                                                @ act(vec_est[i])
                                            )

                self.mat_W_2 = self.mat_W_2 - step * (
                                                self.mat_K_2
                                                @ self.mat_P
                                                @ delta[i]
                                                @ act(vec_est[i])
                                                * vec_u[i]
                                            )

                self.array_hist_W_1[i] = self.mat_W_1.copy()
                self.array_hist_W_2[i] = self.mat_W_2.copy()

            self.smoothed_W_1 = self.smooth(self.array_hist_W_1, k_points)
            self.smoothed_W_2 = self.smooth(self.array_hist_W_2, k_points)

        return vec_est

    def predict(self,
                init_state: np.ndarray,
                vec_u: np.ndarray,
                step: float = 0.01) -> np.ndarray:

        n = self.mat_dim
        nt = len(vec_u)
        vec_est = init_state * np.ones((nt, n))
        act = self.act_func()

        mat_W_1 = self.smoothed_W_1[-1]
        mat_W_2 = self.smoothed_W_2[-1]

        for i in range(nt-1):

            vec_est[i+1] = vec_est[i] + step * (
                                    self.mat_A@vec_est[i]
                                    + mat_W_1@act(vec_est[i])
                                    + mat_W_2@act(vec_est[i])
                                    * vec_u[i]
                                )

        return vec_est
