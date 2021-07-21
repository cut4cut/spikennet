import numpy as np

from scipy.ndimage import gaussian_filter


class SpikeNNet(object):

    def __init__(self,
                 dim: int = 2,
                 mat_A: np.ndarray = 20 * np.diag([-1, -2]),
                 mat_P: np.ndarray = 1575.9 * np.diag([60, 40]),
                 mat_K_1: np.ndarray = 0.1 * np.diag([10, 1]),
                 mat_K_2: np.ndarray = 0.1 * np.diag([1, 1]),
                 mat_W_1: np.ndarray = None,
                 mat_W_2: np.ndarray = None,
                 izh_border: float = 0.15,
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

        self.mat_W_1 = mat_W_1 or 40 * np.ones((dim, dim))
        self.mat_W_2 = mat_W_2 or 40 * np.ones((dim, dim))

        self.array_hist_W_1 = None
        self.array_hist_W_2 = None

        self.izh_border = izh_border

        self.param_a = param_a
        self.param_b = param_b
        self.param_c = param_c
        self.param_d = param_d
        self.param_e = param_e

    def fit(self,
            vec_x: np.ndarray,
            vec_u: np.ndarray,
            step: float = 0.01) -> np.ndarray:

        # self.__init_fit(nt)
        n = self.mat_dim
        nt = len(vec_u)
        vec_est = 0.1 * np.ones((nt, n))
        vec_v = self.param_e * np.ones((nt, n))
        vec_u_izh = np.zeros((nt, n))
        vec_u_izh[0] = self.param_b * vec_v[0]

        self.array_hist_W_1 = np.ones((nt, n, n))
        self.array_hist_W_2 = np.ones((nt, n, n))

        for i in range(nt-1):
            vec_v[i+1] = vec_v[i] + step * (
                                    0.04 * vec_v[i] * vec_v[i]
                                    + 5 * vec_v[i] + 140 - vec_u_izh[i]
                                    + np.ones(2) * vec_u[i]
                                )

            vec_u_izh[i+1] = vec_u_izh[i] + step * (
                                            self.param_a * (
                                                self.param_b * vec_v[i]
                                                - vec_u_izh[i]
                                                )
                                            )

            if np.all(vec_v[i+1] > self.izh_border):
                vec_v[i+1] = self.param_c
                vec_u_izh[i+1] = self.param_d

            delta = vec_est - vec_x

            vec_est[i+1] = vec_est[i] + step * (
                                        self.mat_A@vec_est[i]
                                        + self.mat_W_1@vec_v[i+1]
                                        + self.mat_W_2@vec_v[i+1]
                                        * vec_u[i]
                                    )

            self.mat_W_1 = self.mat_W_1 - step * (
                                            self.mat_K_1
                                            @ self.mat_P
                                            @ delta[i]
                                            @ vec_v[i+1]
                                        )

            self.mat_W_2 = self.mat_W_2 - step * (
                                            self.mat_K_2
                                            @ self.mat_P
                                            @ delta[i]
                                            @ vec_v[i+1]
                                            * vec_u[i]
                                        )

            self.array_hist_W_1[i] = self.mat_W_1.copy()
            self.array_hist_W_2[i] = self.mat_W_2.copy()

        return vec_est

    def predict(self,
                init_state: np.ndarray,
                vec_u: np.ndarray,
                step: float = 0.01) -> np.ndarray:

        n = self.mat_dim
        nt = len(vec_u)
        vec_est = init_state * np.ones((nt, n))
        vec_v = self.param_e * np.ones((nt, n))
        vec_u_izh = np.zeros((nt, n))
        vec_u_izh[0] = self.param_b * vec_v[0]

        mat_W_1 = self.mat_W_1
        mat_W_2 = self.mat_W_2

        for i in range(nt-1):
            vec_v[i+1] = vec_v[i] + step * (
                                    0.04 * vec_v[i] * vec_v[i]
                                    + 5 * vec_v[i] + 140 - vec_u_izh[i]
                                    + np.ones(2) * vec_u[i]
                                )

            vec_u_izh[i+1] = vec_u_izh[i] + step * (
                                            self.param_a * (
                                                    self.param_b * vec_v[i]
                                                    - vec_u_izh[i]
                                                )
                                            )

            if np.all(vec_v[i+1] > self.izh_border):
                vec_v[i+1] = self.param_c
                vec_u_izh[i+1] = self.param_d

            vec_est[i+1] = vec_est[i] + step * (
                                    self.mat_A@vec_est[i]
                                    + mat_W_1@vec_v[i+1]
                                    + mat_W_2@vec_v[i+1]
                                    * vec_u[i]
                                )

        return vec_est
