import os
import mne
import bs4
import requests
import numpy as np
import pandas as pd

from pathlib import Path
from scipy import integrate
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R
from torchvision.datasets.utils import download_url

import matplotlib.pyplot as plt


class PhysioNet(object):

    def __init__(self):
        self.template_url = 'https://physionet.org/files/{0}/1.0.0/'
        self.template_root = './data/raw/{0}/'
        self.downloaded_limit = 400

    @staticmethod
    def calc_disk_usage(path: str) -> float:

        disk_usage = sum(
                            d.stat().st_size for d
                            in os.scandir(path)
                            if d.is_file()
                        ) / np.power(1024, 2)

        return disk_usage

    def recur_load(self, url: str, root: str) -> bool:

        req = requests.get(url)
        parser = bs4.BeautifulSoup(req.text, 'html.parser')

        for file_link in parser.find_all('a'):
            if 'PSG.edf' in file_link['href']:
                dataset_url = url + file_link['href']
                disck_uasege = self.calc_disk_usage(root)

                if disck_uasege < self.downloaded_limit:
                    download_url(dataset_url, root, None)
                else:
                    return False

            if '/' in file_link['href'] and '..' not in file_link['href']:
                dataset_url = url + file_link['href']
                self.recur_load(dataset_url, root)

        return True

    def load(self, dataset_name: str) -> bool:

        url = self.template_url.format(dataset_name)
        root = self.template_root.format(dataset_name)

        os.makedirs(root, exist_ok=True)
        res = self.recur_load(url, root)

        return res


class ExpData(object):

    def __init__(self,
                 file_name: str,
                 raw_data_folder: str = './data/raw/',
                 prep_data_folder: str = './data/prep/',
                 columns: list = ['Frequency', 'Amplitude',
                                  'LeftYaw', 'RightYaw', 'RotAngleZ',
                                  'RotAngleY', 'RotAngleX'],
                 prepproc_maper: dict = {
                     'coor_as_angles': (['DirX', 'DirY', 'DirZ'],
                                        ['Yaw', 'Pitch'],
                                        ['Left', 'Right']),
                     'rot_as_angles': (['HeadRotW', 'HeadRotX',
                                       'HeadRotY', 'HeadRotZ'],
                                       ['RotAngleZ', 'RotAngleY',
                                        'RotAngleX'])
                 }):
        self.raw_data_folder = raw_data_folder
        self.prep_data_folder = prep_data_folder
        self.file_name = file_name
        self.prepproc_maper = prepproc_maper
        self.columns = columns
        self.data = None
        self.keys = None

    @staticmethod
    def coor_as_angles(X: np.array, Y: np.array, Z: np.array) -> np.ndarray:
        yaw = -1 * np.arctan2(X, Z)
        pitch = np.arctan2(Y, np.sqrt((X * X) + (Z * Z)))
        return yaw, pitch

    @staticmethod
    def rot_as_angles(rot: np.array) -> np.ndarray:
        r = R.from_quat(rot)
        angles = r.as_euler('zyx', degrees=False)
        return angles

    @staticmethod
    def get_keys(data: pd.DataFrame,
                 tcols: list = ['Frequency', 'Amplitude']) -> np.ndarray:
        tcols = tcols
        return data[tcols].groupby(by=tcols)\
                          .sum()\
                          .reset_index()\
                          .to_numpy()

    @staticmethod
    def get_columns(data: pd.DataFrame) -> np.ndarray:
        return data.columns.values

    def open(self, sep: str = ' ') -> pd.DataFrame:
        full_file_name = self.raw_data_folder + self.file_name
        return pd.read_csv(full_file_name, sep=sep)

    def save(self, sep: str = ',') -> None:
        full_file_name = self.prep_data_folder + self.file_name
        self.data.to_csv(full_file_name, sep=sep, index=False)

    def prep_file_exists(self) -> bool:
        full_file_name = self.prep_data_folder + self.file_name
        return os.path.isfile(full_file_name)

    def set_eye_angles(self, data: pd.DataFrame) -> pd.DataFrame:
        cols, feats, prefixes = self.prepproc_maper['coor_as_angles']

        for prefix in prefixes:
            tcols = list(map(lambda col: prefix + col, cols))
            tfeats = list(map(lambda feat: prefix + feat, feats))
            yaw, pitch = self.coor_as_angles(*data[tcols].values.T)
            data[tfeats[0]], data[tfeats[1]] = yaw.copy(), pitch.copy()

        return data

    def set_head_angles(self, data: pd.DataFrame) -> pd.DataFrame:
        tcols, tfeats = self.prepproc_maper['rot_as_angles']
        angles = self.rot_as_angles(data[tcols].values.reshape(-1, 4))
        for i, tfeat in enumerate(tfeats):
            data[tfeat] = gaussian_filter(angles[:, i], 2)

        return data

    def prep(self,
             sep: str = ' ',
             start: int = 3,
             force_clean: bool = False) -> None:

        if self.prep_file_exists() and not force_clean:
            full_file_name = self.prep_data_folder + self.file_name
            data = pd.read_csv(full_file_name)
            self.data = data
            self.keys = self.get_keys(data)
            self.columns = self.get_columns(data)
        else:
            data = self.open(sep)
            data = data.iloc[start:]
            data = self.set_eye_angles(data)
            data = self.set_head_angles(data)
            data = data[self.columns].copy()

            self.data = data
            self.keys = self.get_keys(data)
            self.columns = self.get_columns(data)
            self.save()

    def get_data(self, index: int) -> pd.DataFrame:
        filter_query = 'Frequency == {0} & Amplitude == {1}'\
                                        .format(*self.keys[index])
        tmp_data = self.data.query(filter_query).copy()
        clean_perc = 0.5
        clean_index = int(len(tmp_data)*clean_perc)
        return tmp_data[clean_index:].copy()


class DynamicSystem(object):

    def __init__(self, dyn_sys_func):
        self.dyn_sys_func = dyn_sys_func
        self.dyn_values = None

    def integrate(self, init_val: np.array, time: np.array):

        dyn_values, infodict = integrate.odeint(self.dyn_sys_func,
                                                init_val,
                                                time,
                                                full_output=True)
        self.dyn_values = dyn_values

        return (dyn_values, infodict)

    def plot(self, time: np.array):

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        for dt_slice in self.dyn_values.T:
            plt.plot(time, dt_slice)

        plt.xlabel('time')
        plt.ylabel('states of system')
        plt.title('Evolution of dynamical system')
