import os
import mne
import bs4
import requests
import numpy as np

from pathlib import Path
from torchvision.datasets.utils import download_url


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
