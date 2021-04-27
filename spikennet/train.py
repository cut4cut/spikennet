import os
from utils.dataset import PhysioNet


if __name__ == '__main__':
    datasets_name = ['shhpsgdb', 'sleep-edfx']

    physionet = PhysioNet()
    physionet.load(datasets_name[1])
