import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter


def plot_experiment(i: int,
                    time: np.ndarray,
                    split: int,
                    width: int,
                    tr_target: np.ndarray,
                    tr_control: np.ndarray,
                    vl_target: np.ndarray,
                    vl_control: np.ndarray,
                    tr_est: np.ndarray,
                    vl_pred: np.ndarray,
                    norms_W_1: np.ndarray,
                    norms_W_2: np.ndarray) -> None:

    fig, ax = plt.subplots(3, figsize=(18, 12))

    ax[0].plot(time[:split], tr_target[:, 0])
    ax[0].plot(time[split:], vl_target[:, 0])
    ax[0].plot(time[:split], tr_est[:, 0])
    ax[0].plot(time[split:], vl_pred[:, 0])
    ax[0].axvline(x=split, c='grey', linestyle='--')
    ax[0].legend(['train', 'valid',
                  'train predict', 'valid predict'])

    ax[1].plot(time[:split], tr_control)
    ax[1].plot(time[split:], vl_control)
    ax[1].axvline(x=split, c='grey', linestyle='--')
    ax[1].legend(['train control', 'valid control'])

    ax[2].plot(time[:split-1], norms_W_1[i])
    ax[2].plot(time[:split-1], norms_W_2[i])
    ax[2].plot(time[split:], np.ones((width-split)) * norms_W_1[i][-1])
    ax[2].plot(time[split:], np.ones((width-split)) * norms_W_2[i][-1])
    ax[2].axvline(x=split, c='grey', linestyle='--')
    ax[2].legend(['train frobenius norm W1', 'train frobenius norm W2',
                  'valid frobenius norm W1', 'valid frobenius norm W2'])

    fig.savefig('./report/fold_{}.png'.format(i))


def plot_article(i: int,
                 time: np.ndarray,
                 split: int,
                 tr_target: np.ndarray,
                 tr_control: np.ndarray,
                 tr_est: np.ndarray,
                 error: np.ndarray,
                 weaights_dyn: np.ndarray) -> None:

    fig, axs = plt.subplots(2, 2, figsize=(16, 8))

    axs[0, 0].plot(time[:split] / 110,
                   np.degrees(tr_control),
                   color='tab:blue',
                   lw='2')

    axs[1, 0].scatter(time[:split] / 110,
                      np.degrees(tr_target[:, 0]) + 2,
                      linestyle='solid',
                      color=(165/255, 172/255, 175/255),
                      s=4.5,
                      zorder=100)

    axs[1, 0].plot(time[:split] / 110,
                   np.degrees(tr_est[:, 0]) + 2,
                   color='tab:blue',
                   lw='2')

    axs[0, 1].plot(time[:split] / 110,
                   np.degrees(error),
                   color='tab:blue',
                   lw='2')

    axs[1, 1].plot(time[:split-2] / 110,
                   np.abs(weaights_dyn[0][:, 0][:split-2]),
                   color=(165/255, 172/255, 175/255),
                   lw='2')

    axs[1, 1].plot(time[:split-2] / 110,
                   np.abs(weaights_dyn[1][:, 1][:split-2]),
                   color='tab:blue',
                   lw='2')

    # axs[1, 1].yaxis.set_major_formatter(FormatStrFormatter('%.f'))
    # axs[1, 1].yaxis._useMathText = True
    # axs[1, 1].ticklabel_format(style='sci', axis='y')

    axs[0, 0].set_xlabel('Time (s)', fontsize=12)
    axs[0, 1].set_xlabel('Time (s)', fontsize=12)
    axs[1, 1].set_xlabel('Time (s)', fontsize=12)
    axs[1, 0].set_xlabel('Time (s)', fontsize=12)

    axs[0, 0].set_ylabel("Angle of head rotation (°)", fontsize=12)
    axs[1, 0].set_ylabel("Angle of eye rotation (°)", fontsize=12)
    axs[0, 1].set_ylabel('Error (°)', fontsize=12)
    axs[1, 1].set_ylabel('Dynamics of changes in weights', fontsize=12)

    # axs[0, 0].set_xticks(np.arange(len(time)))

    axs[1, 0].legend(['Identification', 'Experimental data'])
    axs[1, 1].legend(['W_1', 'W_2'])

    axs[0, 0].text(-0.1, 1., 'A', transform=axs[0, 0].transAxes,
                   size=20, weight='bold')
    axs[0, 1].text(-0.075, 1., 'B', transform=axs[0, 1].transAxes,
                   size=20, weight='bold')
    axs[1, 1].text(-0.075, 1., 'D', transform=axs[1, 1].transAxes,
                   size=20, weight='bold')
    axs[1, 0].text(-0.1, 1., 'C', transform=axs[1, 0].transAxes,
                   size=20, weight='bold')

    axs[1, 0].set_ylim([-2, 14])

    axs[0, 0].axvline(x=[49 / 110],
                      ymin=0.97,
                      ymax=0.994838,
                      c=(65/255, 65/255, 65/255),
                      linewidth=2,
                      zorder=0,
                      linestyle=(0, (5, 10)),
                      clip_on=False)

    axs[1, 0].axvline(x=[49 / 110],
                      ymin=0.0,
                      ymax=2.2,
                      c=(65/255, 65/255, 65/255),
                      linewidth=2,
                      zorder=100,
                      linestyle=(0, (5, 10)),
                      clip_on=False)

    axs[0, 0].axvline(x=[1290.8 / 110],
                      ymin=0.97,
                      ymax=0.994838,
                      c=(65/255, 65/255, 65/255),
                      linewidth=2,
                      zorder=0,
                      linestyle=(0, (5, 10)),
                      clip_on=False)

    axs[1, 0].axvline(x=[1290.8 / 110],
                      ymin=0.0,
                      ymax=2.2,
                      c=(65/255, 65/255, 65/255),
                      linewidth=2,
                      zorder=100,
                      linestyle=(0, (5, 10)),
                      clip_on=False)

    plt.tight_layout()
    plt.savefig('./report/articl_plot_{}.png'.format(i))
    plt.savefig('./report/articl_plot_{}.pdf'.format(i))
