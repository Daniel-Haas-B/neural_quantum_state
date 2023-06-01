import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

import plot_utils
import cpp_utils
import seaborn as sns
import time

import matplotlib as mpl

cmap = plot_utils.cmap
from matplotlib.lines import Line2D

from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes


import plotly.graph_objects as go


def plot_part1(
    filename='Part1', particles=1, dims=1, interacting=False, save=False
):

    importance = False
    optimizerType = 'vanillaGD'

    Ns = np.array([particles])
    hiddenNodes = np.array([2, 4, 6, 8, 10])
    Ds = np.array([dims])

    analytical_solutions = [0.5 * D * N for D in Ds for N in Ns]

    if interacting:
        assert particles == 2
        assert dims == 2
        analytical_solutions = [3]

    lrs = np.array([0.001, 0.01, 0.1, 1])
    if particles == 2:
        lrs[-1] = 0.5

    logMet = 16   # 2 ^ 17 = 131072
    logEq = 14   # 10 % of logMet is 2^18 * 0.1 = 26214, while 2^15 = 32768

    total = len(Ns) * len(lrs) * len(hiddenNodes) * len(Ds)
    # start time measurement
    start = time.time()
    filename += f'_N={Ns[0]}_D={Ds[0]}'
    # if not cpp_utils.dataPath(filename + ".txt").exists():
    #     for i, N in enumerate(Ns):
    #         for j, D in enumerate(Ds):
    #                 for k, hn in enumerate(hiddenNodes):
    #                         for l, lr in enumerate(lrs):
    #                             cpp_utils.nqsRun(
    #                                 D=D, N=N, hiddenNodes=hn, logMet=logMet, logEq=logEq,
    #                                 stepLength=0.6, importance=importance, optimizerType=optimizerType,
    #                                 learnRate=lr, interacting=interacting, filename=filename, detailed=False
    #                             )
    #                             # time elapsed
    #                             time_elapsed = time.time() - start
    #                             # nice progress bar that updates in place printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r", time_elapsed = 0)
    #                             plot_utils.printProgressBar(i*len(lrs)*len(hiddenNodes)*len(Ds) + j*len(lrs)*len(hiddenNodes) + k*len(lrs) + l + 1, total, prefix = 'Progress:', suffix = 'Complete', length = 50, time_elapsed = time_elapsed)

    df_detailed = cpp_utils.nqsLoad(filename='detailed_' + filename + '.txt')
    df_detailed_2hidden = df_detailed[df_detailed['Hidden-nodes'] == 2]

    # two plots, sharing x axis, one for energy and one for variance
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={'hspace': 0.1}
    )

    # plot energy
    ax1 = sns.lineplot(
        data=df_detailed_2hidden,
        x='Epoch',
        y='Energy',
        hue='LearningRate',
        legend=True,
        palette='pastel',
        ax=ax1,
    )
    ax1.set(xlabel=r'Epoch', ylabel=r'$\langle E_L \rangle$')
    # ax1.set_title(f"{Ns[0]} particles, {Ds[0]} dimensions", fontsize=11)

    # ## add aesthetic error region
    for i, lr in enumerate(lrs):
        df_lr = df_detailed_2hidden[df_detailed_2hidden.LearningRate == lr]
        E, E_std, epoch, MCCs = (
            df_lr['Energy'].to_numpy(),
            df_lr['Energy_std'].to_numpy(),
            df_lr['Epoch'].to_numpy(),
            df_lr['Metro-steps'].to_numpy(),
        )
        E_std = np.sqrt(E_std) / (MCCs)
        # fill between
        ax1.fill_between(
            epoch, E - 2 * E_std, E + 2 * E_std, color=cmap(Ds[0]), alpha=0.2
        )

    df_detailed_2hidden['Energy_std'] = (
        df_detailed_2hidden['Energy_std'] / df_detailed_2hidden['Metro-steps']
    )

    # plot variance
    ax2 = sns.lineplot(
        data=df_detailed_2hidden,
        x='Epoch',
        y='Energy_std',
        hue='LearningRate',
        legend=True,
        palette='pastel',
        ax=ax2,
    )
    ax2.set(xlabel=r'Epoch', ylabel=r'$\sigma$')
    ax1.legend().remove()
    # make y axis scientific notation
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # ax1.set_title(f"{Ns[0]} particle, {Ds[0]} dimensions", fontsize=11)

    ax1.axhline(
        y=analytical_solutions[0],
        color=cmap(Ds[0]),
        linestyle='--',
        label='Analytical solution',
    )

    # add same legend for both plots, in between them
    handles, labels = ax2.get_legend_handles_labels()
    # add legend on top of the plot
    ax2.legend(
        handles=handles,
        labels=labels,
        title='Learning rate',
        fontsize=11,
        loc='upper center',
        bbox_to_anchor=(0.5, 2.4),
        ncol=4,
    )
    ax2.get_legend().get_frame().set_facecolor('white')
    # add analytical solution

    # make the background of the plot white but add grid lines
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax2.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    if save:
        plot_utils.save(filename.replace('.txt', f'_plot'))
    plt.show()

    df_detailed = df_detailed[df_detailed.Epoch == df_detailed.Epoch.max()]
    # find the index of the energy closest to the analytical solution
    idx = np.argmin(np.abs(df_detailed['Energy'] - analytical_solutions[0]))
    print(
        f"Energy closest to analytical solution: {df_detailed['Energy'].iloc[idx]}"
    )

    df_detailed['transformEnergy'] = np.log(
        np.abs(df_detailed['Energy'] - analytical_solutions[0])
    )

    # plot the heatmap
    ax = sns.heatmap(
        df_detailed.pivot('Hidden-nodes', 'LearningRate', 'transformEnergy'),
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        cmap='Purples',
    )
    # lable the colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label(r'$\log(|\langle E_L \rangle - E_{exact}|)$')

    # save the figure
    if save:
        filename += f'_heatmap'
        plot_utils.save(filename.replace('.txt', f'_plot'))
    plt.show()

    ### investigate logMet
    logMets = np.array([12, 13, 14, 15, 16, 17, 18, 19])   # 2 ^ 17 = 131072
    logEq = 14   # 2^12 = 4096
    hiddenNodes = np.array([6])

    total = len(Ns) * len(lrs) * len(hiddenNodes) * len(Ds)
    # start time measurement
    filename = 'Part1_mc_investigation'
    filename += f'_N={Ns[0]}_D={Ds[0]}'
    # start = time.time()
    # if not cpp_utils.dataPath(filename + ".txt").exists():

    #     for i, N in enumerate(Ns):
    #         for j, D in enumerate(Ds):
    #                 for k, hn in enumerate(hiddenNodes):
    #                         for l, logMet in enumerate(logMets):
    #                             for lr_count, lr in enumerate(lrs):
    #                                 cpp_utils.nqsRun(
    #                                     D=D, N=N, hiddenNodes=hn, logMet=logMet, logEq=logEq,
    #                                     stepLength=0.6, importance=importance, optimizerType=optimizerType,
    #                                     learnRate=lr, interacting=interacting, filename=filename, detailed=False
    #                                 )
    #                                 # time elapsed
    #                                 time_elapsed = time.time() - start
    #                                 # nice progress bar that updates in place printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r", time_elapsed = 0)
    #                                 plot_utils.printProgressBar(i*len(lrs)*len(hiddenNodes)*len(Ds)*len(logMets) + j*len(lrs)*len(hiddenNodes)*len(logMets) + k*len(lrs)*len(logMets) + l*len(lrs) + lr_count + 1, total, prefix = 'Progress:', suffix = 'Complete', length = 50, time_elapsed = time_elapsed)

    df_detailed = cpp_utils.nqsLoad(filename='detailed_' + filename + '.txt')

    df_detailed = df_detailed[df_detailed.Epoch == df_detailed.Epoch.max()]

    # print closest energy to analytical solution
    print(
        f"Energy closest to analytical solution: {df_detailed[['Energy', 'Metro-steps']].iloc[np.argmin(np.abs(df_detailed['Energy'] - analytical_solutions[0]))]}"
    )

    df_detailed['Energy'] = np.abs(
        df_detailed['Energy'] - analytical_solutions[0]
    )

    ax = sns.lineplot(
        data=df_detailed,
        x='Metro-steps',
        y='Energy',
        hue='LearningRate',
        legend=True,
        palette='pastel',
    )
    ax.get_legend().remove()
    ax.set(
        xlabel=r'Metro-steps',
        ylabel=r'$\log(|\langle E_L \rangle - E_{exact}|)$',
    )
    # set x axis to power of 2
    ax.set_xscale('log', base=2)

    # set y scale to log
    ax.set_yscale('log')

    # add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles, labels=labels, title='Learning rate', fontsize=11
    )
    # make background white
    ax.set_facecolor('white')
    # add grid lines
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    # add title
    # ax.set_title(f"{Ns[0]} particles, {Ds[0]} dimensions", fontsize=11)

    if save:
        plot_utils.save(filename.replace('.txt', f'_plot'))

    plt.show()


####################################
def plot_part2(filename='Part2', save=False, interacting=False):

    interacting = interacting
    importances = [False, True]
    optimizerType = 'vanillaGD'
    stepLengths = [0.3, 0.6, 0.9]

    N = 2
    hiddenNodes = 6   ## It was the best one
    D = 2
    lr = 0.5   #  It was the best one
    analytical_solution = 0.5 * D * N

    logMet = 16   # 2 ^ 16 = 65536
    logEq = 14   # 10 % of logMet is 2^18 * 0.1 = 26214, while 2^15 = 32768

    # start time measurement
    filename += f'_N={N}_D={D}'
    if not cpp_utils.dataPath(filename + '.txt').exists():
        for i, importance in enumerate(importances):
            for j, stepLength in enumerate(stepLengths):
                cpp_utils.nqsRun(
                    D=D,
                    N=N,
                    hiddenNodes=hiddenNodes,
                    logMet=logMet,
                    logEq=logEq,
                    stepLength=stepLength,
                    importance=importance,
                    optimizerType=optimizerType,
                    learnRate=lr,
                    interacting=interacting,
                    filename=filename,
                    detailed=False,
                )

    df_detailed = cpp_utils.nqsLoad(filename='detailed_' + filename + '.txt')

    # print df when epoch = max
    print(
        df_detailed[df_detailed.Epoch == 99]['Imposampling'],
        df_detailed[df_detailed.Epoch == 99]['Energy'],
        df_detailed[df_detailed.Epoch == 99]['StepLength'],
        df_detailed[df_detailed.Epoch == 99]['Energy_std'],
    )

    df_detailed = df_detailed[
        (df_detailed.Epoch > 0) & (df_detailed.Epoch < 30)
    ]

    # change Imposampling column to be "Metropolis" and "Metropolis-Hastings"
    df_detailed['Imposampling'] = df_detailed['Imposampling'].replace(
        {0: 'Metropolis', 1: 'Metropolis-Hastings'}
    )

    # to show that the importance sampling is better, we plot the energy per epoch for both cases
    df_detailed['transformEnergy'] = np.log(
        np.abs(df_detailed['Energy'] - analytical_solution)
    )
    # take the moving average of the energy per epoch, to smooth the plot with a window of 2
    df_detailed['transformEnergy'] = (
        df_detailed['transformEnergy'].rolling(window=4).mean()
    )

    ax = sns.lineplot(
        data=df_detailed,
        x='Epoch',
        y='transformEnergy',
        hue='StepLength',
        style='Imposampling',
        legend=True,
        palette='pastel',
    )
    ax.get_legend().remove()
    ax.set(
        xlabel=r'Epoch', ylabel=r'$\log(|\langle E_L \rangle - E_{exact}|)$'
    )

    # highlight in another color the region of y < -10.5
    # ax.axhspan(-10.5, -12.8, facecolor='0.02', alpha=0.12, color="purple")

    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles, labels=labels, title="", fontsize=11)

    # make background white
    ax.set_facecolor('white')
    # add grid lines
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.legend(title='', fontsize=11, loc='upper center', ncol=4)
    # remove "imposampling" from the legend
    ax.get_legend().get_texts()[4].set_text('Sampling')
    ax.get_legend().get_texts()[5].set_text('M')
    ax.get_legend().get_texts()[6].set_text('MH')

    # ax.set_title(f"{N} particles, {D} dimensions", fontsize=11)

    # create a tiny plot in the same plot that is just a zoom in the region between epoch 60 and 100

    # axins = inset_axes(ax, width="30%", height="30%", loc="center")
    # # add the data to the tiny plot

    # axins = sns.lineplot(data=df_detailed, x="Epoch", y="Energy", hue="StepLength", style="Imposampling", legend=False, palette="pastel")

    # remove labels
    # axins.set(xlabel=None, ylabel=None)
    # # add the connection lines to show the zoom region, to add dashed lines, use linestyle="--"
    # mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5", linestyle="--") # loc1 and loc2 are the corners of the zoom box, fc is the face color and ec is the edge color ,
    #                                                             # ec = 0.5 is gray

    # axins.set_xlim(0, 1)
    # axins.set_ylim(2.0, 2.04)

    # axins.set_xticks([0,1])

    if save:
        plot_utils.save(filename.replace('.txt', f'_plot'))
    plt.show()

    # plot acceptance rate per epoch
    ax = sns.lineplot(
        data=df_detailed,
        x='Epoch',
        y='Accept_ratio',
        hue='StepLength',
        style='Imposampling',
        legend=True,
        palette='pastel',
    )
    ax.get_legend().remove()
    ax.set(xlabel=r'Epoch', ylabel=r'Acceptance rate')
    # add legend
    handles, labels = ax.get_legend_handles_labels()
    # make labels be Metropolis and Metropolis-Hastings
    ax.legend(handles=handles, labels=labels, title='', fontsize=11)

    # change the name imposampling to sampling
    ax.get_legend().get_texts()[4].set_text('Sampling')
    ax.get_legend().get_texts()[5].set_text('M')
    ax.get_legend().get_texts()[6].set_text('MH')

    # make background white
    ax.set_facecolor('white')
    # add grid lines
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    if save:
        filename += f'_acceptance_rate'
        plot_utils.save(filename.replace('.txt', f'_plot_acceptance_rate'))
    plt.show()


####################################


def plot_part3(filename='Part3', save=False):

    interacting = True
    importance = True
    optimizerTypes = ['adamGD', 'vanillaGD', 'momentumGD', 'rmspropGD']

    N = 2
    hiddenNodes = np.array(
        [2]
    )   # if you loose the nice plot, turn this back to 2
    D = 2

    analytical_solution = 3
    lrs = np.array([0.1])   # next try 0.1

    logMet = 17   # 2 ^ 16 = 65536
    logEq = 14   # 10 % of logMet is 2^18 * 0.1 = 26214, while 2^15 = 32768

    total = len(hiddenNodes) * len(lrs) * len(optimizerTypes)
    # start time measurement
    start = time.time()
    filename += (
        f'_N={N}_D={D}_interacting={interacting}_importance={importance}'
    )
    # if not cpp_utils.dataPath(filename + ".txt").exists():

    #     for i, optimizerType in enumerate(optimizerTypes):
    #         for k, hn in enumerate(hiddenNodes):
    #                 for l, lr in enumerate(lrs):
    #                     cpp_utils.nqsRun(
    #                         D=D, N=N, hiddenNodes=hn, logMet=logMet, logEq=logEq,
    #                         stepLength=0.6, importance=importance, optimizerType=optimizerType,
    #                         learnRate=lr, interacting=interacting, filename=filename, detailed=False
    #                     )
    #                     # time elapsed
    #                     time_elapsed = time.time() - start
    #                     # nice progress bar that updates in place printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r", time_elapsed = 0)
    #                     plot_utils.printProgressBar(i*len(lrs)*len(hiddenNodes) + k*len(lrs) + l + 1, total, prefix = 'Progress:', suffix = 'Complete', length = 50, time_elapsed = time_elapsed)

    df_detailed = cpp_utils.nqsLoad(filename='detailed_' + filename + '.txt')
    df_detailed = df_detailed[df_detailed.Epoch < 400]
    # show only every other epoch
    df_detailed = df_detailed[df_detailed.Epoch % 5 == 0]
    # take moving average of energy per epoch, window of 10
    # df_detailed["Energy"] = df_detailed["Energy"].rolling(window=20).mean()

    ax = sns.lineplot(
        data=df_detailed,
        x='Epoch',
        y='Energy',
        hue='optimizerType',
        legend=True,
        palette='pastel',
    )

    # make background white
    ax.set_facecolor('white')
    # add grid lines
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    ax.get_legend().remove()
    ax.set(xlabel=r'Epoch', ylabel=r'$\langle E_L \rangle $')
    # ax.set_title(f"{N} particles, {D} dimensions, interacting = {interacting}", fontsize=11)

    # ## add analytical solution
    ax.axhline(
        y=analytical_solution,
        color=cmap(D),
        linestyle='--',
        label=f'Analytical',
    )

    # ## add aesthetic error region
    for i, opt in enumerate(optimizerTypes):
        df_opt = df_detailed[df_detailed.optimizerType == opt]
        E, E_std, epoch, MCCs = (
            df_opt['Energy'].to_numpy(),
            df_opt['Energy_std'].to_numpy(),
            df_opt['Epoch'].to_numpy(),
            df_opt['Metro-steps'].to_numpy(),
        )
        E_std = np.sqrt(E_std) / (MCCs)
        # fill between
        ax.fill_between(
            epoch, E - 2 * E_std, E + 2 * E_std, alpha=0.2, color=cmap(i)
        )

    # # add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='Optimizer', fontsize=11)

    if save:
        plot_utils.save(filename.replace('.txt', f'_plot'))
    plt.show()


def plot_part3_adam_investigation(
    filename='Part3_adam_investigation', save=False
):

    interacting = True
    importance = True
    analytical_solution = 3

    # grid search over beta1, beta2, epsilon
    beta1 = np.array([0.7, 0.8, 0.9, 0.99, 0.999])
    beta2 = np.array([0.8, 0.9, 0.99, 0.999])
    epsilon = np.array([1e-9, 1e-8, 1e-7, 1e-6])

    total = len(beta1) * len(beta2) * len(epsilon)
    # start time measurement
    start = time.time()

    df_concat = pd.DataFrame()

    for i, b1 in enumerate(beta1):
        for k, b2 in enumerate(beta2):
            for l, eps in enumerate(epsilon):
                filename = 'Part3_adam_investigation'
                filename = f'{filename}_beta1={b1}_beta2={b2}_epsilon={eps}'
                # if not cpp_utils.dataPath(filename + ".txt").exists():
                #     cpp_utils.nqsRunAdamopt(
                #         hiddenNodes=2, logMet=16, logEq=14, importance=True, beta1=b1, beta2=b2, epsilon=eps,
                #                     learnRate=0.1, filename=filename
                #         ) #lr 0.1 is golden
                #     # time elapsed
                #     time_elapsed = time.time() - start
                #     # nice progress bar that updates in place printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r", time_elapsed = 0)
                #     plot_utils.printProgressBar(i*len(beta2)*len(epsilon) + k*len(beta2) + l + 1, total, prefix = 'Progress:', suffix = 'Complete', length = 50, time_elapsed = time_elapsed)

                df_detailed = cpp_utils.nqsLoad(
                    filename='detailed_' + filename + '.txt'
                )
                df_detailed['beta1'] = b1
                df_detailed['beta2'] = b2
                df_detailed['epsilon'] = eps

                df_concat = pd.concat([df_concat, df_detailed])

    # df_concat = df_concat[df_concat.epsilon == 1e-9]
    # df_concat = df_concat[df_concat["Energy"] - analytical_solution < 0.5]

    # print energy closest to analytical solution
    # min_energy = df_concat["Energy"].min()
    # df_best = df_concat[df_concat["Energy"] == min_energy]

    df_concat = df_concat[df_concat.Epoch == 600]

    print(df_concat.columns)
    # remove columns except for beta1, beta2, epsilon, Energy
    df_concat = df_concat[['beta1', 'beta2', 'epsilon', 'Energy']]
    # drop nan
    df_concat = df_concat.dropna()
    # drop energies below analytical solution
    df_concat = df_concat[df_concat.Energy > 2.9]
    # drop energies above 6
    df_concat = df_concat[df_concat.Energy < 10]

    # reverse sort by energy
    df_concat = df_concat.sort_values(by='Energy', ascending=False)

    # make df_concat["epsilon"]
    df_concat['epsilon'] = df_concat['epsilon'].apply(
        lambda x: np.abs(np.log10(x) / 10)
    )
    print(df_concat['epsilon'])

    # make axis limits be the same
    # plt.xlim(0, 1)
    plt.ylim(0.55, 1.05)

    # rename columns
    df_concat = df_concat.rename(
        columns={
            'beta1': r'$\beta_1$',
            'beta2': r'$\beta_2$',
            'epsilon': r'$|\log_{10}(\epsilon)|/10$',
        }
    )

    # create normalized colormap
    norm = mpl.colors.Normalize(
        vmin=df_concat['Energy'].min(), vmax=df_concat['Energy'].max()
    )

    cmap = mpl.cm.ScalarMappable(norm=norm, cmap='magma_r')
    cmap.set_array([])

    pd.plotting.parallel_coordinates(
        df_concat, 'Energy', color=cmap.to_rgba(df_concat['Energy'])
    )
    # make colorbar on bottom

    plt.gca().lines[0].set_linewidth(
        5
    )   # lines[0] is the first line, lines[1] is the second line, etc
    plt.gca().lines[1].set_linewidth(4)
    plt.gca().lines[2].set_linewidth(3)
    plt.gca().lines[3].set_linewidth(2)
    plt.gca().lines[4].set_linewidth(1)
    plt.gca().lines[5].set_linewidth(0.5)

    # print the best values
    print(df_concat.iloc[-1])

    # adjust the scale of each parallel axis

    # normalize the scale of each axis
    # remove legend
    plt.gca().get_legend().remove()
    # plt.gca().set_ylim(0, 1)
    # add colorbar on top of the graph but horizontally
    cbar = plt.colorbar(cmap, orientation='horizontal', pad=0.1)
    cbar.set_label(r'$\langle E_L \rangle$')

    # save as pdf
    if save:
        plot_utils.save(filename.replace('.txt', f'_plot'))

    plt.show()


def plot_optimal_energy_per_epoch(
    filename='adam_opt_GD_energy_per_epoch',
    D=2,
    interacting=True,
    importance=True,
    save=False,
):
    Ns = np.array([2])
    hiddenNodes = np.array([2])

    lr = np.array([0.1])
    logMet = 16   # 2 ^ 16 = 65536
    logEq = 14   # 10 % of logMet is 2^18 * 0.1 = 26214, while 2^15 = 32768

    opts = ['adamGD']
    # start time measurement
    start = time.time()

    if not cpp_utils.dataPath(filename + '.txt').exists():
        for i, N in enumerate(Ns):
            for k, opt in enumerate(opts):
                info = f'N={N}_D={D}_met={logMet}_interacting={interacting}_opt={opt}_importance={importance}'
                filename += info
                # if not cpp_utils.dataPath(filename + ".txt").exists():
                #     for h, hn in enumerate(hiddenNodes):

                #         #total = len(Ns)*len(alphas)
                #             for j, l in enumerate(lr):
                # print(f"Python Running for N = {N} and lr = {l}...")
                # cpp_utils.nqsRun(D=D, N=N, hiddenNodes=hn, logMet=logMet, logEq=logEq, stepLength=0.6, importance=importance, optimizerType=opt, learnRate=l, interacting=interacting, filename=filename, detailed=False)
                # print(f"Python run done for N = {N} and lr = {l}...")

    df_detailed = cpp_utils.nqsLoad(filename='detailed_' + filename + '.txt')
    df_detailed = df_detailed[df_detailed.Epoch <= 600]
    #
    min_energy = df_detailed['Energy'].min()

    # print where it happens
    print(f'Minimum energy: {min_energy}')
    df_best = df_detailed[df_detailed['Energy'] == min_energy]
    print(df_best)

    # set palette to pastel
    sns.set_palette('pastel')
    ax = sns.lineplot(data=df_detailed, x='Epoch', y='Energy', legend=True)

    ax.set(xlabel=r'Epoch', ylabel=r'$\langle E_L \rangle$')

    # add analytical solution
    ax.axhline(y=3, color=cmap(D), linestyle='--', label=f'Analytical')
    # add errorbar df_detailed["Energy_std"]/np.sqrt(df_detailed["Metro-steps"]) as shaded region
    ax.fill_between(
        df_detailed['Epoch'],
        df_detailed['Energy']
        - df_detailed['Energy_std'] / np.sqrt(df_detailed['Metro-steps']),
        df_detailed['Energy']
        + df_detailed['Energy_std'] / np.sqrt(df_detailed['Metro-steps']),
        alpha=0.2,
        color='purple',
    )
    # get pastel palette first color
    # cmap = sns.color_palette("pastel", as_cmap=True)

    # add legend and include shaded region in the legend
    handles, labels = ax.get_legend_handles_labels()
    labels = [
        'Analytical',
        'Energy optimal model',
        r'$\pm \sigma/\sqrt{MCCs}$',
    ]
    handles = [
        mpl.lines.Line2D(
            [], [], color=cmap(D), linestyle='--', label='Analytical'
        ),
        mpl.lines.Line2D([], [], label='Energy optimal model'),
        mpl.patches.Patch(
            facecolor='purple', alpha=0.2, label=r'$\pm \sigma/\sqrt{M}$'
        ),
    ]
    ax.legend(handles=handles, labels=labels, title='', fontsize=11)

    # make background white
    ax.set_facecolor('white')
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    if save:
        plot_utils.save(filename.replace('.txt', f'_plot'))
    plt.show()

    # plot also the Energy_std
    df_detailed['Energy_std'] = (
        df_detailed['Energy_std'] / df_detailed['Metro-steps']
    )
    # eliminate largest value
    df_detailed = df_detailed[
        df_detailed.Energy_std != df_detailed.Energy_std.max()
    ]
    ax = sns.lineplot(data=df_detailed, x='Epoch', y='Energy_std', legend=True)
    ax.set(xlabel=r'Epoch', ylabel=r'$\sigma$')
    # make y axis scientific notation
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # ax.set_title(f"{Ns[0]} particle, {Ds[0]} dimensions", fontsize=11)
    # get handles and labels
    handles, labels = ax.get_legend_handles_labels()
    labels = [r'$\sigma$ optimal model']
    handles = [mpl.lines.Line2D([], [], label='$\sigma$ optimal model')]
    ax.legend(handles=handles, labels=labels, title='', fontsize=11)

    # make background white
    ax.set_facecolor('white')
    # add grid lines
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    if save:
        filename += f'_variance'
        plot_utils.save(filename.replace('.txt', f'_plot_variance'))
    plt.show()

    # ax.legend(ncol=4, bbox_to_anchor=(1.05, 1.15))
    # ax.set(xlabel=r"Epoch", ylabel=r"$\langle E_L \rangle [\hbar \omega]$")

    df_HiddenGradNorm = df_detailed[['Epoch', 'HiddenGradNorm']]
    df_HiddenGradNorm['grad'] = 'Hidden bias GradNorm'
    df_VisibleGradNorm = df_detailed[['Epoch', 'VisibleGradNorm']]
    df_VisibleGradNorm['grad'] = 'Visible bias GradNorm'
    df_WeightGradNorm = df_detailed[['Epoch', 'WeightGradNorm']]
    df_WeightGradNorm['grad'] = 'Weights GradNorm'

    # concat the dataframes
    df_gradNorm = pd.concat(
        [df_HiddenGradNorm, df_VisibleGradNorm, df_WeightGradNorm]
    )
    # take moving average of energy per epoch, window of 10
    df_gradNorm['HiddenGradNorm'] = (
        df_gradNorm['HiddenGradNorm'].rolling(window=10).mean()
    )

    # print energy per epoch and use grad as hue
    ax = sns.lineplot(
        data=df_gradNorm,
        x='Epoch',
        y='HiddenGradNorm',
        hue='grad',
        legend=True,
        palette='pastel',
    )
    ax.set(xlabel=r'Epoch', ylabel=r'Gradient norm')
    # add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='Gradient')

    if save:
        filename += f'_gradNorm'
        plot_utils.save(filename.replace('.txt', f'_plot'))
    plt.show()


if __name__ == '__main__':

    # Part 1 energy of 1 and 2 particles in 1D and 2D. Experiment with different learning rates and number of hidden nodes
    plot_part1(filename="Part1", particles=1, dims=1, save=True)
    plot_part1(filename="Part1", particles=2, dims=2, save=True)

    ## From now on we only use 2 particles
    # Part 2 investigate the importance sampling effect
    plot_part2(filename="Part2", save=True)

    # Interaction Investigation
    plot_part1(filename="Part3_lr_D_investigation", particles=2, dims=2, interacting=True, save=True)
    plot_part2(filename='Part3_AR', interacting=False, save=True)
    plot_optimal_energy_per_epoch(filename="Optimal", D=2, interacting=True, optimizerType = "vanillaGD", importance=True, save=False)

    # Part 3 add interaction, use learning rate 0.5 and 2 hidden nodes, around 2^15 metro steps and test various optimizers

    plot_part3(filename="Part3_optimizer_investigation", save=True)
    plot_part3_adam_investigation(filename="Part3_adam_investigation", save=True)


