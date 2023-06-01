import pathlib as pl
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler

import seaborn as sns


def make_figs_path(filename):
    cur_path = pl.Path(__file__)
    root_path = cur_path

    while root_path.name != 'neural_quantum_state':
        root_path = root_path.parent

    figs_path = root_path / pl.Path('Analysis/figs')

    if not figs_path.exists():
        return None
    if not filename.endswith('.pdf'):
        filename += '.pdf'

    figs_path /= filename

    return str(figs_path)


def save(filename):
    if filename:
        filename = make_figs_path(filename)
        plt.savefig(filename)


colors = [
    sns.color_palette('husl')[-3],
    sns.color_palette('husl')[-2],
    sns.color_palette('husl')[-1],
    'mediumorchid',
    sns.color_palette('deep')[-1],
    sns.color_palette('dark')[-1],
]

markers = ['s', 'o', 'X', '*', '^']

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    '', colors[:2] + [colors[3]]
)

cmap_terrain = matplotlib.colors.LinearSegmentedColormap.from_list(
    '',
    [
        'darkslategrey',
        sns.color_palette('dark')[-1],
        sns.color_palette('husl')[-3],
        # sns.color_palette('deep')[-1],
        'darkturquoise',
        sns.color_palette('husl')[-2],
        # 'mediumorchid',
        sns.color_palette('husl')[-1],
    ],
)

sns.set_style('darkgrid')
# Set all fonts to be equal to tex
# https://stackoverflow.com/questions/11367736/matplotlib-consistent-font-using-latex
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['text.usetex'] = True

# Saving parameters
plt.rcParams['savefig.dpi'] = 300

# Figure options, set tight layout
plt.rc('figure', autolayout=True)

# Font sizes
plt.rc('axes', titlesize=18, labelsize=16, prop_cycle=cycler('color', colors))
plt.rc('legend', fontsize=14, shadow=True)

# Tick parameters
_ticks_default_parameters = {'labelsize': 12}
plt.rc('xtick', **_ticks_default_parameters)
plt.rc('ytick', **_ticks_default_parameters)

# Line options
plt.rc('lines', linewidth=2)


# To see more paramteres, print the possible options:
# print(plt.rcParams)


def printProgressBar(
    iteration,
    total,
    prefix='',
    suffix='',
    decimals=1,
    length=100,
    fill='█',
    printEnd='\r',
    time_elapsed=0,
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        time_elapsed- Optional  : time elapsed (float)
    """
    percent = ('{0:.' + str(decimals) + 'f}').format(
        100 * (iteration) / float(total)
    )
    time_left = (time_elapsed / (iteration + 1)) * (total - iteration - 1)
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(
        f'\r{prefix} |{bar}| {percent}% {suffix} Time left: {time_left:.2f}s',
        end=printEnd,
    )
    # Print New Line on Complete
    if iteration == total:
        print()
