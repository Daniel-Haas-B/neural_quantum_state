import subprocess
import pathlib as pl
import pandas as pd
import numpy as np

import os


def rootPath():
    cur_path = pl.Path(__file__)
    root_path = cur_path

    while root_path.name != 'neural_quantum_state':
        root_path = root_path.parent

    return root_path


def nqsPath():
    nqs_path = rootPath() / pl.Path('build/nqs')
    return nqs_path


def testsPath():
    tests_path = rootPath() / pl.Path('build/tests')
    return tests_path


def dataPath(filename):
    filename_path = rootPath() / pl.Path(f'Data/{filename}')
    return filename_path


def nqsRun(
    D=2,
    N=2,
    hiddenNodes=2,
    logMet=20,
    logEq=16,
    stepLength=0.6,
    importance=False,
    optimizerType='vanillaGD',
    learnRate=0.1,
    interacting=False,
    filename='test.txt',
    detailed=False,
):
    nqs_path = nqsPath()
    filename_path = dataPath(filename)

    assert (
        nqs_path.exists()
    ), f'I cannot find {nqs_path} :((, are you sure you have compiled?'
    args = [
        nqs_path,
        D,
        N,
        hiddenNodes,
        logMet,
        logEq,
        stepLength,
        int(importance),
        optimizerType,
        learnRate,
        int(interacting),
        filename_path,
        int(detailed),
    ]

    if not filename:
        args.pop()

    args_run = [str(arg) for arg in args]
    subprocess.run(args_run)


def nqsRunAdamopt(
    hiddenNodes=2,
    logMet=14,
    logEq=14,
    importance=True,
    beta1=0.9,
    beta2=0.99,
    epsilon=1e-8,
    learnRate=0.1,
    interacting=True,
    filename='adam_investigation',
    detailed=False,
):
    nqsadam_path = nqsPath()

    filename_path = dataPath(filename)

    assert (
        nqsadam_path.exists()
    ), f'I cannot find {nqsadam_path} :((, are you sure you have compiled?'
    args = [
        nqsadam_path,
        2,
        2,
        hiddenNodes,
        logMet,
        logEq,
        0.6,
        int(importance),
        'adamGD',
        learnRate,
        int(interacting),
        filename_path,
        int(detailed),
        beta1,
        beta2,
        epsilon,
    ]

    if not filename:
        args.pop()

    args_run = [str(arg) for arg in args]
    subprocess.run(args_run)


def nqsLoad(filename):
    print('DEBUG: Loading file', filename)
    filename_path = dataPath(filename)

    df = pd.read_csv(filename_path, delim_whitespace=True)

    int_cols = [
        'Dimensions',
        'Particles',
        'Hidden-nodes',
        'Metro-steps',
        'Imposampling',
        'Epoch',
        'Interaction',
    ]
    numeric_cols = [col for col in df.columns if col not in int_cols]
    numeric_cols.remove('optimizerType')
    for col in numeric_cols:
        df[col] = df[col].astype(float)

    return df


def binaryLoad(filename):
    filename = dataPath(filename)
    with open(filename, 'rb') as f:
        f.seek(0, 2)
        num_doubles = f.tell() // 8
        f.seek(0)

        return np.fromfile(f, dtype=np.float64, count=num_doubles)


def rLoad(filename):
    filename = dataPath(filename + '_Rs.txt')

    return pd.read_csv(filename, delim_whitespace=True)
