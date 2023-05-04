import subprocess
import pathlib as pl
import pandas as pd
import numpy as np

import os

def rootPath():
    cur_path = pl.Path(__file__)
    root_path = cur_path

    while root_path.name != "neural_quantum_state":
        root_path = root_path.parent

    return root_path

def nqsPath():
    nqs_path = rootPath() / pl.Path("build/nqs")
    return nqs_path

def testsPath():
    tests_path = rootPath() / pl.Path("build/tests")
    return tests_path

def dataPath(filename):
    filename_path = rootPath() / pl.Path(f"Data/{filename}")
    return filename_path


def nqsRun(D=2, N=2, hiddenNodes=2,  logMet=20, logEq=16, stepLength=0.6, importance=False, analytical=True, learnRate=0.1 ,interacting=False, filename="test.txt", detailed=False):
    nqs_path = nqsPath()
    filename_path = dataPath(filename)

    assert nqs_path.exists(), f"I cannot find {nqs_path} :((, are you sure you have compiled?"
    args = [
        nqs_path,
        D,
        N,
        hiddenNodes,
        logMet,
        logEq,
        stepLength,
        int(importance),
        int(analytical),
        learnRate,
        int(interacting),
        filename_path,
        int(detailed)
    ]

    if not filename:
        args.pop()

    args_run = [str(arg) for arg in args]
    subprocess.run(args_run)


def timingRun(D=3, N=10, logMet=6, logEq=5, omega=1.0, alpha=0.5, stepLength=0.1, analytical=True, filename="timing.txt"):
    timing_path = timingPath()
    filename_path = dataPath(filename)

    assert timing_path.exists(), f"I cannot find {timing_path} :((, are you sure you have compiled?"
    args = [
        timing_path,
        D,
        N,
        logMet,
        logEq,
        omega,
        alpha,
        stepLength,
        int(analytical),
        filename_path,
    ]

    if not filename:
        args.pop()

    args_run = [str(arg) for arg in args]

    subprocess.run(args_run)

def nqsLoad(filename):
    print("DEBUG: Loading file", filename)
    filename_path = dataPath(filename)

    df = pd.read_csv(filename_path, delim_whitespace=True)

    int_cols = ["Dimensions", "Particles", "Hidden-nodes","Metro-steps", "Imposampling", "Analytical", "Epoch", "Interaction"]
    numeric_cols = [col for col in df.columns if col not in int_cols]
    
    for col in numeric_cols:
        df[col] = df[col].astype(float)

    return df

def binaryLoad(filename):
    filename = dataPath(filename)
    with open(filename, "rb") as f:
        f.seek(0, 2)
        num_doubles = f.tell() // 8
        f.seek(0)

        return np.fromfile(f, dtype=np.float64, count=num_doubles)
    
def rLoad(filename):
    filename = dataPath(filename + "_Rs.txt")

    return pd.read_csv(filename, delim_whitespace=True)